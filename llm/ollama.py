"""Ollama LLM 호출. 기본 qwen3:8b · 고급 탐색은 qwen3:32b 로 토글."""
import os
import json
import re

import requests

from .gateway import llm_gate, gate_for

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# 수준별 라우팅 — 입문/중급은 8B 강제 (어차피 큰 모델 필요 없는 질문 위주)
# 심화만 사용자가 고른 모델 사용. 32B GPU 부담 감소.
LIGHT_MODEL  = os.getenv("LIGHT_MODEL",  "qwen3:8b")
AUTO_ROUTE   = os.getenv("AUTO_ROUTE_BY_LEVEL", "1") == "1"


def _resolve_model(level_info):
    """수준별 라우팅. 입문/중급 → LIGHT_MODEL. 심화 또는 정보 없음 → 사용자 선택."""
    if not AUTO_ROUTE or not level_info:
        return OLLAMA_MODEL
    label = (level_info.get("label") or "").strip()
    if label in ("입문", "중급"):
        return LIGHT_MODEL
    return OLLAMA_MODEL

_LEVEL_GUIDE = {
    "입문": "학생은 이 분야가 처음이다. 전문 용어를 최대한 피하고, 일상적인 비유와 쉬운 예시로 설명해라.",
    "중급": "학생은 기본 개념을 알고 있다. 원리와 이유를 중심으로 설명하고 적절한 예시를 포함해라.",
    "심화": "학생은 개념을 깊이 이해하고 있다. 세부 메커니즘, 예외 사항, 다른 개념과의 연관성까지 다뤄라.",
}

_BASE_SYSTEM = (
    "너는 대학 강의 전문 AI 튜터다. "
    "반드시 주어진 강의 문서를 근거로만 답변하고, 문서에 없는 내용은 추측하지 마라. "
    "답변은 반드시 한국어로 작성하라 — 영어로 사고 과정을 노출하지 말고 결과만 출력하라. "
    "답변은 체계적이고 명확하게 작성해라."
)


def _system_prompt(level_info=None):
    if level_info and level_info.get("label"):
        guide = _LEVEL_GUIDE.get(level_info["label"], "")
        if guide:
            return f"{_BASE_SYSTEM}\n\n[학생 수준: {level_info['label']}] {guide}"
    return _BASE_SYSTEM


def _messages(context, question, level_info=None):
    # /no_think — qwen3 시리즈 매직 워드. reasoning 노출 차단
    return [
        {"role": "system", "content": _system_prompt(level_info)},
        {"role": "user",   "content": f"[강의 문서]\n{context}\n\n[요청]\n{question}\n\n/no_think"},
    ]


def _post_ollama(context, question, level_info=None, stream=False, *, max_tokens=None, ctx_size=None):
    """게이트 잡기 전의 raw POST. 호출자가 게이트 잡고 있어야 함.
    max_tokens / ctx_size 명시 시 우선 (채팅 같은 짧은 응답용)."""
    model = _resolve_model(level_info)
    model_lc = model.lower()
    is_large = any(k in model_lc for k in ("32b", "30b", "70b", "27b", "exaone3.5:32"))

    if max_tokens is not None:
        num_predict = max_tokens
    else:
        num_predict = 1024 if is_large else 16384
    if ctx_size is not None:
        num_ctx = ctx_size
    else:
        num_ctx = 8192 if is_large else 16384

    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    model,
            "messages": _messages(context, question, level_info),
            "stream":   stream,
            "think":    False,
            "options": {
                "num_predict": num_predict,
                "num_ctx":     num_ctx,
            },
        },
        timeout=900,
        stream=stream,
    )
    resp.raise_for_status()
    return resp


def _strip_reasoning_prefix(text, prefer_json=False):
    """영어 reasoning prefix (Okay, Let me, ...) 제거.
    prefer_json=True 면 [ 또는 { 위치 우선 (JSON 구조 보존).
    아니면 첫 한국어 또는 형식 마커부터."""
    if not text:
        return text
    if prefer_json:
        # JSON 응답 기대 — [ 또는 { 위치까지 자름 (한국어 prefix 보다 우선)
        markers = [text.find(c) for c in ("[", "{")]
        markers = [i for i in markers if i >= 0]
        if markers:
            cut = min(markers)
            return text[cut:] if cut > 0 else text
        return text
    m = _HANGUL_RE.search(text)
    markers = [text.find(c) for c in ("1.", "[", "▶", "#", "▷", "✓")]
    markers = [i for i in markers if i >= 0]
    cut = m.start() if m else (min(markers) if markers else -1)
    return text[cut:] if cut > 0 else text


def ask_qwen(context, question, level_info=None, *, prefer_json=False, max_tokens=None, ctx_size=None):
    model = _resolve_model(level_info)
    with gate_for(model).acquire():
        raw = _post_ollama(context, question, level_info, stream=False, max_tokens=max_tokens, ctx_size=ctx_size).json()["message"]["content"]
    return _strip_reasoning_prefix(raw, prefer_json=prefer_json)


_HANGUL_RE = re.compile(r"[가-힣]")


def ask_qwen_stream(context, question, level_info=None, *, max_tokens=None, ctx_size=None):
    """영어 reasoning prefix (Okay, Let me ...) 자동 필터.
    첫 한국어 글자 또는 형식 마커(`1.`, `[`, `▶`, `#`) 가 나오기 전까지 버퍼링.
    스트림이 끝날 때까지 LLM 게이트 보유 — GPU 1장 직렬화 유지.
    """
    buf       = ""
    started   = False
    model     = _resolve_model(level_info)
    with gate_for(model).acquire():
        for line in _post_ollama(context, question, level_info, stream=True, max_tokens=max_tokens, ctx_size=ctx_size).iter_lines():
            if not line:
                continue
            token = json.loads(line).get("message", {}).get("content", "")
            if not token:
                continue
            if started:
                yield token
                continue
            buf += token
            m = _HANGUL_RE.search(buf)
            markers = [buf.find(c) for c in ("1.", "[", "▶", "#", "▷", "✓")]
            markers = [i for i in markers if i >= 0]
            cut = m.start() if m else (min(markers) if markers else -1)
            if cut >= 0:
                yield buf[cut:]
                buf = ""
                started = True
            elif len(buf) > 2000:
                yield buf
                buf = ""
                started = True
