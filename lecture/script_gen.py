"""슬라이드 스크립트 생성 — 풍부한 본문 + 자세한 나레이션.

흐름:
    강의 자료 → qwen3:8b → 도입 / 본문(섹션별) / 정리 구조의 슬라이드 N장.

각 슬라이드 필드:
    title     — 슬라이드 제목 (20자)
    headline  — 핵심 한 문장 (한 줄 강조, 50자)
    bullets   — 세부 항목 3~5개 (각 40~70자)
    narration — 강사가 자세히 풀어 설명 (400~600자)
"""
from __future__ import annotations

import json
import re

from llm import ask_qwen


SCRIPT_PROMPT = """\
업로드된 강의 자료를 풍부하게 풀어 설명하는 강의 슬라이드를 만들어라.

분량 규칙:
- 강의 자료가 짧아도 최소 6장 만들어라
- 강의 자료가 길면 최대 12장까지 늘려라
- 짧은 자료여도 비유와 예시를 추가해서 깊이있게 설명해라

구조:
- 1장:   도입 (강의 주제 + 학습 목표 3~5개)
- 본문: 핵심 개념을 섹션별로 분해해서 자세히 설명 (4~10장)
- 마지막: 정리 (배운 것 요약 + 시험 대비 포인트)

각 슬라이드는 반드시 다음 4개 필드:

  title     — 슬라이드 제목 (한 줄, 20자 이내)
  headline  — 그 슬라이드의 핵심을 한 문장으로 (40~60자, 코랄색으로 강조 표시됨)
  bullets   — 세부 항목 3~5개. 각 항목은 반드시 50~90자의 실제 설명
              · "X는 Y이다" 정의 X — "X는 Y이고 그 이유는 Z이다" 정도로 풀어라
              · 한 단어/구절 금지. 완성된 문장으로
  narration — 강사가 학생 앞에서 칠판 설명하듯 500~700자
              · 비유 1~2개, 일상 예시 1~2개 포함
              · 슬라이드 화면에 없는 깊은 디테일을 narration 에서 보충
              · 슬라이드 텍스트를 그대로 읽지 말고 더 자세히 풀어 설명
              · 다음 슬라이드로 자연스럽게 연결되는 한 문장 마무리

반드시 아래 JSON 배열 형식으로만 출력하라 (다른 텍스트 절대 금지):

[
  {"title": "...", "headline": "...", "bullets": ["...", "...", "..."], "narration": "..."},
  ...
]
"""

_FENCE_RE      = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*?(?=\[|\{|$)", re.DOTALL)  # 닫히지 않은 think
_ARRAY_RE      = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE     = re.compile(r"\{[^{}]*?\}", re.DOTALL)  # 단일 객체 추출 fallback

_REQUIRED = ("title", "bullets", "narration")  # headline 은 optional 폴백 가능


def _extract_json_array(raw: str) -> list[dict]:
    # 1단계 — 정상 닫힌 think 제거
    text = _THINK_RE.sub("", raw)
    # 2단계 — 닫히지 않은 think 도 제거 (열린 채로 끝났거나 [ { 직전까지)
    text = _THINK_OPEN_RE.sub("", text)
    # 3단계 — 코드 펜스 제거
    text = _FENCE_RE.sub("", text)

    # 1차 시도: [ ... ] 배열 찾기
    m = _ARRAY_RE.search(text)
    if m:
        raw_json = m.group(0)
        # 1-A: 그대로
        try:
            return json.loads(raw_json, strict=False)
        except json.JSONDecodeError:
            pass
        # 1-B: trailing comma 제거
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw_json), strict=False)
        except json.JSONDecodeError:
            pass
        # 1-C: 마지막 잘린 객체 제거 후 재시도 — 마지막 완성된 } 까지만
        last_brace = raw_json.rfind("}")
        if last_brace > 0:
            truncated = raw_json[:last_brace + 1] + "]"
            try:
                return json.loads(truncated, strict=False)
            except json.JSONDecodeError:
                pass
        # 1-D: json-repair
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw_json), strict=False)
        except Exception:
            pass

    # 2차 시도: 단일 객체들 모아서 배열로 — `{...} {...}` 같이 나열된 경우
    objects = []
    for om in _OBJECT_RE.finditer(text):
        try:
            objects.append(json.loads(om.group(0), strict=False))
        except json.JSONDecodeError:
            continue
    if objects:
        return objects

    # 3차 시도: json-repair 전체 텍스트
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        if repaired:
            result = json.loads(repaired, strict=False)
            if isinstance(result, list) and result:
                return result
            if isinstance(result, dict):
                return [result]
    except Exception:
        pass

    raise ValueError(f"응답에서 JSON 배열을 찾을 수 없음 (앞 200자: {text[:200]!r})")


def _validate(slides: list[dict]) -> list[dict]:
    """필드 누락은 폴백으로 채워서 슬라이드 살리기 (LLM 응답 잘림 가드)."""
    if not isinstance(slides, list) or not slides:
        raise ValueError("슬라이드 리스트가 비어있음")
    # nested list 풀기: [[{...}, {...}], ...] → [{...}, {...}, ...]
    flat = []
    for s in slides:
        if isinstance(s, list):
            flat.extend(s)
        else:
            flat.append(s)
    slides = flat
    cleaned: list[dict] = []
    for idx, s in enumerate(slides, 1):
        if not isinstance(s, dict):
            # dict 아니면 폴백 슬라이드 만들기 — 영상 자체는 살림
            cleaned.append({
                "index":     idx,
                "title":     f"슬라이드 {idx}",
                "headline":  "",
                "bullets":   [str(s)[:90]] if s else [f"슬라이드 {idx} 내용"],
                "narration": str(s)[:600] if s else f"슬라이드 {idx}",
            })
            continue
        title     = str(s.get("title", "")).strip() or f"슬라이드 {idx}"
        headline  = str(s.get("headline", "")).strip()
        raw_bull  = s.get("bullets", [])
        if not isinstance(raw_bull, list):
            raw_bull = []
        bullets   = [str(b).strip() for b in raw_bull if str(b).strip()]
        narration = str(s.get("narration", "")).strip()

        # 폴백: 비었으면 다른 필드에서 합성
        if not bullets:
            bullets = [headline] if headline else [title]
        if not narration:
            narration = f"{title}. " + " ".join(bullets)

        cleaned.append({
            "index":     idx,
            "title":     title,
            "headline":  headline,
            "bullets":   bullets,
            "narration": narration,
        })
    return cleaned


_MIN_SLIDES_OK     = 3      # 이거 미만이면 1회 retry
_MIN_AVG_NARRATION = 100    # 평균 narration 자수. 이거 미만이면 1회 retry


def _is_suspicious(slides: list[dict]) -> tuple[bool, str]:
    """11초 영상 사건 가드 — 슬라이드 수 / narration 길이 임계 검증.
    True 면 사용 가능, False 면 retry 추천."""
    if len(slides) < _MIN_SLIDES_OK:
        return True, f"슬라이드 {len(slides)}장 (최소 {_MIN_SLIDES_OK}장 미달)"
    avg = sum(len(s.get("narration", "")) for s in slides) / max(len(slides), 1)
    if avg < _MIN_AVG_NARRATION:
        return True, f"평균 narration {avg:.0f}자 (최소 {_MIN_AVG_NARRATION}자 미달)"
    return False, ""


def generate_script(context: str, level_info: dict | None = None) -> list[dict]:
    """강의 자료 텍스트 → 슬라이드 스크립트 리스트.

    각 원소: {index, title, headline, bullets, narration}
    sanity check 통과 못 하면 1회 retry — 11초 영상 사건 방지.
    """
    def _one_shot(prompt_suffix: str = "") -> list[dict]:
        raw = ask_qwen(context, SCRIPT_PROMPT + prompt_suffix, level_info, prefer_json=True)
        return _validate(_extract_json_array(raw))

    try:
        slides = _one_shot()
    except Exception as e:
        snippet = ""    # 1차 실패 — 즉시 retry
        try:
            slides = _one_shot("\n\n반드시 6장 이상 만들고 각 narration 500자 이상 작성하라.")
        except Exception as e2:
            raise RuntimeError(f"슬라이드 스크립트 생성 실패: {e2}") from e2

    bad, why = _is_suspicious(slides)
    if bad:
        # 1회 retry — 명시적 minimum 강조
        try:
            retry = _one_shot(f"\n\n주의: 이전 응답이 {why}. 반드시 6장 이상, narration 각 500자 이상 작성하라.")
            r_bad, r_why = _is_suspicious(retry)
            if not r_bad:
                return retry
            # retry 도 부족하면 더 나은 쪽 채택
            if len(retry) > len(slides):
                return retry
        except Exception:
            pass
        # retry 도 실패 — 명시적 경고 후 원본 사용 (사용자에게 부족함 알림)
        print(f"⚠ generate_script sanity check 실패 (1회 retry 후): {why}", flush=True)
    return slides
