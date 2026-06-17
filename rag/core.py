"""RAG 코어 — 사용자별 인덱스 + 검색 + 답변.

모든 함수가 student_id 를 받음. 전역 상태 없음.
인덱스는 rag.state 가 메모리/디스크 자동 관리.
"""
from __future__ import annotations

import numpy as np

from document import extract_text
from llm      import ask_qwen, ask_qwen_stream

from .chunker   import split_text
from .embedding import create_embeddings, model
from .vector    import build_index
from .reranker  import rerank
from .state     import save_state, load_state, has_state


def process_document(doc_path: str, student_id: str, *, filename: str = "") -> dict:
    """문서 경로 → 추출 → 청킹 → 임베딩 → faiss → 디스크/메모리 저장."""
    pages = extract_text(doc_path)
    if not pages:
        raise ValueError("문서에서 텍스트를 추출할 수 없습니다. 이미지 전용이거나 손상된 파일일 수 있습니다.")

    parents, children = split_text(pages)
    if not children:
        raise ValueError("문서 내용이 너무 짧아 처리할 수 없습니다.")

    embeddings = create_embeddings([c["text"] for c in children])
    index      = build_index(embeddings)
    save_state(student_id, parents, children, index, filename=filename)

    return {
        "pages":    len(pages),
        "parents":  len(parents),
        "children": len(children),
    }


# 이전 이름 호환
process_pdf = process_document


def _require_state(student_id: str) -> dict:
    state = load_state(student_id)
    if state is None:
        raise RuntimeError("문서가 업로드되지 않았습니다.")
    return state


def _get_context(question: str, student_id: str, *, initial_k: int = 20, final_k: int = 3, skip_rerank: bool = False) -> str:
    state    = _require_state(student_id)
    parents  = state["parents"]
    children = state["children"]
    index    = state["index"]

    # skip_rerank — FAISS top-k 만 사용. CPU reranker (~1~2초) 건너뜀 → 채팅 빠르게
    if skip_rerank:
        k = min(final_k * 2, len(children))   # 후보 좀 더 잡아서 페이지 다양성
    else:
        k = min(initial_k, len(children))
    q_emb = model.encode([question], normalize_embeddings=True)
    _, I  = index.search(np.array(q_emb, dtype="float32"), k=k)

    candidates = [children[idx] for idx in I[0] if idx < len(children)]
    if skip_rerank:
        top_children = candidates[:final_k]
    else:
        top_children = rerank(question, candidates, top_k=min(final_k, len(candidates)))

    context, seen = "", set()
    for child in top_children:
        pid = child["parent_id"]
        if pid not in seen:
            seen.add(pid)
            p = parents[pid]
            context += f"[p.{p['page']}] {p['text']}\n\n"
    return context


def _full_context(student_id: str, max_chars: int = 6000) -> str:
    state   = _require_state(student_id)
    parents = state["parents"]
    return "\n\n".join(f"[p.{p['page']}] {p['text']}" for p in parents)[:max_chars]


# ── 빠른 키워드 검색 (Ctrl-F 대용) ────────────────────
# LLM 안 거침. parents 전체 순회 매칭. sub-second.
import re as _re

_TOK_RE = _re.compile(r"[가-힣A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text or "")]


def _score(text_lc: str, tokens: list[str], phrase: str) -> int:
    """매칭 점수.
    - 구문 통째 일치 : 토큰당 +5
    - 토큰 개별 일치 : +1 (대소문자 무시)
    """
    if not tokens:
        return 0
    s = 0
    if phrase and len(phrase) >= 2 and phrase in text_lc:
        s += 5 * max(1, len(tokens))
    for t in tokens:
        if len(t) >= 2 and t in text_lc:
            s += 1
    return s


def _snippet(text: str, phrase: str, tokens: list[str], window: int = 80) -> str:
    """매칭 위치 기준 ±window 자르고 마커 추가."""
    text_lc = text.lower()
    pos = text_lc.find(phrase) if phrase and len(phrase) >= 2 else -1
    if pos < 0:
        for t in tokens:
            if len(t) >= 2:
                pos = text_lc.find(t)
                if pos >= 0:
                    break
    if pos < 0:
        return text[:window * 2]
    start = max(0, pos - window)
    end   = min(len(text), pos + window)
    snip  = text[start:end]
    if start > 0:
        snip = "…" + snip
    if end < len(text):
        snip = snip + "…"
    return snip


_FUZZY_MIN_RESULTS = 3       # 정확 매칭이 이거 미만이면 임베딩 폴백 발동
_FUZZY_SCORE_FLOOR = 0.45    # 코사인 유사도 임계값 (BGE-m3, normalize=True)


def _fuzzy_snippet(text: str, window: int = 160) -> str:
    """폴백 결과는 매칭 위치를 찾을 수 없음 — 앞부분 잘라서 보여줌."""
    snip = text[:window * 2]
    if len(text) > window * 2:
        snip += "…"
    return snip


def quick_find(query: str, student_id: str, *, top_k: int = 5) -> list[dict]:
    """parents 에서 키워드 매칭. 정확 매칭 결과 부족 시 임베딩 폴백으로 의미·동의어·오타 보강."""
    state    = _require_state(student_id)
    parents  = state["parents"]
    children = state["children"]
    index    = state["index"]
    if not query or not query.strip():
        return []

    phrase = query.strip().lower()
    tokens = _tokenize(query)

    # 1차: 정확 substring 매칭
    exact = []
    for pid, p in enumerate(parents):
        text    = p["text"]
        text_lc = text.lower()
        s = _score(text_lc, tokens, phrase)
        if s <= 0:
            continue
        exact.append({
            "parent_id": pid,
            "page":      p["page"],
            "score":     s,
            "snippet":   _snippet(text, phrase, tokens),
            "match":     "exact",
        })
    exact.sort(key=lambda r: -r["score"])

    # 정확 결과 충분하면 그대로
    if len(exact) >= _FUZZY_MIN_RESULTS:
        return exact[:top_k]

    # 2차: 임베딩 폴백 — 동의어 / 오타 / 띄어쓰기 변형 잡기
    try:
        q_emb     = model.encode([query.strip()], normalize_embeddings=True)
        k_search  = min(max(top_k * 4, 12), len(children)) if children else 0
        if k_search > 0:
            sims, idxs = index.search(np.array(q_emb, dtype="float32"), k=k_search)
            seen_pids  = {r["parent_id"] for r in exact}
            fuzzy      = []
            for sim, idx in zip(sims[0], idxs[0]):
                if idx < 0 or idx >= len(children):
                    continue
                if sim < _FUZZY_SCORE_FLOOR:
                    continue
                pid = children[idx]["parent_id"]
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                p = parents[pid]
                fuzzy.append({
                    "parent_id": pid,
                    "page":      p["page"],
                    "score":     float(sim),
                    "snippet":   _fuzzy_snippet(p["text"]),
                    "match":     "fuzzy",
                })
            return (exact + fuzzy)[:top_k]
    except Exception:
        # 임베딩 폴백 실패해도 정확 매칭 결과는 살림
        pass

    return exact[:top_k]


def get_parents(student_id: str) -> list[dict]:
    """QA 생성 등에서 직접 parents 접근용."""
    return _require_state(student_id)["parents"]


# ── 답변 생성 ────────────────────────────────────────
# FAST_CHAT — 채팅 빠르게 (reranker 건너뜀, 응답 토큰 캡). 환경변수 0 으로 끄면 정확도 우선.
import os as _os
_FAST_CHAT = _os.getenv("FAST_CHAT", "1") == "1"


def ask(question: str, student_id: str, level_info=None):
    return ask_qwen(_get_context(question, student_id, skip_rerank=_FAST_CHAT), question, level_info)


def ask_stream(question: str, student_id: str, level_info=None):
    """채팅용 — 짧고 빠르게. 컨텍스트는 FAISS 직접 (reranker 건너뜀)."""
    ctx = _get_context(question, student_id, skip_rerank=_FAST_CHAT)
    yield from ask_qwen_stream(ctx, question, level_info, max_tokens=2048, ctx_size=8192)


def ask_full(question: str, student_id: str, max_tokens: int = 6000):
    return ask_qwen(_full_context(student_id, max_tokens), question)


def ask_full_stream(question: str, student_id: str, max_tokens: int = 6000):
    yield from ask_qwen_stream(_full_context(student_id, max_tokens), question)
