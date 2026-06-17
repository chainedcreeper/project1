"""RAG 통합 인터페이스 — 사용자별 인덱스 + 디스크 영속화.

흐름:
    문서 처리 → 청킹 → 임베딩 → faiss → reranker → LLM (수준별).
"""
from . import core
from .core import (
    process_document, process_pdf,
    ask, ask_stream, ask_full, ask_full_stream,
    get_parents, quick_find,
)
from .qa    import generate_qa_pairs, save_qa_pairs
from .state import has_state, load_state, save_state, clear_state, get_meta

__all__ = [
    "core",
    "process_document", "process_pdf",
    "ask", "ask_stream", "ask_full", "ask_full_stream",
    "get_parents", "quick_find",
    "generate_qa_pairs", "save_qa_pairs",
    "has_state", "load_state", "save_state", "clear_state", "get_meta",
]
