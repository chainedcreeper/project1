"""사용자별 RAG 인덱스 상태 관리.

- 메모리 dict 캐시: 같은 학생 반복 호출은 빠름
- 디스크 영속화: 서버 재시작 시 자동 복구
- thread-safe (RLock)

저장 구조:
    rag_indexes/
        <student_id>/
            parents.json
            children.json
            faiss.index
            meta.json       (filename, uploaded_at)
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import faiss

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "rag_indexes")

_lock = threading.RLock()
_cache: dict[str, dict] = {}


def _student_dir(student_id: str) -> str:
    return os.path.join(INDEX_DIR, student_id)


def _index_mtime(student_id: str) -> float:
    p = os.path.join(_student_dir(student_id), "faiss.index")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def has_state(student_id: str) -> bool:
    """디스크 우선 — 멀티 워커 환경에서도 일관."""
    return os.path.exists(os.path.join(_student_dir(student_id), "faiss.index"))


def save_state(student_id: str, parents: list[dict], children: list[dict], index, *, filename: str = "") -> None:
    """메모리 + 디스크 동시 저장."""
    d = _student_dir(student_id)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "parents.json"),  "w", encoding="utf-8") as f:
        json.dump(parents,  f, ensure_ascii=False)
    with open(os.path.join(d, "children.json"), "w", encoding="utf-8") as f:
        json.dump(children, f, ensure_ascii=False)
    with open(os.path.join(d, "meta.json"),     "w", encoding="utf-8") as f:
        json.dump({"filename": filename, "uploaded_at": time.time()}, f, ensure_ascii=False)

    faiss.write_index(index, os.path.join(d, "faiss.index"))

    with _lock:
        _cache[student_id] = {
            "parents":  parents,
            "children": children,
            "index":    index,
            "_mtime":   _index_mtime(student_id),
        }


def load_state(student_id: str) -> dict | None:
    """메모리 → 디스크 순으로 로드. 디스크 mtime 이 캐시보다 새로우면 강제 재로드.
    멀티 워커 환경에서 다른 워커가 새 자료 올린 걸 감지."""
    disk_mtime = _index_mtime(student_id)
    if not disk_mtime:
        return None

    with _lock:
        cached = _cache.get(student_id)
        if cached and cached.get("_mtime", 0) >= disk_mtime:
            return cached

    d = _student_dir(student_id)
    idx_path = os.path.join(d, "faiss.index")
    with open(os.path.join(d, "parents.json"),  encoding="utf-8") as f:
        parents = json.load(f)
    with open(os.path.join(d, "children.json"), encoding="utf-8") as f:
        children = json.load(f)
    index = faiss.read_index(idx_path)

    state = {"parents": parents, "children": children, "index": index, "_mtime": disk_mtime}
    with _lock:
        _cache[student_id] = state
    return state


def get_meta(student_id: str) -> dict:
    """업로드 파일명 / 시간 등 메타."""
    meta_path = os.path.join(_student_dir(student_id), "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def clear_state(student_id: str) -> None:
    """메모리 + 디스크 모두 삭제."""
    with _lock:
        _cache.pop(student_id, None)
    d = _student_dir(student_id)
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
