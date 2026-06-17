"""LLM 호출 동시성 게이트.

Ollama 는 GPU 1장 → 동시 요청 직렬화가 사실상 강제됨.
여기서 일관된 큐를 통해 통과시키고, 대기자에게는 순번을 노출.

핵심:
- `LLM_CONCURRENCY` (기본 1) — 동시에 흘릴 요청 수
- 대기 카운터 — `position()` 으로 클라에 노출
- 정확히 하나의 락(Semaphore) 으로 단순화 → 데드락 위험 0

사용:
    with llm_gate.acquire() as ticket:
        # ticket.position : 0(처리중) 1, 2, ...
        # ticket.wait_seconds : 들어와서 락 잡기까지 걸린 시간
        ...
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager


_CONCURRENCY     = int(os.getenv("LLM_CONCURRENCY",     "1"))
_DEFAULT_TIMEOUT = float(os.getenv("LLM_QUEUE_TIMEOUT", "180"))   # 큐 대기 한도(초)


class GateBusy(RuntimeError):
    """게이트 대기 타임아웃. 호출자가 503 으로 변환할 수 있도록 별도 타입."""
    pass


class _Ticket:
    __slots__ = ("position", "wait_seconds")

    def __init__(self, position: int, wait_seconds: float):
        self.position     = position
        self.wait_seconds = wait_seconds


class LLMGate:
    def __init__(self, concurrency: int = 1):
        self._sem      = threading.Semaphore(concurrency)
        self._lock     = threading.Lock()
        self._waiting  = 0    # 대기 중인 호출자 수 (락 못 잡음)
        self._active   = 0    # 처리 중인 호출자 수
        self.capacity  = concurrency

    def position(self) -> int:
        """지금 새로 들어오면 몇 번째인지. 0이면 즉시 처리."""
        with self._lock:
            # active 가 capacity 미만이면 0, 아니면 (active - capacity) + waiting + 1
            if self._active < self.capacity:
                return 0
            return self._waiting + 1

    @contextmanager
    def acquire(self, timeout: float | None = None):
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        with self._lock:
            self._waiting += 1
        t0 = time.time()
        acquired = False
        try:
            acquired = self._sem.acquire(timeout=timeout)
            if not acquired:
                raise GateBusy(f"LLM 대기 시간 초과 ({timeout:.0f}초). 지금 너무 붐빔 — 잠시 후 다시 시도해줘.")
            elapsed = time.time() - t0
            with self._lock:
                self._waiting -= 1
                self._active  += 1
            yield _Ticket(position=0, wait_seconds=elapsed)
        finally:
            if acquired:
                with self._lock:
                    if self._active > 0:
                        self._active -= 1
                self._sem.release()
            else:
                # 타임아웃: waiting 만 감소, sem 은 release 안 함
                with self._lock:
                    if self._waiting > 0:
                        self._waiting -= 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "capacity": self.capacity,
                "active":   self._active,
                "waiting":  self._waiting,
            }


# 전역 단일 인스턴스
llm_gate = LLMGate(concurrency=_CONCURRENCY)
