"""Edge-TTS 래퍼.

각 슬라이드 narration → mp3 + 단어/문장 타임스탬프.
타임스탬프는 자막 SRT 생성·슬라이드 노출 시간 계산에 사용.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess

import edge_tts
import imageio_ffmpeg

DEFAULT_VOICE = "ko-KR-SunHiNeural"  # 가장 자연스러운 여성 한국어
_HNS_TO_SEC = 1 / 10_000_000  # Edge-TTS 시간 단위는 100ns
_MIN_DURATION = 1.0  # 슬라이드 최소 노출 시간 (초)
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_DURATION_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


async def _synthesize_async(text: str, mp3_path: str, voice: str) -> list[dict]:
    """Edge-TTS 호출 → mp3 + 타임스탬프. WordBoundary 없으면 SentenceBoundary 도 받음
    (한국어는 단어 단위 boundary 가 안 옴 — 문장 단위라도 잡아야 자막·길이 동기화됨)."""
    communicate = edge_tts.Communicate(text, voice)
    word_times: list[dict] = []
    with open(mp3_path, "wb") as f:
        async for chunk in communicate.stream():
            ctype = chunk["type"]
            if ctype == "audio":
                f.write(chunk["data"])
            elif ctype in ("WordBoundary", "SentenceBoundary"):
                word_times.append({
                    "text":  chunk["text"],
                    "start": chunk["offset"] * _HNS_TO_SEC,
                    "end":   (chunk["offset"] + chunk["duration"]) * _HNS_TO_SEC,
                })
    return word_times


def synthesize(text: str, mp3_path: str, voice: str = DEFAULT_VOICE) -> list[dict]:
    """텍스트 → mp3 저장. 반환: 단어/문장별 [{text, start, end}]."""
    return asyncio.run(_synthesize_async(text, mp3_path, voice))


def _probe_duration(mp3_path: str) -> float:
    """mp3 실제 길이(초) — imageio_ffmpeg 의 ffmpeg 사용 (ffprobe 따로 안 깔려도 동작)."""
    try:
        r = subprocess.run(
            [_FFMPEG, "-i", mp3_path, "-f", "null", "-"],
            capture_output=True, timeout=15,
        )
        # ffmpeg 은 진행 정보를 stderr 로 보냄. 마지막 time= 가 실제 길이
        stderr = r.stderr.decode("utf-8", errors="replace")
        m = None
        for m in _DURATION_RE.finditer(stderr):
            pass    # 마지막 매치
        if m:
            h, mn, s = m.groups()
            return int(h) * 3600 + int(mn) * 60 + float(s)
    except Exception:
        pass
    return 0.0


async def _synthesize_one(s: dict, out_dir: str, voice: str) -> dict:
    path = os.path.join(out_dir, f"slide_{s['index']:02d}.mp3")
    narration = (s.get("narration") or "").strip()
    if not narration:
        narration = s.get("title", "")
    word_times = await _synthesize_async(narration, path, voice) if narration else []

    # 실제 mp3 길이를 우선 — Edge-TTS 의 boundary 끝 시간은 trailing silence 미포함 가능
    probed = _probe_duration(path)
    if probed >= _MIN_DURATION:
        duration = probed
    elif word_times:
        duration = word_times[-1]["end"]
    else:
        duration = _MIN_DURATION
    if duration < _MIN_DURATION:
        duration = _MIN_DURATION

    s["audio_path"] = path
    s["word_times"] = word_times
    s["duration"]   = duration
    return s


async def _synthesize_all_async(slides: list[dict], out_dir: str, voice: str, on_progress=None) -> list[dict]:
    """슬라이드 전체 병렬 TTS — Edge-TTS 동시 호출. asyncio.gather + 진행 콜백."""
    total = len(slides)
    completed = 0
    results: list[dict] = [None] * total

    async def wrapped(idx, slide):
        nonlocal completed
        out = await _synthesize_one(slide, out_dir, voice)
        results[idx] = out
        completed += 1
        if on_progress:
            try: on_progress(completed, total)
            except Exception: pass

    await asyncio.gather(*(wrapped(i, s) for i, s in enumerate(slides)))
    return results


def synthesize_slides(
    slides: list[dict], out_dir: str, voice: str = DEFAULT_VOICE,
    on_progress=None,
) -> list[dict]:
    """슬라이드별 TTS — 병렬(asyncio.gather)로 동시 호출.

    Edge-TTS 가 외부 MS API 라서 동시 호출 가능 → 직렬 2~3분 → 20~30초.
    각 slide 에 {audio_path, word_times, duration} 채워 반환.
    on_progress(current, total): 슬라이드 끝날 때마다 호출.
    """
    os.makedirs(out_dir, exist_ok=True)
    return asyncio.run(_synthesize_all_async(slides, out_dir, voice, on_progress))
