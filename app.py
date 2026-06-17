import os as _os
_env_path = _os.path.join(_os.path.dirname(__file__), "local.env")
if _os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _os.environ.setdefault(_k.strip(), _v.strip())
_os.environ.setdefault("SURYA_DEVICE", "cpu")

import asyncio
import json
import os
import queue
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from rag import (
    process_document, ask_stream, ask_full_stream,
    generate_qa_pairs, save_qa_pairs,
    get_parents, has_state, get_meta,
    quick_find,
)
from rag.core import _get_context, _full_context
from document import SUPPORTED_EXTS
from student import (
    init_db, log_interaction, should_train, export_student_data, get_student,
    init_level_db, assess_and_update, get_student_level,
    init_notes_db, add_note, list_notes, delete_note, update_note_category,
    init_schedule_db, add_schedule, list_schedules, update_schedule, delete_schedule,
)
from llm import (
    personal_model_exists, ask_personal_stream,
    ask_server_stream, is_server_available,
)
from llm.gateway import llm_gate
from auth import (
    init_auth_db, register_user, authenticate_user, create_token,
    get_current_user, get_user_from_query,
)
from lecture import generate_lecture

init_db()
init_auth_db()
init_level_db()
init_notes_db()
init_schedule_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app      = FastAPI()
# 두 풀로 분리:
#  - executor      : 검색·임베딩·디스크 IO. /find, /document-status, 업로드 등.
#  - llm_executor  : /chat /analyze /generate-video produce. 게이트 대기 잡혀도
#                    위 풀은 멀쩡해서 빠른 검색 응답 가능.
executor     = ThreadPoolExecutor(max_workers=int(os.getenv("WORKER_POOL", "8")))
llm_executor = ThreadPoolExecutor(max_workers=int(os.getenv("LLM_POOL",    "8")))


# ── 모델 ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    student_id: str
    name:       str
    password:   str


class ChatMessage(BaseModel):
    message: str


class FindRequest(BaseModel):
    query: str
    top_k: int = 5


class NoteRequest(BaseModel):
    content:  str
    category: str = "general"   # general / exam / important
    source:   str = "manual"


class NoteCategory(BaseModel):
    category: str


class ScheduleRequest(BaseModel):
    date:  str          # YYYY-MM-DD
    title: str
    note:  str = ""


class ScheduleUpdate(BaseModel):
    date:  str | None = None
    title: str | None = None
    note:  str | None = None
    done:  bool | None = None


# ── 인증 ────────────────────────────────────────────

@app.post("/register")
async def register(req: RegisterRequest):
    register_user(req.student_id, req.name, req.password)
    return {"status": "ok", "message": f"{req.name}님 등록 완료"}


@app.post("/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user  = authenticate_user(form.username, form.password)
    token = create_token(user["student_id"], user["name"])
    return {"access_token": token, "token_type": "bearer", "name": user["name"]}


# ── 화면 ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/favicon.ico")
async def favicon():
    path = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=204)


# ── 문서 업로드 (사용자별 인덱스) ───────────────────

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    student_id = user["student_id"]

    filename = file.filename or "uploaded"
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식: {ext or '없음'} (지원: {', '.join(SUPPORTED_EXTS)})",
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(
            executor, lambda: process_document(tmp_path, student_id, filename=filename)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # 새 자료 업로드됨 — 메모리 job 만 폐기 (옛 mp4 는 강의자료별 이름으로 보관됨)
    with _video_lock:
        _video_jobs.pop(student_id, None)
    # 사전처리 캐시도 폐기 — 새 자료에 옛 요약·개념 쓰면 안 됨
    with _pre_lock:
        _pre_cache.pop(student_id, None)

    return {"status": "ok", "filename": filename, **info}


@app.get("/document-status")
async def document_status(user=Depends(get_current_user)):
    sid = user["student_id"]
    if not has_state(sid):
        return {"ready": False}
    return {"ready": True, **get_meta(sid)}


# ── 분석 (요약/개념/시험) ──────────────────────────

PROMPTS = [
    ("summary", """업로드된 강의자료 전체를 분석해서 아래 형식으로 요약해줘.

1. 강의 주제 (한 줄)
2. 학습 목표 (3~5개, 번호 목록)
3. 섹션별 핵심 내용 (강의 흐름 순서대로)
4. 중요 포인트 (반드시 기억해야 할 것 5개 이내)

조건:
- 강의 흐름을 유지하며 요약
- 전문 용어는 괄호 안에 간단히 설명
- 학생이 시험 전 복습에 바로 쓸 수 있는 수준"""),

    ("concepts", """업로드된 강의자료에서 핵심 개념을 모두 추출해서 아래 형식으로 정리해줘.

각 개념마다:
▶ 개념명
  - 정의: 한 문장으로
  - 핵심 특징: 2~3가지
  - 관련 개념: (있으면 명시)

조건:
- 중요도 높은 순서로 정렬
- 전문 용어는 쉬운 설명 병기
- 강의에서 강조된 내용 위주로 선별"""),

    ("exam", """업로드된 강의자료를 기반으로 실제 시험에 나올 법한 문제를 만들어줘.

[객관식 5문제]
각 문제 형식:
번호. 문제 내용
① 보기1 ② 보기2 ③ 보기3 ④ 보기4 ⑤ 보기5
정답: 번호
해설: 정답 근거 (강의 내용 기반)

[주관식 5문제]
각 문제 형식:
번호. 문제 내용
모범답안: (핵심 키워드 포함)
채점 기준: 주요 포인트 2~3개

조건:
- 강의 핵심 내용에서 골고루 출제
- 객관식: 명확한 정답, 그럴듯한 오답 포함
- 주관식: 단순 암기보다 개념 이해를 측정"""),
]


async def _analyze_stream(student_id: str):
    loop = asyncio.get_running_loop()
    for type_key, prompt in PROMPTS:
        yield f"data: {json.dumps({'type': type_key, 'event': 'start'}, ensure_ascii=False)}\n\n"
        token_queue: queue.Queue = queue.Queue()
        buf: list[str] = []

        def produce(p=prompt):
            try:
                for token in ask_full_stream(p, student_id):
                    token_queue.put(token)
            except Exception as e:
                token_queue.put(('__error__', str(e)))
            finally:
                token_queue.put(None)

        loop.run_in_executor(llm_executor, produce)
        while True:
            item = await loop.run_in_executor(None, token_queue.get)
            if item is None:
                break
            if isinstance(item, tuple):
                yield f"data: {json.dumps({'type': 'error', 'content': item[1]}, ensure_ascii=False)}\n\n"
                break
            buf.append(item)
            yield f"data: {json.dumps({'type': type_key, 'event': 'token', 'token': item}, ensure_ascii=False)}\n\n"

        # summary / concepts 는 영상 컨텍스트로 재사용 — 캐시에 저장
        full = "".join(buf).strip()
        if full and type_key in ("summary", "concepts"):
            _pre_set(student_id, type_key, full)

        yield f"data: {json.dumps({'type': type_key, 'event': 'done'}, ensure_ascii=False)}\n\n"
    yield 'data: {"type":"done"}\n\n'


@app.get("/analyze")
async def analyze(user=Depends(get_user_from_query)):
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")
    return StreamingResponse(
        _analyze_stream(sid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 사전처리 체인 (요약 → 핵심개념 → 영상) ────────
# 학생 입장: 업로드 → 빠른 요약 → 개념 정리 → 그 둘 기반 영상
# /analyze 는 요약+개념+시험 풀스트림 (학생 능동 요청)
# /summary, /concepts 는 단독 endpoint — 사전처리 체인용

_SUMMARY_PROMPT  = next(p for k, p in PROMPTS if k == "summary")
_CONCEPTS_PROMPT = next(p for k, p in PROMPTS if k == "concepts")
_EXAM_PROMPT     = next(p for k, p in PROMPTS if k == "exam")

# 사전처리 캐시 — 영상 생성 시 raw 문서 대신 요약+개념 사용
_pre_cache: dict[str, dict] = {}   # sid → {"summary": str, "concepts": str}
_pre_lock  = threading.RLock()


def _pre_set(sid: str, key: str, text: str) -> None:
    with _pre_lock:
        d = _pre_cache.setdefault(sid, {})
        d[key] = text


def _pre_get(sid: str) -> dict:
    with _pre_lock:
        return dict(_pre_cache.get(sid, {}))


async def _single_prompt_stream(student_id: str, key: str, prompt: str):
    """공통 단발 SSE — 토큰 스트림 + 종료 시 캐시 저장."""
    loop = asyncio.get_running_loop()
    yield f"data: {json.dumps({'event': 'start', 'type': key}, ensure_ascii=False)}\n\n"
    token_queue: queue.Queue = queue.Queue()
    buf = []

    def produce():
        try:
            for token in ask_full_stream(prompt, student_id):
                token_queue.put(token)
        except Exception as e:
            token_queue.put(('__error__', str(e)))
        finally:
            token_queue.put(None)

    loop.run_in_executor(llm_executor, produce)
    while True:
        item = await loop.run_in_executor(None, token_queue.get)
        if item is None:
            break
        if isinstance(item, tuple):
            yield f"data: {json.dumps({'event': 'error', 'type': key, 'content': item[1]}, ensure_ascii=False)}\n\n"
            return
        buf.append(item)
        yield f"data: {json.dumps({'event': 'token', 'type': key, 'token': item}, ensure_ascii=False)}\n\n"

    full = "".join(buf).strip()
    if full:
        _pre_set(student_id, key, full)
    yield f"data: {json.dumps({'event': 'done', 'type': key}, ensure_ascii=False)}\n\n"


@app.get("/summary")
async def summary_only(user=Depends(get_user_from_query)):
    """빠른 요약 — 학생 첫 인상."""
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")
    return StreamingResponse(
        _single_prompt_stream(sid, "summary", _SUMMARY_PROMPT),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/concepts")
async def concepts_only(user=Depends(get_user_from_query)):
    """핵심 개념 단독."""
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")
    return StreamingResponse(
        _single_prompt_stream(sid, "concepts", _CONCEPTS_PROMPT),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/exam")
async def exam_only(user=Depends(get_user_from_query)):
    """예상 문제 단독."""
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")
    return StreamingResponse(
        _single_prompt_stream(sid, "exam", _EXAM_PROMPT),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 채팅 (수준별 + 라우팅) ─────────────────────────

@app.post("/chat")
async def chat(msg: ChatMessage, user=Depends(get_current_user)):
    student_id = user["student_id"]
    if not has_state(student_id):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")

    async def sse_gen():
        import queue as q_mod
        token_queue = q_mod.Queue()

        def produce():
            buf       = ""
            in_think  = False
            collected = []
            import threading
            threading.Thread(
                target=assess_and_update, args=(student_id, msg.message), daemon=True
            ).start()
            try:
                level_info = get_student_level(student_id)

                if personal_model_exists(student_id):
                    ctx    = _get_context(msg.message, student_id)
                    stream = ask_personal_stream(student_id, ctx, msg.message)
                    source = "personal"
                elif is_server_available():
                    ctx    = _get_context(msg.message, student_id)
                    stream = ask_server_stream(ctx, msg.message)
                    source = "server"
                else:
                    stream = ask_stream(msg.message, student_id, level_info)
                    source = "local"

                for token in stream:
                    buf += token
                    while True:
                        if in_think:
                            end = buf.find("</think>")
                            if end != -1:
                                buf      = buf[end + 8:]
                                in_think = False
                            else:
                                buf = ""
                                break
                        else:
                            start = buf.find("<think>")
                            if start != -1:
                                if buf[:start]:
                                    token_queue.put(buf[:start])
                                    collected.append(buf[:start])
                                buf      = buf[start + 7:]
                                in_think = True
                            else:
                                safe = max(0, len(buf) - 7)
                                if buf[:safe]:
                                    token_queue.put(buf[:safe])
                                    collected.append(buf[:safe])
                                buf = buf[safe:]
                                break
                if buf and not in_think:
                    token_queue.put(buf)
                    collected.append(buf)
            finally:
                full_answer = "".join(collected)
                if full_answer:
                    log_interaction(student_id, msg.message, full_answer, source)
                if should_train(student_id):
                    token_queue.put("__TRAIN_READY__")
                token_queue.put(None)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(llm_executor, produce)

        train_ready = False
        while True:
            token = await loop.run_in_executor(None, token_queue.get)
            if token is None:
                break
            if token == "__TRAIN_READY__":
                train_ready = True
                continue
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True, 'train_ready': train_ready}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 빠른 키워드 검색 (LLM 안 거침, sub-second) ─────

@app.post("/find")
async def find(req: FindRequest, user=Depends(get_current_user)):
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        executor, lambda: quick_find(req.query, sid, top_k=max(1, min(20, req.top_k))),
    )
    return {"query": req.query, "results": results}


# ── LLM 게이트 상태 (동시 접속 모니터링) ────────────

@app.get("/llm-status")
async def llm_status():
    return llm_gate.stats()


# ── 모델 전환 ──────────────────────────────────────

class ModelChange(BaseModel):
    name: str


@app.get("/model")
async def get_model(user=Depends(get_current_user)):
    """현재 활성 모델 + ollama 에 받아둔 사용 가능 모델 목록."""
    import requests
    import llm.ollama as ollama_mod
    try:
        r = requests.get(f"{ollama_mod.OLLAMA_HOST}/api/tags", timeout=5)
        installed = sorted([m["name"] for m in r.json().get("models", [])])
    except Exception:
        installed = [ollama_mod.OLLAMA_MODEL]
    return {"current": ollama_mod.OLLAMA_MODEL, "available": installed}


@app.post("/model")
async def set_model(req: ModelChange, user=Depends(get_current_user)):
    """런타임 모델 전환 — ollama 에 받아둔 모델만 허용."""
    import requests
    import llm.ollama as ollama_mod
    try:
        r = requests.get(f"{ollama_mod.OLLAMA_HOST}/api/tags", timeout=5)
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        installed = set()
    if req.name not in installed:
        raise HTTPException(
            status_code=400,
            detail=f"모델 '{req.name}' 가 ollama 에 없음. 먼저 'ollama pull {req.name}' 필요",
        )
    ollama_mod.OLLAMA_MODEL = req.name
    return {"current": req.name, "available": sorted(installed)}


# ── 인강 영상 (SSE 진행 + 별도 다운로드) ────────────

def _safe_name(name: str) -> str:
    """파일명에서 확장자 제거 + 경로/특수문자 sanitize."""
    base = (name or "untitled").rsplit(".", 1)[0]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    return (safe.strip("_.") or "untitled")[:80]


def _video_path(student_id, source_filename=None):
    """현재 강의자료에 매칭되는 영상 경로.
    source_filename 안 주면 RAG 메타에서 자동 조회. 메타 없으면 None (옛 영상 매칭 차단)."""
    if source_filename is None:
        meta = get_meta(student_id) or {}
        source_filename = meta.get("filename")
    if not source_filename:
        return None
    return os.path.join(BASE_DIR, f"lecture_{student_id}_{_safe_name(source_filename)}.mp4")


def _video_context(sid: str) -> str:
    """영상 스크립트용 컨텍스트 — 사전처리된 요약+개념 우선, 없으면 원본 발췌.
    요약+개념은 LLM이 정제한 결과라 스크립트 품질↑ + 토큰 소모↓."""
    pre = _pre_get(sid)
    summary  = pre.get("summary",  "").strip()
    concepts = pre.get("concepts", "").strip()
    if summary and concepts:
        return (
            f"[강의 요약]\n{summary}\n\n"
            f"[핵심 개념]\n{concepts}\n\n"
            f"[원본 발췌]\n{_full_context(sid, max_chars=2000)}"
        )
    if summary:
        return f"[강의 요약]\n{summary}\n\n[원본 발췌]\n{_full_context(sid, max_chars=3000)}"
    return _full_context(sid)


@app.get("/generate-video")
async def generate_video(user=Depends(get_user_from_query)):
    """SSE: 진행 상황 토큰 + 완료 시 download URL."""
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")

    async def progress_stream():
        import queue as q_mod
        event_queue = q_mod.Queue()

        def emit(stage, current, total, msg):
            event_queue.put({
                "stage":   stage,
                "current": current,
                "total":   total,
                "msg":     msg,
            })

        def produce():
            try:
                level_info = get_student_level(sid)
                context    = _video_context(sid)
                out_path   = _video_path(sid)
                if not out_path:
                    raise RuntimeError("강의자료 메타 없음")
                generate_lecture(context, level_info, out_path, on_progress=emit)
                bust = int(os.path.getmtime(out_path)) if os.path.exists(out_path) else 0
                event_queue.put({"done": True, "url": f"/video?t={bust}"})
            except Exception as e:
                event_queue.put({"error": str(e)})
            finally:
                event_queue.put(None)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(llm_executor, produce)

        yield f"data: {json.dumps({'stage': 'start', 'msg': '인강 생성 시작'}, ensure_ascii=False)}\n\n"
        # cloudflared / nginx 등 reverse proxy 의 buffering 방지용 패딩 (2KB)
        yield ":" + (" " * 2048) + "\n\n"
        while True:
            try:
                item = await asyncio.wait_for(
                    loop.run_in_executor(None, event_queue.get),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"  # SSE comment — 클라이언트 무시, proxy buffering 깨기
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/video")
async def get_video(user=Depends(get_user_from_query)):
    sid  = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=404, detail="문서 업로드 필요")
    path = _video_path(sid)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="인강 영상이 아직 생성되지 않음")
    meta = get_meta(sid) or {}
    src  = (meta.get("filename") or "강의자료").rsplit(".", 1)[0]
    download_name = f"요약본_{src}.mp4"
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
    return FileResponse(path, media_type="video/mp4", filename=download_name, headers=headers)


# ── 영상 생성 폴링 방식 (cloudflared SSE buffering 우회) ───

_video_jobs: dict[str, dict] = {}   # sid → {stage, current, total, msg, done, error, url}
_video_lock = threading.RLock()


@app.post("/video/generate")
async def start_video_generation(user=Depends(get_current_user)):
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")

    import time as _time
    with _video_lock:
        existing = _video_jobs.get(sid)
        if existing and not existing.get("done"):
            return {"status": "already_running"}
        _video_jobs[sid] = {
            "stage":   "start",
            "current": 0, "total": 1,
            "msg":     "준비 중",
            "done":    False,
            "error":   None,
            "url":     None,
            "started_at":   _time.time(),
            "completed_at": None,
            "elapsed":      0,
        }

    def emit(stage, cur, tot, msg):
        with _video_lock:
            j = _video_jobs.get(sid)
            if j:
                j["stage"]   = stage
                j["current"] = cur
                j["total"]   = tot
                j["msg"]     = msg

    def worker():
        try:
            level_info = get_student_level(sid)
            context    = _video_context(sid)
            out_path   = _video_path(sid)
            if not out_path:
                raise RuntimeError("강의자료 메타 없음")
            generate_lecture(context, level_info, out_path, on_progress=emit)
            bust = int(os.path.getmtime(out_path)) if os.path.exists(out_path) else 0
            with _video_lock:
                j = _video_jobs.get(sid, {})
                j["done"]         = True
                j["url"]          = f"/video?t={bust}"
                j["stage"]        = "done"
                j["completed_at"] = _time.time()
                if j.get("started_at"):
                    j["elapsed"] = round(j["completed_at"] - j["started_at"], 1)
        except Exception as e:
            import traceback; traceback.print_exc()
            with _video_lock:
                j = _video_jobs.get(sid, {})
                j["done"]         = True
                j["error"]        = str(e)
                j["completed_at"] = _time.time()
                if j.get("started_at"):
                    j["elapsed"] = round(j["completed_at"] - j["started_at"], 1)

    executor.submit(worker)
    return {"status": "started"}


@app.get("/video/status")
async def get_video_status(user=Depends(get_current_user)):
    import time as _time
    sid = user["student_id"]
    with _video_lock:
        job = _video_jobs.get(sid)
        if job:
            # 진행 중이면 elapsed 동적 계산
            if not job.get("done") and job.get("started_at"):
                job = {**job, "elapsed": round(_time.time() - job["started_at"], 1)}
    if job:
        return job
    # 메모리에 job 없는 경우 — 현재 강의자료에 매칭되는 영상이 있을 때만 ready
    if not has_state(sid):
        return {"stage": "idle", "current": 0, "total": 1, "msg": "",
                "done": False, "error": None, "url": None}
    path = _video_path(sid)
    if path and os.path.exists(path):
        # 캐시 우회용 timestamp — 같은 url 이라도 매번 새로 받음
        bust = int(os.path.getmtime(path))
        return {
            "stage": "ready", "current": 1, "total": 1, "msg": "",
            "done": True, "error": None, "url": f"/video?t={bust}",
        }
    return {
        "stage": "idle", "current": 0, "total": 1, "msg": "",
        "done": False, "error": None, "url": None,
    }


# ── QA 데이터셋 생성 ───────────────────────────────

@app.get("/generate-qa")
async def generate_qa(user=Depends(get_user_from_query)):
    sid = user["student_id"]
    if not has_state(sid):
        raise HTTPException(status_code=400, detail="문서 업로드 필요")

    async def sse_gen():
        import queue as q_mod
        progress_queue = q_mod.Queue()
        all_pairs      = []

        def produce():
            try:
                for current, total, new_pairs in generate_qa_pairs(get_parents(sid)):
                    all_pairs.extend(new_pairs)
                    progress_queue.put((current, total, False))
            finally:
                progress_queue.put((0, 0, True))

        loop = asyncio.get_running_loop()
        loop.run_in_executor(llm_executor, produce)

        while True:
            current, total, done = await loop.run_in_executor(None, progress_queue.get)
            if done:
                path = save_qa_pairs(all_pairs, output_path=f"qa_{sid}.jsonl")
                yield f"data: {json.dumps({'done': True, 'total': len(all_pairs), 'path': path}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'current': current, 'total': total}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 학생 정보 ──────────────────────────────────────

@app.get("/my-level")
async def my_level(user=Depends(get_current_user)):
    return get_student_level(user["student_id"])


@app.get("/export-student/{student_id}")
async def export_student(student_id: str, user=Depends(get_current_user)):
    # 본인 데이터만 export 가능
    if user["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="본인 데이터만 export 가능합니다.")
    path = export_student_data(student_id, f"qa_{student_id}.jsonl")
    info = get_student(student_id)
    return {
        "student_id":    student_id,
        "interactions":  info["interaction_count"],
        "model_trained": info["model_trained"],
        "exported_to":   path,
    }


# ── 노트 (저장/모아보기/삭제/카테고리 변경) ─────────

@app.post("/notes")
async def post_note(req: NoteRequest, user=Depends(get_current_user)):
    note_id = add_note(user["student_id"], req.content, req.category, req.source)
    return {"id": note_id}


@app.get("/notes")
async def get_notes(category: str | None = None, user=Depends(get_current_user)):
    return list_notes(user["student_id"], category)


@app.delete("/notes/{note_id}")
async def del_note(note_id: int, user=Depends(get_current_user)):
    if not delete_note(user["student_id"], note_id):
        raise HTTPException(status_code=404, detail="노트 없음")
    return {"ok": True}


@app.patch("/notes/{note_id}")
async def patch_note(note_id: int, req: NoteCategory, user=Depends(get_current_user)):
    if not update_note_category(user["student_id"], note_id, req.category):
        raise HTTPException(status_code=404, detail="노트 없음")
    return {"ok": True}


# ── 학습 일정 (CRUD) ──────────────────────────────

@app.post("/schedule")
async def post_schedule(req: ScheduleRequest, user=Depends(get_current_user)):
    sid = add_schedule(user["student_id"], req.date, req.title, req.note)
    return {"id": sid}


@app.get("/schedule")
async def get_schedule(user=Depends(get_current_user)):
    return list_schedules(user["student_id"])


@app.patch("/schedule/{sched_id}")
async def patch_schedule(sched_id: int, req: ScheduleUpdate, user=Depends(get_current_user)):
    if not update_schedule(user["student_id"], sched_id, **req.dict(exclude_none=True)):
        raise HTTPException(status_code=404, detail="일정 없음")
    return {"ok": True}


@app.delete("/schedule/{sched_id}")
async def del_schedule(sched_id: int, user=Depends(get_current_user)):
    if not delete_schedule(user["student_id"], sched_id):
        raise HTTPException(status_code=404, detail="일정 없음")
    return {"ok": True}


if __name__ == "__main__":
    # 학교 서버 운영용 ENV:
    #   APP_HOST=0.0.0.0  APP_PORT=7860
    #   APP_WORKERS=1     (Ollama 단일 GPU 라 워커 1 권장 - 모델 RAM 중복 회피)
    #   WORKER_POOL=8     (검색 / IO 스레드풀 크기)
    #   LLM_CONCURRENCY=1 (LLM 게이트 동시 처리)
    host    = os.getenv("APP_HOST", "0.0.0.0")
    port    = int(os.getenv("APP_PORT", "7860"))
    workers = int(os.getenv("APP_WORKERS", "1"))
    if workers > 1:
        # 멀티 워커 모드 — 모듈 경로 문자열로 넘겨야 함
        uvicorn.run("app:app", host=host, port=port, workers=workers,
                    timeout_keep_alive=30)
    else:
        uvicorn.run(app, host=host, port=port, timeout_keep_alive=30)
