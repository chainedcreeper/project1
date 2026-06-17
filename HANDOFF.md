# 학교 작업 인수 문서

이 문서는 **집 PC 에서 박은 거 + 학교에서 이어서 박을 거** 를 정리한 핸드오버임.
다음 세션 (학교) 의 Claude 가 이걸 읽고 바로 이어서 작업하면 됨.

---

## 지금까지 (2026-06-18 집 세션) 박힌 거

### Commit history (project1/main)

| Commit | 내용 |
|---|---|
| `96ca733` | 멀티유저 대응 인프라 + 유사 검색 폴백 (게이트 / `/find` / 임베딩 폴백 / 모드 토글) |
| `99d87e2` | 사전처리 체인 + 채팅 자동 라우팅 + 영상 동기화 fix |
| `51ee40d` | docs: PPT v3 + 발표대본 보완 |
| `d0490ed` | 페이지 이미지 + 미니맵 + sanity check + 게이트 분리 |

### 핵심 동작 흐름 (현재)

```
업로드
  ↓
원본 파일 → uploaded_docs/{sid}/source.{ext} 영구 저장
  ↓
자동 사전처리 체인:
  요약 (~3분) → 영상 백그라운드 (요약 기반 스크립트) → 핵심개념 → 예상문제
  ↓
학생 학습:
  - 채팅 입력 → 자동 라우팅
    - 짧은 키워드 → /find (sub-second)
    - 의문문 → /chat (LLM)
    - "/find 키워드" 명시 명령
  - find 결과:
    - 페이지 이미지 inline (PDF 즉시 / PPTX 는 soffice 있어야)
    - 미니맵 — 등장 페이지 칩 클릭으로 점프
    - 본문 스니펫
    - exact (노랑) / fuzzy (보라) 뱃지
```

### 환경변수 (현재 지원)

| 변수 | 기본 | 설명 |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | 바인드 |
| `APP_PORT` | `7860` | 포트 |
| `APP_WORKERS` | `1` | uvicorn 워커 (모델 RAM 중복 회피용 1 권장) |
| `WORKER_POOL` | `8` | 검색·IO 스레드풀 |
| `LLM_CONCURRENCY` | `1` | 8B 게이트 동시 처리 (L40S 면 2~3 권장) |
| `LLM_CONCURRENCY_LARGE` | `1` | 32B 게이트 동시 처리 |
| `LLM_QUEUE_TIMEOUT` | `180` | 큐 대기 한도(초) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 엔드포인트 |
| `OLLAMA_MODEL` | `qwen3:8b` | 기본 모델 |
| `LIGHT_MODEL` | `qwen3:8b` | 입문/중급 자동 라우팅 모델 |
| `AUTO_ROUTE_BY_LEVEL` | `1` | 학생 레벨별 모델 자동 선택 |
| `REMOTE_LLM_URL` | (학교 IP) | 32B 폴백 서버 |
| `REMOTE_LLM_MODEL` | `exaone3.5:32b` | 32B 모델명 |
| `FAST_CHAT` | `1` | 채팅 reranker 건너뜀 + num_predict 2048 캡 |
| `RAG_INDEX_DIR` | `rag_indexes` | 인덱스 저장 경로 |

---

## 학교 첫 부팅 — 가동 절차

### 1. 코드 받기
```bash
cd ~
git clone https://github.com/chainedcreeper/project1.git pj_home
cd pj_home
git checkout main
```

### 2. Python 의존성
```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn sentence-transformers faiss-cpu requests
pip install python-jose[cryptography] "passlib[bcrypt]==1.7.4" "bcrypt<4.1"  # ← 호환성 고정 필수
pip install python-multipart edge-tts imageio-ffmpeg PyMuPDF pillow matplotlib
pip install python-pptx python-docx
pip install json_repair
```

### 3. Ollama
```bash
ollama pull qwen3:8b
ollama pull qwen3:32b      # 옵션 (32B 라우팅 박으면 필요)

# Ollama 동시 요청 — L40S 면 2 가능
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=24h ollama serve
```

### 4. 임베딩/리랭커 모델 사전 다운로드 (오프라인 환경 대비)
```bash
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
  SentenceTransformer('BAAI/bge-m3'); \
  CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

### 5. PPTX 페이지 이미지용 LibreOffice (PPT 업로드 시 페이지 이미지 보고 싶으면)
```bash
# Debian/Ubuntu
sudo apt install libreoffice --no-install-recommends

# 또는 학교 컨테이너 sudo 안 되면 conda
conda install -c conda-forge libreoffice
```

### 6. 가동
```bash
export LLM_CONCURRENCY=2          # L40S 면 8B 동시 2 가능
export LLM_CONCURRENCY_LARGE=1    # 32B 는 직렬
export FAST_CHAT=1                # 채팅 빠르게
export OLLAMA_MODEL=qwen3:8b
python app.py
```

### 7. 동작 확인 (체크리스트)

- [ ] 회원가입 → 로그인 (500 에러 없어야)
- [ ] PDF 업로드 → 자동 사전처리 체인 (~3~5분 학교 기준)
- [ ] 영상 슬라이드↔음성 동기화 정상 (11초로 안 끝나야)
- [ ] 채팅창: "데이터베이스" → find 자동, "왜 ACID 가 어려워?" → chat 자동
- [ ] find 결과 카드에 페이지 이미지 보임
- [ ] `/llm-status` 큐 카운터 노출

---

## 학교에서 다음 박을 거 — 우선순위 순

### [1] 32B 라우팅 + 인터넷 검색 (예상 1.5시간)

**목적**: 자료 안에 없는 추가 개념·실제 사례 질문 시 32B + 웹 검색.

**구현 위치**:
- `llm/remote.py` — 이미 32B endpoint 스텁 있음
- `app.py /chat` 에서 라우팅 추가

**라우팅 로직**:
```python
# rag.core 에서 _get_context 의 reranker top score 노출
def chat_route(question, sid, level_info):
    ctx, score = _get_context_with_score(question, sid)
    if score > 0.7:           # 자료에 직답 있음
        return 'local_8b', ctx
    if score < 0.4:           # 자료 못 찾음
        return 'remote_32b_with_web', None
    return 'local_8b', ctx    # 중간 — 일단 8B 시도
```

**웹 검색**:
```bash
pip install duckduckgo-search   # 무료, API 키 X
```
```python
from duckduckgo_search import DDGS
results = DDGS().text(query, max_results=5)
# 32B 에 검색 결과 + 질문 같이 넣음
```

### [2] PDF/PPTX 페이지 이미지 PPT 지원 확인 (예상 30분)

**현재 상태**: `_render_page_png` 에 PPTX → soffice 변환 코드 있음. 학교 LibreOffice 설치 확인만.

```bash
which soffice || which libreoffice
# 없으면 위 5번 설치
```

PPTX 업로드 → find 검색 → 카드에 페이지 이미지 보이면 OK.

### [3] 전체 모듈화 (예상 2시간)

**현재 `app.py` 700+ 줄**. router 별 분리.

**제안 구조**:
```
app/
  __init__.py     # FastAPI app 인스턴스 + 미들웨어
  auth.py         # /register /login /model
  upload.py       # /upload /document-status
  analyze.py      # /analyze /summary /concepts /exam (PROMPTS 포함)
  video.py        # /generate-video /video /video/generate /video/status
  chat.py         # /chat
  find.py         # /find /page-image
  notes.py        # 기존 student/notes 그대로
  schedule.py     # 기존 그대로
  status.py       # /llm-status
main.py           # uvicorn entry, ENV 처리만
```

회귀 위험 — router decorator 옮기는 거라 한 번에 박지 말고 한 모듈씩 검증.

### [4] Redis 응답 캐시 (예상 1.5~2시간)

**목적**: 동접 500 시나리오 — 비슷한 질문 반복되면 캐시 hit.

**키 설계**:
```
chat:{sid}:{question_hash}     # 학생별 질문 캐시
analyze:{sid}                  # 분석 결과 캐시 (TTL 24h)
summary:{sid}                  # 요약 캐시
```

**구현**:
```bash
pip install redis hiredis
# Docker: docker run -d -p 6379:6379 redis:7-alpine
```
```python
import redis
r = redis.Redis(decode_responses=True)
def cached_ask(sid, q):
    key = f"chat:{sid}:{hash(q)}"
    if cached := r.get(key):
        return cached
    answer = ask_stream(...)
    r.setex(key, 3600, answer)
    return answer
```

### [5] 학생 학습 데이터 수집 (예상 2~3시간)

**목적**: 학생 행동 로그 → level_assessor 정확도 ↑ + 시험 출제 약점 위주.

**스키마**:
```sql
CREATE TABLE student_action (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sid TEXT NOT NULL,
  action TEXT NOT NULL,   -- 'chat', 'find', 'note_save', 'exam_correct', 'exam_wrong'
  query TEXT,
  metadata JSON,
  created_at REAL DEFAULT (strftime('%s','now'))
);
```

**활용**:
- `/chat`, `/find` 호출마다 INSERT
- 시험 채점 시 정답/오답 INSERT
- 일정 주기로 약점 키워드 추출 → 다음 시험에 가중치

### [6] 영상 sanity check 강화 (예상 30분)

**현재**: script_gen 결과 슬라이드 수·narration 길이 1회 retry.
**추가**: 영상 compose 후에도 길이 검증 (mp4 duration vs 예상 length).

```python
def _validate_video(path: str, expected_min_sec: float) -> bool:
    """fitz/ffmpeg probe 로 영상 길이 측정. 너무 짧으면 재생성 권장."""
    import subprocess, re
    from imageio_ffmpeg import get_ffmpeg_exe
    r = subprocess.run([get_ffmpeg_exe(), '-i', path, '-f', 'null', '-'],
                       capture_output=True)
    m = re.search(rb'time=(\d+):(\d+):(\d+\.\d+)', r.stderr)
    if not m: return False
    h, mn, s = m.groups()
    duration = int(h)*3600 + int(mn)*60 + float(s)
    return duration >= expected_min_sec
```

---

## 알려진 한계 / 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 로그인 500 | bcrypt 5.x + passlib 비호환 | `pip install "bcrypt<4.1"` |
| 영상 11초 끝남 | Edge-TTS 한국어 WordBoundary 없음 | 이미 fix (SentenceBoundary 수집 + ffmpeg probe) |
| 페이지 이미지 404 | 원본 파일 없음 / 페이지 범위 초과 / PPTX + soffice 없음 | uploaded_docs/{sid}/ 확인, soffice 설치 |
| 채팅 5분+ | 정상 (집 PC). 학교 L40S 면 30초~1분 | 시간 측정만 |
| `OSError: bge-reranker not found` | 모델 미캐시 | 위 4번 사전 다운로드 |
| 분석/영상 동시 클릭 시 굶음 | 게이트 capacity=1 | `LLM_CONCURRENCY=2` (L40S) |

---

## 발표 자료

- `docs/presentation/AI_Tutor_프로젝트_소개_v3.pptx` (16장 — v2 11장 + 추가 5장)
- `docs/presentation/AI_Tutor_발표대본.md` (시행착오 + 발전 가능 섹션 포함)
- `docs/presentation/AI_Tutor_예상질문집.md` (기존 그대로)

PPT 추가된 5장:
1. 자동 사전처리 체인
2. 시행착오 — 영상 11초 사건
3. 채팅 자동 라우팅 + /find
4. 발전 가능 — 학교 서버에서 다음 박을 것
5. 학교 시연 체크리스트

---

## 학교 Claude 에게 한 줄

> 위 [1]~[6] 우선순위대로 박으면 됨. 코드 위치·예상 시간·구현 힌트 다 적혀있음.
> 의문 생기면 commit history (`git log --oneline`) 보면 흐름 따라갈 수 있음.
> 마요 말투는 집 PC (100.83.173.111) 에서만 ON. 학교 PC 면 일반 말투.
