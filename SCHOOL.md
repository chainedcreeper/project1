# 학교 서버 배포 — 작업 인수 문서

이 문서는 **집 PC (RTX 4060 8GB) 에서 검증 끝낸 코드를 학교 GPU 서버 (L40S 48GB) 에 올려 가동** 하기 위한 인수 문서.

---

## 환경 차이 요약

| | 집 PC | 학교 서버 |
|---|---|---|
| GPU | RTX 4060 8GB | L40S 48GB |
| qwen3:8b | VRAM | VRAM |
| bge-m3 (임베딩) | CPU swap (느림) | VRAM |
| bge-reranker-v2-m3 | CPU swap (느림) | VRAM |
| qwen3:32b (고급) | 미설치 | **학교에서 설치** |

→ 학교 GPU 는 모든 모델 VRAM 상주 → 추론 3~5배 빠름

---

## 학교에서 추가로 해야 할 것

### 1. 32B 모델 설치
```bash
ollama pull qwen3:32b      # 또는 exaone3.5:32b
```

### 2. 환경변수 (학교 서버 시작 전 export)
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3:8b               # 기본 = 8B
export REMOTE_LLM_URL=http://localhost:11434  # 32B 도 같은 ollama
export REMOTE_LLM_MODEL=qwen3:32b
export FAST_CHAT=1                         # 채팅 빠르게 (rerank 건너뜀)
export LLM_CONCURRENCY=2                   # L40S 면 동시 2 가능 (batching)
export APP_HOST=0.0.0.0
export APP_PORT=7860
```

### 3. Ollama 동시 요청 설정
```bash
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=24h ollama serve
```

### 4. Python 의존성
```bash
pip install -r requirements.txt  # FastAPI, sentence-transformers, faiss-cpu, edge_tts, imageio_ffmpeg, ...
pip install python-jose[cryptography] "passlib[bcrypt]==1.7.4" "bcrypt<4.1"  # ⚠ bcrypt 5.x 와 passlib 호환 문제 — 4.0.x 고정
pip install json_repair
```

### 5. 모델 사전 다운로드 (오프라인 환경 대비)
```bash
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
  SentenceTransformer('BAAI/bge-m3'); \
  CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

### 6. 서버 가동
```bash
cd ~/pj_home
source ~/venv/bin/activate
python app.py
```

---

## 집에서 검증 끝낸 것 (학교에서 재현 확인만 하면 됨)

- ✅ 자동 사전처리 체인: 업로드 → 요약 → 영상 (요약 기반) → 개념 + 시험
- ✅ `/find` 빠른 키워드 검색 (LLM 안 거침, sub-second)
- ✅ `/find` 임베딩 폴백 (오타 / 동의어 / 띄어쓰기 변형, bge-m3)
- ✅ 채팅창 자동 라우팅 — 짧은 키워드 → find, 의문문 → chat
- ✅ `/find <키워드>` 명시 명령어
- ✅ `/summary`, `/concepts`, `/exam` 단독 endpoint (사전처리 체인 + 수동 재실행 가능)
- ✅ 영상 슬라이드↔음성 동기화 버그 fix (Edge-TTS 한국어 WordBoundary 미지원 → SentenceBoundary + 번들 ffmpeg probe)
- ✅ LLM 게이트 (Semaphore) + 큐 상태 노출 (`/llm-status`)
- ✅ 멀티유저 캐시 무효화 (rag.state mtime)
- ✅ 사전처리 결과를 영상 스크립트 컨텍스트로 재활용 (`_video_context`)
- ✅ 채팅 빠르게 — `FAST_CHAT=1` 일 때 reranker 건너뜀 + `num_predict=2048`

---

## 학교에서 박을 것 — 다음 Claude 에게 (시간순 우선)

### 1. PPT/PDF 페이지를 이미지로 직접 표시 (직관성 ↑) — 최우선

학생 의도: find 결과 카드에 **본문 스니펫만 보여주지 말고 원본 페이지 이미지 그대로 보여주기**. PPT 한 장 / PDF 한 페이지 캡처처럼.

구현:
- 새 endpoint `/page-image?page=N` — fitz (PyMuPDF) 로 페이지 PNG 렌더링
- `_safe_name(filename).png` 식으로 캐시
- find 결과 카드에 `<img src="/page-image?page=N">` 추가
- 디바이스별 lazy loading

```python
import fitz  # 이미 깔려있음
def render_page_png(pdf_path: str, page_num: int, dpi: int = 110) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    return page.get_pixmap(matrix=mat).tobytes("png")
```

PPTX 는 LibreOffice 헤드리스 변환 (`soffice --headless --convert-to pdf`) 으로 PDF 변환 후 동일 처리.

### 2. 32B 라우팅 + 인터넷 검색 통합 — 중요

학생 의도:
- 단순 개념 / PPT 내용 설명 → 8B (PPT 에서 찾아서 설명)
- 추가 개념 / 자료 밖 정보 필요 → 32B + 인터넷 검색

구현:
- `/chat` 에서 답변 신뢰도 측정 (RAG 매칭 점수)
- 신뢰도 낮음 → 32B + 웹 검색 (e.g. SerpAPI, DuckDuckGo)
- 32B 가 검색 결과 종합

이미 `llm/remote.py` 에 32B 엔드포인트 스텁 있음. 인터넷 검색 통합만 추가.

### 3. 전체 모듈화 — 시간 있으면

현재 `app.py` 가 너무 큼 (800+ 줄). 분리:
- `app/auth.py` — 인증 endpoint
- `app/upload.py` — 업로드 + 메타
- `app/analyze.py` — 요약 / 개념 / 시험
- `app/video.py` — 영상 생성
- `app/chat.py` — 채팅
- `app/find.py` — find + page-image

`app.py` 는 라우터 등록만.

### 4. 32B 동시 실행 시 게이트 capacity 분리

8B 게이트와 32B 게이트 분리. L40S 48GB 면 둘 다 동시 가능.

```python
llm_gate_8b  = LLMGate(capacity=2)
llm_gate_32b = LLMGate(capacity=1)
```

---

## 알려진 한계

- 집 PC 검증 시 모든 LLM 작업은 학교 대비 **3~5배 느림** (정상)
- bge-m3 + bge-reranker 가 집 4060 VRAM 못 들어가서 CPU swap → 채팅 + 분석 추가로 느림
- 학교 GPU 검증 한 번이면 실제 시연 속도 확보됨

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 로그인 500 에러 | bcrypt 5.x + passlib 호환 | `pip install "bcrypt<4.1"` |
| 영상 11초로 끝남 | Edge-TTS 한국어 WordBoundary 없음 | 이 commit 에서 fix 됨 (SentenceBoundary + ffmpeg probe) |
| `ImportError: jose` | `pip install python-jose[cryptography]` |
| `OSError: bge-reranker not found` | 모델 다운로드 안 됨 | 위 "5. 모델 사전 다운로드" 실행 |
| 채팅 느림 | `FAST_CHAT=0` 또는 reranker CPU | `export FAST_CHAT=1` |
| 분석 5분+ | 정상 (학교에선 1~3분) | 시간 측정만 |

---

## Tailscale 접속 (집 PC ↔ 학교 서버)

```bash
# 집 PC 에서
ssh 107@100.118.126.123             # 학교 PC 들어가기
ssh -L 7860:GPU_SERVER:7860 ...     # GPU 서버 포워딩
# 집 브라우저: http://localhost:7860
```

---

## commit 정책

- master 가 아니라 main 브랜치 사용 (`chainedcreeper/project1`)
- 새 자료 업로드 → 캐시 (`_pre_cache`, `_video_jobs`) 자동 폐기
- 메모리 상태는 워커 재시작 시 사라짐 → 디스크 캐시 의존
