# 학교 서버 실사용 배포 가이드

## 핵심 설계
- **FastAPI 단일 프로세스 + 스레드풀** — 임베딩 모델 / Reranker 가 GPU·RAM 공유라 워커를 여러 개 띄우면 모델이 N배 복제됨. 1 worker 권장.
- **LLM 게이트 (`llm_gate`)** — Ollama 단일 GPU 직렬화. 동시 요청은 Semaphore 로 큐잉, 대기자에게는 `/llm-status` 로 순번 노출.
- **검색·IO 병렬** — `WORKER_POOL` 스레드로 빠른 검색 (`/find`), DB, 파일 IO 만 병렬.
- **빠른 검색 모드** — 채팅창 우측 [🔍 근거 찾기] 토글. LLM 안 거치고 parents 직검색 → sub-second.

## 실행

```bash
cd ~/pj_home
source ~/venv/bin/activate          # jose, faiss 등은 venv 안에만 있음

# 기본 (단일 워커)
APP_HOST=0.0.0.0 APP_PORT=7860 \
WORKER_POOL=8 LLM_CONCURRENCY=1 \
python app.py
```

## 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `APP_HOST` | `0.0.0.0` | 바인드 주소 |
| `APP_PORT` | `7860` | 포트 |
| `APP_WORKERS` | `1` | uvicorn 워커 (1 권장 - 모델 RAM 중복 회피) |
| `WORKER_POOL` | `8` | 검색·IO 스레드풀 |
| `LLM_CONCURRENCY` | `1` | Ollama 동시 요청 한도 (GPU 1장 = 1) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 엔드포인트 |
| `OLLAMA_MODEL` | `qwen3:8b` | 기본 모델 |
| `RAG_INDEX_DIR` | `rag_indexes` | 사용자별 인덱스 저장 경로 |

## Ollama 측 설정 (서버에서)

```bash
# 동시 요청 처리 한도 (GPU 1장이면 1)
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_KEEP_ALIVE=24h \
ollama serve
```

## Tailscale 노출 (학교 PC 외부에서 접근)

학교 GPU 서버는 Kubeflow 내부망. 학교 PC 가 Tailscale 게이트.
1. 학교 PC 에 `tailscale up` (이미 설정됨)
2. 학교 PC 에서 GPU 서버로 SSH 포워딩:
   ```bash
   ssh -L 7860:GPU_SERVER_IP:7860 jovyan@GPU_SERVER_IP -N
   ```
3. 집 PC 에서: `http://100.118.126.123:7860`

## 동시 접속 테스트 (간이)

```bash
# 5명 동시 채팅 모사
for i in $(seq 1 5); do
  curl -s -X POST http://localhost:7860/chat \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"message":"이 자료의 핵심 개념 요약"}' &
done
wait

# 큐 상태 실시간 확인
watch -n 1 'curl -s http://localhost:7860/llm-status'
```

기대 동작:
- 1명은 즉시 처리 (`active=1, waiting=4`)
- 4명은 대기 — 첫 번째 끝나면 순차 진행
- 클라이언트 UI 상단에 "다른 학생이 LLM 사용 중" 배너 자동 표시

## "30초 답변" 문제 — 해결 흐름

기존: 우측 채팅 → `/chat` → RAG (BGE rerank) → Ollama LLM → 30초+
신규: 우측 채팅에 [🔍 근거 찾기] 토글 → `/find` → parents 직접 매칭 → 100ms 안짝.

학생 사용 패턴:
- "이게 어디 나옴?" / "○○ 정의는?" → 🔍 근거 찾기 (페이지 + 스니펫)
- "왜 그런지 설명해줘" / "정리해줘" → 💬 답변 받기 (기존 LLM 흐름)

## 학교 서버 첫 배포 체크리스트

- [ ] `~/pj_home` 동기화 (`scp` 또는 `git pull`)
- [ ] `python -c "import faiss, jose, fastapi"` 통과
- [ ] Ollama 서비스 떠 있음 (`curl http://localhost:11434/api/tags`)
- [ ] 기본 모델 pull (`ollama pull qwen3:8b`)
- [ ] `rag_indexes/` 디렉터리 쓰기 권한
- [ ] 방화벽 7860 포트 (내부망만)
- [ ] `python app.py` → 부팅 로그에 임베딩 모델 로드 성공 확인
- [ ] 브라우저에서 회원가입 → 업로드 → 분석 → 채팅 / 검색 양 모드 시연
