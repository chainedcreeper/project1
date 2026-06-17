"""학생별 개인화 모델 (ollama 로 등록)."""
import json
import os
import subprocess

import requests

from .gateway import llm_gate

MODELS_DIR  = "personal_models"
OLLAMA_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def personal_model_name(student_id: str) -> str:
    return f"personal_{student_id}"


def personal_model_exists(student_id: str) -> bool:
    try:
        resp   = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return personal_model_name(student_id) in models
    except Exception:
        return False


def register_personal_model(student_id: str, gguf_path: str):
    model_dir = os.path.join(MODELS_DIR, student_id)
    os.makedirs(model_dir, exist_ok=True)

    modelfile_path = os.path.join(model_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(f"FROM {os.path.abspath(gguf_path)}\n")
        f.write('SYSTEM "너는 개인화된 대학 강의 AI Tutor다. 이 학생의 수준과 패턴에 맞게 답변하라."\n')

    subprocess.run(
        ["ollama", "create", personal_model_name(student_id), "-f", modelfile_path],
        check=True,
    )


def _build_prompt(context: str, question: str) -> str:
    return f"문서:\n{context}\n\n질문: {question}"


def ask_personal(student_id: str, context: str, question: str) -> str:
    with llm_gate.acquire():
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   personal_model_name(student_id),
                "prompt":  _build_prompt(context, question),
                "stream":  False,
                "options": {"num_predict": 512, "num_ctx": 2048},
            },
        )
    return resp.json()["response"]


def ask_personal_stream(student_id: str, context: str, question: str):
    with llm_gate.acquire():
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   personal_model_name(student_id),
                "prompt":  _build_prompt(context, question),
                "stream":  True,
                "options": {"num_predict": 512, "num_ctx": 2048},
            },
            stream=True,
        )
        for line in resp.iter_lines():
            if line:
                token = json.loads(line).get("response", "")
                if token:
                    yield token
