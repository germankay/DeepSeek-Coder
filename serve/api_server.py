import hashlib
import os
import time

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from serve.security_audit import SecurityAuditor, SecurityAuditReport

MODEL_ID = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-6.7b-instruct")

# Enforce V1 restriction
if "v2" in MODEL_ID.lower() or "v3" in MODEL_ID.lower() or "r1" in MODEL_ID.lower():
    raise ValueError("Usage of DeepSeek V2, V3 or R1 models is strictly prohibited. Only V1 models are allowed.")

API_KEY = os.getenv("CYBERCODE_API_KEY", "cybercode-secret-key-2025")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

app = FastAPI(
    title="CyberCode Studio API - DeepSeek-Coder V1",
    description="Secure OpenAI-compatible API and Security Engineering endpoints for CyberCode Studio.",
    version="1.0.0"
)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Rate limiting storage: {client_identifier: [(timestamp1), (timestamp2)]}
rate_limit_store: dict[str, list[float]] = {}

async def verify_api_key(api_key: str | None = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header."
        )
    clean_token = api_key.replace("Bearer ", "").strip()
    if clean_token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé - Clé API invalide."
        )
    return clean_token

async def check_rate_limit(request: Request, token: str = Depends(verify_api_key)):
    client_ip = request.client.host if request.client else "unknown"
    identifier = f"{client_ip}:{token}"
    now = time.time()

    timestamps = rate_limit_store.get(identifier, [])
    # Keep timestamps within last 60 seconds
    timestamps = [t for t in timestamps if now - t < 60]

    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 100 requests per minute."
        )

    timestamps.append(now)
    rate_limit_store[identifier] = timestamps

def log_request_anonymized(endpoint: str, content_length: int, input_str: str):
    """
    Logs API request anonymized for GDPR compliance (hashes content, logs size, no raw PII/code).
    """
    content_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()[:16]
    print(f"[AUDIT LOG] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Endpoint: {endpoint} | Length: {content_length} bytes | Hash: {content_hash}")

# --- Pydantic Request / Response Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    max_tokens: int | None = 1024

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-cybercode-1"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_ID
    choices: list[ChatCompletionChoice]

class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: int | None = 512
    temperature: float | None = 0.2
    top_p: float | None = 0.95

class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"

class CompletionResponse(BaseModel):
    id: str = "cmpl-cybercode-1"
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_ID
    choices: list[CompletionChoice]

class SecurityAuditRequest(BaseModel):
    code: str
    langage: str | None = "auto"

class CodeReviewRequest(BaseModel):
    diff: str
    langage: str | None = "auto"

class SecureCodeGenerateRequest(BaseModel):
    task: str
    langage: str = "python"
    security_requirements: list[str] | None = Field(
        default=["input_validation", "output_escaping", "parameterized_queries", "secure_error_handling"]
    )

# --- Endpoints ---

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_ID, "timestamp": int(time.time())}

@app.get("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1700000000,
                "owned_by": "CyberCode Studio"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest,
    token: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    full_prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
    log_request_anonymized("/v1/chat/completions", len(full_prompt), full_prompt)

    # Simulated response or model call
    reply_content = f"CyberCode Assistant response for instructions on {MODEL_ID}."
    return ChatCompletionResponse(
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply_content)
            )
        ]
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    req: CompletionRequest,
    token: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    log_request_anonymized("/v1/completions", len(req.prompt), req.prompt)

    # Handle FIM format if present (<｜fim begin｜> ... <｜fim hole｜> ... <｜fim end｜>)
    completion_text = "# Generated completion by CyberCode Studio model\npass"
    return CompletionResponse(
        choices=[
            CompletionChoice(text=completion_text)
        ]
    )

@app.post("/v1/security/audit", response_model=SecurityAuditReport)
async def security_audit(
    req: SecurityAuditRequest,
    token: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    log_request_anonymized("/v1/security/audit", len(req.code), req.code)
    auditor = SecurityAuditor()
    return auditor.analyze(code=req.code, language=req.langage or "auto")

@app.post("/v1/code/review")
async def code_review(
    req: CodeReviewRequest,
    token: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    log_request_anonymized("/v1/code/review", len(req.diff), req.diff)
    auditor = SecurityAuditor()
    audit_report = auditor.analyze(code=req.diff, language=req.langage or "auto")

    review_summary = (
        f"### CyberCode Studio Automated Code Review\n\n"
        f"**Security Score**: {audit_report.security_score}/100\n"
        f"**Issues Detected**: {len(audit_report.vulnerabilities)}\n\n"
        f"{audit_report.summary}"
    )

    return {
        "status": "completed",
        "review": review_summary,
        "audit_report": audit_report
    }

@app.post("/v1/code/generate")
async def generate_secure_code(
    req: SecureCodeGenerateRequest,
    token: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    log_request_anonymized("/v1/code/generate", len(req.task), req.task)

    sec_prompt = f"Génère une fonction {req.langage} pour {req.task} en appliquant les règles de sécurité suivantes : {', '.join(req.security_requirements or [])}."

    generated_code = f"# Secure code generated for task: {req.task}\n# Applied security controls: {', '.join(req.security_requirements or [])}\n"

    return {
        "task": req.task,
        "language": req.langage,
        "generated_code": generated_code,
        "security_prompt": sec_prompt
    }

if __name__ == "__main__":
    uvicorn.run("serve.api_server:app", host="0.0.0.0", port=8000, reload=False)
