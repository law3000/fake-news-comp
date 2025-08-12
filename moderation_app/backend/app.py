from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Literal

app = FastAPI()

# CORS: allow your frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class Citation(BaseModel):
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None

class Explanation(BaseModel):
    summary: str
    reasoning: str
    limitations: Optional[str] = None
    citations: List[Citation]

class AnalyzeResponse(BaseModel):
    label: Literal["fake", "real", "abstain"]
    confidence: float
    explanation: Explanation
    snippets: List[Citation]
    report_text: str  # ready-to-copy moderation note

class AnalyzeRequest(BaseModel):
    url: Optional[HttpUrl] = None
    text: Optional[str] = None
    return_debug: bool = False

# ---------- Health & root ----------
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------- Pipeline imports (relative; app.py is inside backend/) ----------
from .models.extract import extract_text
from .models.classifier import classify_article
from .models.rag import retrieve_support
from .models.llm import generate_explanation
from .models.report import format_report

# ---------- Endpoint ----------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # Guard: require either url or text
    if not (req.text or req.url):
        raise HTTPException(status_code=400, detail="Provide either url or text.")

    # Extract content (prefer raw text if provided)
    if req.text and req.text.strip():
        content = req.text
    else:
        content = extract_text(str(req.url))  # req.url is not None here

    if not content or not content.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text content.")

    # Pipeline
    clf = classify_article(content)            # {label, confidence, ...}
    support = retrieve_support(content)        # list[ Citation-like dicts ]
    expl = generate_explanation(clf, support)  # Explanation dict/model

    # Build ready-to-copy moderation note
    report_text = format_report(
        clf.get("label", "abstain"),
        float(clf.get("confidence", 0.0)),
        expl
    )

    return {
        "label": clf.get("label", "abstain"),
        "confidence": float(clf.get("confidence", 0.0)),
        "explanation": expl,
        "snippets": support,
        "report_text": report_text,
    }
