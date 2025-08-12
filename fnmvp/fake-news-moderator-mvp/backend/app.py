import os, glob, re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import joblib
from pathlib import Path
import numpy as _np

load_dotenv()
app = FastAPI(title="Fake News Moderator MVP (Lite)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Optional trained model artifacts (set if available)
MODELS_DIR = Path(__file__).parent / 'models'
CONTENT_DIR = MODELS_DIR / 'content'
ART_CONTENT = None
ART_TOKENIZER = None
ART_CONTEXT_MODEL = None
ART_CONTEXT_SCALER = None
ART_META = None
ART_CFG = None

try:
    if MODELS_DIR.exists():
        # Load meta-learner
        meta_p = MODELS_DIR / 'meta_learner.joblib'
        if meta_p.exists():
            ART_META = joblib.load(meta_p)
        # Load context artifacts
        ctx_m = MODELS_DIR / 'context_model.joblib'
        if ctx_m.exists():
            ART_CONTEXT_MODEL = joblib.load(ctx_m)
        ctx_s = MODELS_DIR / 'context_scaler.joblib'
        if ctx_s.exists():
            ART_CONTEXT_SCALER = joblib.load(ctx_s)
        # Load content model/tokenizer if transformers available
        if CONTENT_DIR.exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                ART_TOKENIZER = AutoTokenizer.from_pretrained(str(CONTENT_DIR))
                ART_CONTENT = AutoModelForSequenceClassification.from_pretrained(str(CONTENT_DIR))
                ART_CONTENT.eval()
            except Exception:
                ART_TOKENIZER = None
                ART_CONTENT = None
        # Load config
        cfg_p = MODELS_DIR / 'config.json'
        if cfg_p.exists():
            import json as _json
            ART_CFG = _json.loads(cfg_p.read_text())
except Exception:
    # Ignore load errors; we'll fall back to heuristic
    pass

FAKE_MARKERS = [
    "breaking!!!","shocking","what they dont want you to know","cure in 24 hours",
    "miracle remedy","share before deleted","wake up","do your own research",
    "100% proven","secret plan","msm won't tell you","you won't believe"
]
STOP = set("a an the and or but if then so to of in on for at by from with as is are was were be have has had this that these those it its their our your you we they them not no yes about into over under between after before during without within across more most many few some any much".split())

class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    rationale: Optional[str] = None

def extract_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article")
        if article:
            return article.get_text(" ", strip=True)
        for sel in ["main", "#content", ".post-content", ".article-body"]:
            node = soup.select_one(sel)
            if node:
                return node.get_text(" ", strip=True)
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def _ctx_features_from_json(s: str):
    try:
        import json
        d = json.loads(s) if s and s.strip() else {}
    except Exception:
        d = {}
    return [
        d.get("share_count", 0),
        d.get("unique_users", 0),
        d.get("avg_followers", 0),
        d.get("burstiness", 0.0),
        d.get("sentiment_score", 0.0),
        d.get("engagement_rate", 0.0),
    ]


def heuristic_classify(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    score = 0.0
    for m in FAKE_MARKERS:
        if m in t:
            score += 0.15
    if "http://" in t or "https://" in t:
        score += 0.05
    score = min(score, 0.95)
    label = "Fake" if score >= 0.5 else "Real"
    return {"label": label, "confidence": float(score), "rationale": "Heuristic-only (no trained model yet)."}

def _tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP]

def _chunk_words(words: List[str], size=80, overlap=20) -> List[str]:
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+size]
        if not chunk: break
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return chunks

def load_verified_chunks() -> List[Dict[str, Any]]:
    out = []
    base = os.path.join(os.path.dirname(__file__), "..", "verified")
    for path in glob.glob(os.path.join(base, "*.txt")) + glob.glob(os.path.join(base, "*.md")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            words = _tokenize(text)
            for j, ch in enumerate(_chunk_words(words)):
                out.append({"source": os.path.basename(path), "text": ch})
        except Exception:
            pass
    # Optional: include a few web results via Serper (if SERPER_API_KEY provided)
    try:
        key = os.getenv("SERPER_API_KEY")
        if key:
            import requests as _rq
            q = os.getenv("SERPER_BOOTSTRAP_QUERY", "site:reuters.com OR site:apnews.com fact check")
            resp = _rq.post("https://google.serper.dev/search", json={"q": q, "num": 5}, headers={"X-API-KEY": key, "Content-Type": "application/json"}, timeout=10)
            if resp.ok:
                data = resp.json()
                for it in (data.get("organic") or [])[:5]:
                    title = it.get("title", "")
                    snippet = it.get("snippet", "")
                    if snippet:
                        for ch in split_into_chunks(snippet, size=120):
                            out.append({"source": title or it.get("link","web"), "text": ch})
    except Exception:
        pass
    return out

VERIFIED_CHUNKS = load_verified_chunks()

# Optional embedding model for better retrieval (RAG)
EMB_MODEL = None
VERIFIED_EMB = None  # normalized embeddings (n, d)
try:
    from sentence_transformers import SentenceTransformer
    if VERIFIED_CHUNKS:
        EMB_MODEL = SentenceTransformer(os.getenv("SENTENCE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        texts = [c["text"] for c in VERIFIED_CHUNKS]
        embs = EMB_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        VERIFIED_EMB = embs  # already normalized
except Exception:
    EMB_MODEL = None
    VERIFIED_EMB = None

def _score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t: return 0.0
    return len(q & t) / (len(q) ** 0.5 * len(t) ** 0.5)

class RetrieveRequest(BaseModel):
    query: str
    k: int = 4

class RetrieveResponse(BaseModel):
    snippets: List[Dict[str, Any]]

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not VERIFIED_CHUNKS:
        return RetrieveResponse(snippets=[])

    # If embedding index available, use cosine similarity
    if VERIFIED_EMB is not None and EMB_MODEL is not None:
        try:
            q_emb = EMB_MODEL.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = (VERIFIED_EMB @ q_emb)  # dot product since normalized
            idxs = _np.argsort(-sims)[:max(1, req.k)]
            out = []
            for i in idxs:
                c = VERIFIED_CHUNKS[int(i)]
                out.append({"source": c["source"], "text": c["text"], "score": float(sims[int(i)])})
            return RetrieveResponse(snippets=out)
        except Exception:
            pass

    # Fallback to token overlap
    scored = sorted(
        ({"source": c["source"], "text": c["text"], "score": _score(req.query, c["text"])} for c in VERIFIED_CHUNKS),
        key=lambda x: x["score"],
        reverse=True
    )
    return RetrieveResponse(snippets=[s for s in scored[:max(1, req.k)]])

class ExplainRequest(BaseModel):
    article_text: str
    classifier_label: str
    classifier_confidence: float
    snippets: List[Dict[str, Any]]

class ExplainResponse(BaseModel):
    explanation: str

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    # If no snippets provided, try to retrieve from verified sources
    snippets = req.snippets
    if not snippets and VERIFIED_CHUNKS:
        try:
            tmp = retrieve(RetrieveRequest(query=req.article_text, k=4))
            snippets = tmp.snippets
        except Exception:
            snippets = []

    bullets = []
    for s in snippets[:4]:
        bullets.append(f"- Source: {s.get('source','?')} | Score: {s.get('score',0):.3f}\\n  Snippet: {s.get('text','')[:240]}")
    joined = "\\n".join(bullets) if bullets else "- (No supporting sources found in the local folder.)"

    # Try LLM (OpenAI HTTP) if OPENAI_API_KEY present
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import requests as _rq
            system = "You are a helpful moderation assistant that produces short, factual explanations."
            user = (
                f"Article text:\n{req.article_text[:1600]}\n\n"
                f"Classifier label: {req.classifier_label} (confidence={req.classifier_confidence:.2f})\n"
                f"Top evidence from verified notes:\n{joined}\n\n"
                f"Explain in 5-8 concise bullet points why the article may be {req.classifier_label}."
            )
            payload = {
                "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.2,
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            resp = _rq.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if resp.ok:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return ExplainResponse(explanation=content)
        except Exception:
            pass

    # Fallback template explanation
    msg = (
        f"Classification: {req.classifier_label} ({req.classifier_confidence*100:.1f}% confidence)\\n"
        f"Why flagged: This article may conflict with your verified notes.\\n"
        f"Top evidence:\\n{joined}\\n"
        f"Note: Lite explainer. Set OPENAI_API_KEY to enable LLM explanations."
    )
    return ExplainResponse(explanation=msg)

class ProbsResponse(BaseModel):
    content: Optional[float] = None
    context: Optional[float] = None
    nli: Optional[float] = None
    final: Optional[float] = None

@app.post("/probs", response_model=ProbsResponse)
def probs(req: ClassifyRequest):
    text = req.text or (extract_text_from_url(req.url) if req.url else "")
    if not text:
        return ProbsResponse()
    out = ProbsResponse()
    try:
        probs = []
        # content
        if ART_CONTENT is not None and ART_TOKENIZER is not None:
            import torch
            enc = ART_TOKENIZER([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                logits = ART_CONTENT(**enc).logits
                p = torch.softmax(logits, dim=1)[0,1].item()
            out.content = float(p)
            probs.append(p)
        # context (no real context json provided at classify-time)
        if ART_CONTEXT_MODEL is not None and ART_CONTEXT_SCALER is not None:
            x = _np.asarray([_ctx_features_from_json("")], dtype=float)
            xs = ART_CONTEXT_SCALER.transform(x)
            p = ART_CONTEXT_MODEL.predict_proba(xs)[:,1][0]
            out.context = float(p)
            probs.append(p)
        # meta
        if ART_META is not None and probs:
            import numpy as _np
            X = _np.asarray(probs).reshape(1,-1)
            out.final = float(ART_META.predict_proba(X)[:,1][0])
    except Exception:
        pass
    return out

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text or (extract_text_from_url(req.url) if req.url else "")
    if not text:
        return ClassifyResponse(label="Unknown", confidence=0.0, rationale="No text provided or URL extraction failed.")

    # If trained artifacts are present, use them; otherwise fallback
    try:
        used_any = False
        feats = []
        probs = []

        # Content branch
        if ART_CONTENT is not None and ART_TOKENIZER is not None:
            try:
                import torch
                enc = ART_TOKENIZER([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
                with torch.no_grad():
                    logits = ART_CONTENT(**enc).logits
                    p_fake = torch.softmax(logits, dim=1)[0,1].item()
                probs.append(p_fake)
                used_any = True
            except Exception:
                pass

        # Context branch (empty context for now)
        if ART_CONTEXT_MODEL is not None and ART_CONTEXT_SCALER is not None:
            import numpy as _np
            x = _np.asarray([_ctx_features_from_json("")], dtype=float)
            xs = ART_CONTEXT_SCALER.transform(x)
            p_ctx = ART_CONTEXT_MODEL.predict_proba(xs)[:,1][0]
            probs.append(p_ctx)
            used_any = True

        # Meta-learner combine if available
        if used_any and ART_META is not None and probs:
            import numpy as _np
            X = _np.asarray(probs).reshape(1,-1)
            p = float(ART_META.predict_proba(X)[:,1][0])
            label = "Fake" if p >= 0.5 else "Real"
            return ClassifyResponse(label=label, confidence=p, rationale="Stacked model")
    except Exception:
        pass

    # Fallback heuristic
    return ClassifyResponse(**heuristic_classify(text))

@app.get("/healthz")
def healthz():
    has_model = bool(ART_META is not None)
    version = None
    if ART_CFG and isinstance(ART_CFG, dict):
        version = ART_CFG.get('version')
    return {"ok": True, "model_loaded": has_model, "version": version}
