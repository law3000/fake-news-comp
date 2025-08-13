import os
import json
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from app import app

client = TestClient(app)


def test_healthz_structure():
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert data.get("ok") is True
    assert "model_loaded" in data
    # version and features are optional
    assert "version" in data
    assert "features" in data


def test_metrics_structure():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "cache" in data and isinstance(data["cache"], dict)
    assert set(["size", "hits", "misses"]).issubset(set(data["cache"].keys()))
    assert "model" in data and isinstance(data["model"], dict)


def test_probs_final_range():
    payload = {"text": "This is a quick test article about policy and data."}
    r = client.post("/probs", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Always present keys
    assert "content" in data
    assert "final" in data
    # If final is present, it should be a probability in [0,1]
    if data["final"] is not None:
        assert 0.0 <= data["final"] <= 1.0


def test_retrieve_cache_toggle():
    # Rebuild to clear caches and embeddings
    r0 = client.post("/rag/rebuild")
    assert r0.status_code == 200
    # First call should be cached = False (miss)
    rq = {"query": "fact checking guidance", "k": 2}
    r1 = client.post("/retrieve", json=rq)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "snippets" in d1
    assert d1.get("cached") in (False, None)  # default False/None on first build
    # Second call should be cached = True
    r2 = client.post("/retrieve", json=rq)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2.get("cached") is True

