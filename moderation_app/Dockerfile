# moderation_app/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app
COPY backend /app/backend

# 1) Torch from the PyTorch CPU index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else from PyPI default
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic \
    trafilatura readability-lxml httpx beautifulsoup4 lxml \
    sentence-transformers faiss-cpu tqdm numpy transformers

EXPOSE 8000
CMD ["python","-m","uvicorn","backend.app:app","--host","0.0.0.0","--port","8000"]
