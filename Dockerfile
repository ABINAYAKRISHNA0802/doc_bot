# ── Stage: Python 3.11 slim base image ──────────────────────────────────────
# slim variant use பண்றோம் — full image (~900MB) பதிலாக ~130MB மட்டுமே
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# gcc: sentence-transformers compile-க்கு தேவை
# curl: health check-க்கு உதவும்
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencies முதலில் copy பண்றோம் (Docker layer cache optimize) ─────────
# requirements மட்டும் copy → install → பிறகு code copy
# இப்படி பண்ணினால் code மாறும்போது pip install மறுபடியும் run ஆகாது
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code copy ────────────────────────────────────────────────────
COPY api/        ./api/
COPY ingestion/  ./ingestion/
COPY retrieval/  ./retrieval/

# ── Environment variables ────────────────────────────────────────────────────
# HuggingFace symlinks warning suppress பண்றோம்
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Python output buffer off — logs உடனே தெரியும்
ENV PYTHONUNBUFFERED=1
# .pyc files create பண்ணாமல் இருக்க
ENV PYTHONDONTWRITEBYTECODE=1

# ── Port expose ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# 30s-க்கு ஒரு முறை / check பண்ணும் — unhealthy container detect பண்ண
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Startup command ──────────────────────────────────────────────────────────
# workers=1: in-memory index_store share ஆக — multiple workers use பண்ணாதீங்க
# Production-ல் persistent index (FAISS.save_local) use பண்ணினால் workers அதிகப்படுத்தலாம்
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
