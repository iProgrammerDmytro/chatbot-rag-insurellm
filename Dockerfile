FROM python:3.11-slim-bookworm

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1 \
    PORT=8000 \
    APP_MODULE="app.main:app" \
    HF_HOME=/cache \
    TORCH_HOME=/cache \
    SENTENCE_TRANSFORMERS_HOME=/cache \
    HF_HUB_DISABLE_TELEMETRY=1

# Install tiny native deps. libgomp1 is often needed by torch/OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy source code (heavy data like chroma-db and knowledge-base are excluded via .dockerignore)
COPY . /app

# Ensure runtime dirs exist & owned by non-root
RUN mkdir -p /chroma-db /knowledge-base /cache \
    && chown -R app:app /app /cache

USER app

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).getcode()==200 else 1)"

CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
