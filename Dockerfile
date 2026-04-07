# Municipal Policy Simulation — OpenEnv
# HuggingFace Spaces compatible (port 7860, non-root user)

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY core/           ./core/
COPY models.py       ./
COPY environment.py  ./
COPY tasks.py        ./
COPY app.py          ./
COPY baseline.py     ./
COPY inference.py    ./
COPY openenv.yaml    ./

# Non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port 7860 & \
    sleep 3 && \
    python inference.py
