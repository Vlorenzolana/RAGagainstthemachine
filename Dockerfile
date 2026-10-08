FROM python:3.12-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./
COPY src/ src/
COPY web/ web/

RUN uv pip install --python 3.12 --system .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
    OPENROUTER_DEFAULT_MODEL=openai/gpt-4o-mini

RUN mkdir -p /app/data/raw /app/data/processed /app/data/output

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

ENTRYPOINT ["python", "-m", "student"]
CMD ["help"]
