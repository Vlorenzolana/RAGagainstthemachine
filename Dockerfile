# Multi-stage Dockerfile for RAG Against the Machine
# Stage 1: Builder - prepares dependencies and models
# Stage 2: Runtime - lean final image with only essential files

# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM python:3.12-slim-bullseye as builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock* README.md ./
COPY src/ src/

# Install Python dependencies into the system Python (no venv inside image)
RUN uv pip install --python 3.12 --system .

# Download model weights during build (optional - can be done at runtime instead)
# Set cache directories for model downloads
ENV TRANSFORMERS_CACHE=/models/transformers
ENV TORCH_HOME=/models/torch
ENV HF_HOME=/models/huggingface

RUN mkdir -p /models && \
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_id = 'Qwen/Qwen3-0.6B'; \
    print(f'Downloading {model_id}...'); \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='cpu'); \
    print('Model download complete')" || echo "Model download skipped (can download at runtime)"

# ============================================================================
# STAGE 2: Runtime
# ============================================================================
FROM python:3.12-slim-bullseye

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src ./src

# Note: data/ is mounted at runtime via volumes (see docker-compose.yml).
# Do not COPY data/ at build time to keep the image small.

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/models/transformers \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/huggingface \
    PYTHONPATH=/app

# Copy pre-downloaded models from builder stage (may be empty if download skipped)
COPY --from=builder /models /models

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/output /models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; from pathlib import Path; \
    sys.exit(0 if Path('/app/data/processed/bm25_index').exists() else 1)" || echo "Index not found"

# Default command
ENTRYPOINT ["python", "-m", "student"]
CMD ["help"]

# ============================================================================
# Usage:
# ============================================================================
# 
# Build the image:
#   docker build -t ragagainsthemachine:latest .
#
# Run with local data mount:
#   docker run --rm \
#     -v C:\path\to\data:/app/data \
#     -v /tmp/models:/models \
#     ragagainsthemachine:latest \
#     answer "How does vLLM work?"
#
# Interactive shell:
#   docker run -it --rm \
#     -v C:\path\to\data:/app/data \
#     ragagainsthemachine:latest \
#     /bin/bash
#
# ============================================================================
