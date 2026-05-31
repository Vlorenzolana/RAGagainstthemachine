# Deployment Guide

This guide covers deployment on your local Windows machine and on 42 School Linux systems.

## Table of Contents

- [Quick Checklist](#quick-checklist)
- [Local Machine Deployment](#local-machine-deployment)
  - [Prerequisites](#prerequisites)
  - [Docker Installation](#docker-installation)
  - [Building the Image](#building-the-image)
  - [Running Locally](#running-locally)
- [42 School Deployment](#42-school-deployment)
  - [Prerequisites](#42-prerequisites)
  - [Deployment Steps](#deployment-steps)
  - [Troubleshooting on 42](#troubleshooting-on-42)
- [Docker-Compose Quick Start](#docker-compose-quick-start)
- [Performance Tuning](#performance-tuning)

---

## Quick Checklist

Before starting, run the capacity checker:

```bash
python check_capacity.py
```

This validates:
- ✓ Python version (≥ 3.10)
- ✓ System RAM (≥ 4 GB, 8+ recommended)
- ✓ Disk space (≥ 20 GB free)
- ✓ CPU cores (≥ 4 recommended)
- ✓ Docker installed and running
- ✓ vLLM corpus present

---

## Local Machine Deployment

### Prerequisites

1. **Docker Desktop for Windows**
   - Download: https://www.docker.com/products/docker-desktop
   - Includes Docker CLI, Docker Daemon, and Docker Compose
   - Requires Windows 10+ with WSL2 enabled

2. **WSL2 (Windows Subsystem for Linux 2)**
   ```powershell
   # In PowerShell as Administrator
   wsl --install
   wsl --set-default-version 2
   ```

3. **vLLM corpus** in `data/raw/vllm-0.10.1/`
   ```bash
   # Option A: Extract provided zip
   Expand-Archive -Path vllm-0.10.1.zip -DestinationPath data/raw/

   # Option B: Clone from GitHub
   git clone https://github.com/vllm-project/vllm.git data/raw/vllm-0.10.1
   ```

### Docker Installation

#### Step 1: Download Docker Desktop

1. Go to https://www.docker.com/products/docker-desktop
2. Click "Download for Windows"
3. Run the installer
4. Check "Use WSL 2 instead of Hyper-V" (if available)
5. Restart your computer when prompted

#### Step 2: Verify Installation

```powershell
docker --version
docker run hello-world
```

Expected output:
```
Docker version 25.0.0 (or newer)
Hello from Docker!
```

#### Step 3: Configure Resources

Docker Desktop needs sufficient resources. On your system (~7.7 GB RAM):

1. Open **Docker Desktop Settings**
2. Go to **Resources**
3. Set:
   - **CPUs:** 4 (leave room for OS)
   - **Memory:** 4 GB (leave room for OS)
   - **Disk image size:** 50 GB (for models + index + corpus)
4. Click **Apply & Restart**

Check allocation:

```powershell
docker system df
```

### Building the Image

**First build** (downloads dependencies + models):

```bash
docker build -t ragagainsthemachine:latest .
```

**Build time:** ~15 minutes (depends on internet speed)

**Resulting image size:** ~4 GB (includes Python, torch, transformers, model weights)

#### Speed up subsequent builds:

Use Docker BuildKit for layer caching:

```bash
# Enable BuildKit (one-time)
$env:DOCKER_BUILDKIT=1

# Build with improved caching
docker build --progress=plain -t ragagainsthemachine:latest .
```

### Running Locally

#### Option A: Using docker-compose (Recommended)

```bash
# View help
docker-compose run --rm rag help

# Index the corpus
docker-compose run --rm rag index

# Search
docker-compose run --rm rag search "how does vllm handle batching?" --k 5

# Answer
docker-compose run --rm rag answer "Explain paged attention" --k 10
```

#### Option B: Using docker run directly

```bash
# Ensure data directory is mounted
$data_dir = "C:\Users\Vanessa\Documents\practice\RAGagainsthemachine\data"

# Index
docker run --rm `
    -v "${data_dir}:/app/data" `
    -v "C:\tmp\rag_models:/models" `
    -m 4g `
    --cpus 4 `
    ragagainsthemachine:latest `
    index

# Answer query
docker run --rm `
    -v "${data_dir}:/app/data" `
    -v "C:\tmp\rag_models:/models" `
    -m 4g `
    --cpus 4 `
    ragagainsthemachine:latest `
    answer "Your question here" --k 10
```

#### Option C: Interactive shell

```bash
docker run -it --rm `
    -v "${data_dir}:/app/data" `
    -v "C:\tmp\rag_models:/models" `
    ragagainsthemachine:latest `
    /bin/bash

# Inside container
python -m student help
python -m student search "query"
```

---

## 42 School Deployment

### 42 Prerequisites

1. **SSH access** to a 42 machine (Linux, not Windows)
2. **Docker** installed and running (usually pre-configured)
3. **Git** for cloning the repository
4. **Enough disk space** in `/tmp` or a mounted shared directory

### Checking 42 Environment

```bash
# On the 42 machine
docker --version
docker ps

# Check available space
df -h /tmp
df -h /home

# Check CPU and memory
nproc
free -h
```

If `/tmp` is too small (<30 GB), use a larger mount:

```bash
# Find available mounts
mount | grep -E "^/dev"
df -h
```

### Deployment Steps

#### Step 1: Clone/Upload Project

```bash
# Option A: Clone from your git repo
git clone https://github.com/yourusername/RAGagainsthemachine.git
cd RAGagainsthemachine

# Option B: Upload via SCP
scp -r ./RAGagainsthemachine your-login@42.fr:~/
ssh your-login@42.fr
cd RAGagainsthemachine
```

#### Step 2: Check Capacity

```bash
python check_capacity.py
```

Note any warnings about disk space or resources.

#### Step 3: Build Docker Image

```bash
# Standard build
docker build -t ragagainsthemachine:latest .

# If /tmp is small, use a larger directory
docker build \
    --build-arg TMPDIR=/mnt/shared/tmp \
    -t ragagainsthemachine:latest .
```

**Build time on 42:** ~10–20 minutes (depending on machine speed)

#### Step 4: Run Commands

Use `docker-compose.prod.yml` for production settings:

```bash
# Run with production compose file
docker-compose -f docker-compose.prod.yml run --rm rag-42 index

# Or use docker run directly with 42-specific settings
docker run --rm \
    --memory 2g \
    --cpus 2 \
    -v /tmp/rag_data:/app/data \
    -v /tmp/rag_models:/models \
    ragagainsthemachine:latest \
    answer "your question"
```

#### Step 5: Handle Large Models

If `/tmp` is too small for models (~5 GB):

```bash
# Use a larger mount (e.g., /mnt/shared)
export HF_HOME=/mnt/shared/.cache/huggingface
export TORCH_HOME=/mnt/shared/.cache/torch

docker run --rm \
    -e HF_HOME=$HF_HOME \
    -e TORCH_HOME=$TORCH_HOME \
    -v /mnt/shared:/cache \
    -v /tmp/rag_data:/app/data \
    ragagainsthemachine:latest \
    answer "query"
```

Or create a wrapper script `deploy_42.sh`:

```bash
#!/bin/bash

# 42 deployment wrapper
set -e

# Configure cache directories
export HF_HOME=${HF_HOME:-/mnt/shared/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/mnt/shared/.cache/torch}

# Ensure directories exist
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Run docker with proper environment
docker run --rm \
    --memory 2g \
    --cpus 2 \
    -e HF_HOME="$HF_HOME" \
    -e TORCH_HOME="$TORCH_HOME" \
    -v /tmp/rag_data:/app/data \
    -v /mnt/shared/.cache:/cache \
    ragagainsthemachine:latest \
    "$@"
```

Run it:

```bash
chmod +x deploy_42.sh
./deploy_42.sh answer "your question"
```

### Troubleshooting on 42

#### Problem: "Disk space exceeded"

```bash
# Check disk usage
docker system df

# Clean up old images/containers
docker system prune -a

# Use larger mount for models
export HF_HOME=/mnt/shared/.cache/huggingface
```

#### Problem: "Out of memory"

```bash
# Reduce memory allocated to container
docker run --memory 1.5g ...

# Or reduce model/context size
docker run ... answer "query" --max_context_chars 4000
```

#### Problem: "Model download fails"

```bash
# Pre-download model outside container
huggingface-cli download Qwen/Qwen3-0.6B

# Then run container with model cache mounted
docker run -v ~/.cache/huggingface:/models ...
```

#### Problem: "Index not found"

```bash
# Build index first
docker run -v /tmp/rag_data:/app/data ... index

# Verify
ls -la /tmp/rag_data/processed/
```

---

## Docker-Compose Quick Start

### Using docker-compose.yml (Local Development)

```bash
# View all commands
docker-compose run --rm rag help

# Build index
docker-compose run --rm rag index --max_chunk_size 2000

# Search
docker-compose run --rm rag search "your query" --k 10

# Answer
docker-compose run --rm rag answer "your question" --k 10

# Batch process
docker-compose run --rm rag search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset.json

# Evaluate
docker-compose run --rm rag evaluate \
    --student_answer_path data/output/results.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset.json

# Interactive shell
docker-compose run --rm rag /bin/bash
```

### Using docker-compose.prod.yml (42 School)

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml run --rm rag-42 help

# Scale resource limits differently for batch jobs
docker-compose -f docker-compose.prod.yml run \
    -e MEMORY=1.5g \
    --rm rag-42 answer "query"
```

---

## Performance Tuning

### On Your Local Machine

| Setting | Default | Tuning |
|---|---|---|
| **Memory** | 4 GB | Increase to 6–8 GB if available |
| **CPUs** | 4 | Increase to 6–8 if available |
| **Chunk size** | 2000 | Reduce to 1000 for faster indexing |
| **Max tokens** | 384 | Reduce to 256 for faster generation |
| **Temperature** | 0.0 | Keep at 0.0 for consistency |

### On 42 School

| Setting | Local | 42 Conservative |
|---|---|---|
| **Memory** | 4 GB | 1.5–2 GB (shared resource) |
| **CPUs** | 4 | 2 (shared resource) |
| **Chunk size** | 2000 | 1500 |
| **Max context** | 8000 | 4000 |
| **Model** | Qwen/Qwen3-0.6B | Qwen/Qwen3-0.5B (smaller) |

### Enable GPU (if available)

```bash
# Check for GPU
docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi

# Run with GPU
docker run --rm --gpus all \
    -v /tmp/rag_data:/app/data \
    ragagainsthemachine:latest \
    answer "query" \
    --model "Qwen/Qwen3-0.6B"
```

---

## Tips & Best Practices

1. **First-time setup:** Allow 20–30 minutes for initial build (downloads models, builds index)
2. **Reuse images:** Don't rebuild after every change; only rebuild if dependencies change
3. **Use volumes:** Always mount `data/` to persist results and models
4. **Monitor resources:** Use `docker stats` to watch memory/CPU usage
5. **Batch operations:** For large datasets, split into smaller batches to avoid memory issues
6. **Cache models:** Download models once, reuse across multiple container runs
7. **Test locally first:** Before deploying to 42, ensure everything works on your machine

---

## Support

If you encounter issues, check:

1. [README.md](README.md) for general help
2. [Troubleshooting section](README.md#troubleshooting) for common problems
3. Logs: `docker logs <container_id>`
4. System: `docker system df` and `docker stats`

---

## References

- Docker Docs: https://docs.docker.com/
- Docker-Compose: https://docs.docker.com/compose/
- vLLM: https://github.com/vllm-project/vllm
- 42 School: https://42.fr/
