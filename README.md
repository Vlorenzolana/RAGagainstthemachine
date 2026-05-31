*This project has been created as part of the 42 curriculum by student.*

# RAG against the machine

Retrieval-Augmented Generation over the [vLLM](https://github.com/vllm-project/vllm)
codebase. Given a question about vLLM, the system finds the most relevant code
and documentation snippets, then asks a small local LLM
(`Qwen/Qwen3-0.6B`) to answer using only that retrieved context.

## Table of Contents

- [Description](#description)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Indexing](#indexing)
  - [Search](#search)
  - [Answer](#answer)
  - [Batch Operations](#batch-operations)
  - [Evaluation](#evaluation)
- [Docker Deployment](#docker-deployment)
  - [Building the Image](#building-the-image)
  - [Running Locally](#running-locally)
  - [Deployment on 42 School](#deployment-on-42-school)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

---

## Description

The project implements the four canonical RAG stages:

1. **Ingestion / indexing** – walks the vLLM corpus, chunks Python code with
   an AST-aware strategy and text/Markdown by paragraph, then builds a sparse
   BM25 index (`bm25s`) with code-aware tokenisation.
2. **Retrieval** – top-k BM25 search returning `MinimalSource` records
   (`file_path`, `first_character_index`, `last_character_index`).
3. **Augmentation** – retrieved chunks are assembled into a budgeted context
   block with explicit source headers.
4. **Generation** – `Qwen/Qwen3-0.6B` generates a grounded answer through a
   strict system prompt that forbids hallucinations.

Evaluation is `recall@k` with a 5% overlap rule (matching the spec).

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 20 GB | 50 GB (includes model cache) |
| **CPU** | 4 cores | 8+ cores |
| **Time (indexing)** | N/A | ≤ 5 min (reference corpus) |

> **Note:** The model weights (~300 MB) and index will be cached locally after first run.
> On 42 School machines, disk space in `/tmp` may be limited; set `HF_HOME` and `TORCH_HOME` 
> to a larger mounted directory if needed.

---

## Installation

### Prerequisites

1. **Python 3.10+** – Download from [python.org](https://www.python.org/downloads/)
2. **uv** package manager – Fast, reliable Python package installer
   ```bash
   # On Windows, macOS, or Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or on Windows using PowerShell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **vLLM repository** – Extract to `data/raw/vllm-0.10.1/`
   - If you have the provided zip file, extract it to that location
   - Or clone: `git clone https://github.com/vllm-project/vllm.git data/raw/vllm-0.10.1`

### Local Setup

```bash
# Clone or navigate to the project directory
cd RAGagainsthemachine

# Install dependencies using uv
make install
# Or manually:
uv sync

# Verify installation
make run  # Should show the help menu
```

**That's it!** All models will download on first use.

---

## Quick Start

```bash
# 1. View help and examples
python -m student help

# 2. Build the index (one-time, ~5 minutes)
python -m student index

# 3. Search for relevant code
python -m student search "How to configure OpenAI server?" --k 10

# 4. Get a grounded answer
python -m student answer "How to configure OpenAI server?" --k 10
```

All output is **JSON-formatted** for easy parsing and integration.

---

## Commands

### Indexing

Build a BM25 index from the vLLM corpus. **Must be run before any search or answer commands.**

```bash
python -m student index \
    --max_chunk_size 2000 \
    --raw_dir data/raw \
    --processed_dir data/processed
```

**Parameters:**
- `--max_chunk_size` (int): Characters per chunk. Smaller chunks (1000–2000) = more precise but slower. Default: 2000
- `--raw_dir` (str): Path to vLLM repo. Default: `data/raw` (auto-detects subdirs)
- `--processed_dir` (str): Output directory for index. Default: `data/processed`

**Output:** Creates `data/processed/chunks.json` and `data/processed/bm25_index/`

---

### Search

Retrieve the top-k most relevant code/documentation snippets for a query.

```bash
python -m student search "How to initialize the tokenizer?" --k 10
```

**Parameters:**
- `query` (str, required): Search query
- `--k` (int): Number of results. Default: 10
- `--processed_dir` (str): Index directory. Default: `data/processed`

**Output:** JSON with file paths, character indices, relevance scores, and text previews.

---

### Answer

Full RAG pipeline: retrieve context → generate grounded answer.

```bash
python -m student answer "Explain the paged attention mechanism" \
    --k 10 \
    --max_new_tokens 384 \
    --temperature 0.0
```

**Parameters:**
- `query` (str, required): Question to answer
- `--k` (int): Number of source snippets to retrieve. Default: 10
- `--processed_dir` (str): Index directory. Default: `data/processed`
- `--model` (str): Hugging Face model ID. Default: `Qwen/Qwen3-0.6B`
- `--max_context_chars` (int): Max context length. Default: 8000
- `--max_new_tokens` (int): Max tokens to generate. Default: 384
- `--temperature` (float): Sampling temperature (0.0 = deterministic). Default: 0.0
- `--thinking` (bool): Enable chain-of-thought. Default: False

**Output:** JSON with question, generated answer, source references, and scores.

---

### Batch Operations

#### Search Dataset

Process all questions in a JSON file and save results.

```bash
python -m student search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results
```

#### Answer Dataset

Generate answers for pre-computed search results.

```bash
python -m student answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer \
    --k 10
```

---

### Evaluation

Compute `recall@k` against ground-truth annotations.

```bash
python -m student evaluate \
    --student_answer_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --ks 1 3 5 10
```

**Output:** Table of recall scores at each k value.

---

## Docker Deployment

### Building the Image

```bash
docker build -t ragagainsthemachine:latest .
```

This creates a multi-stage image (~4 GB) with:
- Python 3.12
- All dependencies (transformers, torch, bm25s, etc.)
- Pre-indexed vLLM corpus (if `data/raw/vllm-*/` exists)

**Build time:** ~10–15 minutes on first build (downloading models and creating index).

### Running Locally

#### Option 1: With pre-built index (fast)

```bash
# First, build locally to pre-compute the index
python -m student index

# Then build Docker image (index is cached in container)
docker build -t ragagainsthemachine:latest .

# Run queries inside the container
docker run --rm \
    -v C:\Users\Vanessa\Documents\practice\RAGagainsthemachine\data\output:/app/data/output \
    ragagainsthemachine:latest \
    python -m student answer "How does vLLM handle batching?" --k 5
```

#### Option 2: Interactive shell

```bash
docker run -it --rm \
    -v C:\Users\Vanessa\Documents\practice\RAGagainsthemachine\data:/app/data \
    ragagainsthemachine:latest \
    /bin/bash

# Inside container:
python -m student search "your query" --k 10
python -m student answer "your question"
```

#### Memory and Resource Limits

On your machine (~7.7 GB RAM, 10 cores), use:

```bash
docker run --rm \
    --memory 4g \
    --cpus 4 \
    -v C:\Users\Vanessa\Documents\practice\RAGagainsthemachine\data:/app/data \
    ragagainsthemachine:latest \
    python -m student answer "query"
```

---

### Deployment on 42 School

42 School machines have Linux with Docker available. Here's how to deploy:

#### Step 1: Push to School Repository

```bash
# Build
docker build -t ragagainsthemachine:latest .

# Tag with 42 registry (if required by your school)
docker tag ragagainsthemachine:latest 42.fr/your-username/ragagainsthemachine:latest

# Push (requires authentication)
docker push 42.fr/your-username/ragagainsthemachine:latest
```

#### Step 2: Run on 42 Machine

```bash
# Pull the image
docker pull 42.fr/your-username/ragagainsthemachine:latest

# Or use local image if already built:
docker load < ragagainsthemachine.tar

# Run with appropriate resource limits
docker run --rm \
    --memory 2g \
    --cpus 2 \
    -v /tmp/rag_data:/app/data \
    ragagainsthemachine:latest \
    python -m student answer "your question"
```

#### Disk Space Considerations for 42

42 School `/tmp` directories are often limited. Create a script to handle this:

```bash
#!/bin/bash

# Check available space
AVAILABLE=$(df /tmp | tail -1 | awk '{print $4}')
if [ "$AVAILABLE" -lt 20000000 ]; then  # Less than 20GB
    echo "Not enough space in /tmp for models (~5GB) + data"
    exit 1
fi

# Set Hugging Face cache outside /tmp if needed
export HF_HOME=/larger/mounted/path/.cache/huggingface
export TORCH_HOME=/larger/mounted/path/.cache/torch

docker run --rm \
    -e HF_HOME="$HF_HOME" \
    -e TORCH_HOME="$TORCH_HOME" \
    -v /tmp/rag_data:/app/data \
    ragagainsthemachine:latest \
    python -m student answer "your question"
```

---

## Troubleshooting

### Issue: "Raw corpus directory not found"
**Solution:** Ensure vLLM is extracted to `data/raw/vllm-*/`:
```bash
# Check if directory exists
ls -la data/raw/

# If not, extract the provided zip or clone:
git clone https://github.com/vllm-project/vllm.git data/raw/vllm-0.10.1
```

### Issue: "Index not found"
**Solution:** Build the index first:
```bash
python -m student index
```

### Issue: Out of memory during indexing
**Solution:** Reduce chunk size:
```bash
python -m student index --max_chunk_size 1000
```

### Issue: Generation is very slow
**Solutions:**
1. Use GPU if available:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m student answer "query"
   ```
2. Use a smaller model:
   ```bash
   python -m student answer "query" --model "Qwen/Qwen2-0.5B"
   ```
3. Reduce context or tokens:
   ```bash
   python -m student answer "query" --max_context_chars 4000 --max_new_tokens 256
   ```

### Issue: Docker image is huge (~4 GB)
**Solution:** Use a smaller base image or Python slim variant. Edit `Dockerfile` to use `python:3.12-slim-bullseye`.

### Issue: Container runs out of disk space
**Solution:** Increase Docker's disk allocation:
```bash
# Check current usage
docker system df

# Increase Docker desktop allocation (Windows/Mac):
# Settings → Resources → Disk image size → increase to 50+ GB
```

---

## Architecture

```
                ┌──────────────────┐
                │  data/raw/vllm-* │
                └────────┬─────────┘
                         │  iter_corpus_files
                         ▼
                ┌──────────────────┐
   chunking.py  │  build_chunks    │  → Chunk[]
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
   indexing.py  │ tokenize + bm25s │  → data/processed/{chunks,bm25_index}
                └────────┬─────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌────────┐        ┌──────────┐         ┌──────────┐
│ search │        │  answer  │         │ evaluate │
└────────┘        └──────────┘         └──────────┘
    │                    │
    │          ┌─────────┴─────────┐
    │          ▼                   ▼
    │       ┌──────────┐      ┌──────────┐
    │       │retrieval │      │generation│
    │       └──────────┘      └──────────┘
    │          │                   │
    └──────────┴───────────────────┘
               │
               ▼
           JSON output
```

---

## Development

### Lint & Test

```bash
make lint        # flake8 + mypy
make lint-strict # mypy --strict (stricter checks)
make test        # Run pytest
make clean       # Clean caches
```

---

## Performance Notes

- **Indexing:** ~5 minutes for reference corpus (vLLM 0.10.1)
- **Search:** <100 ms per query
- **Answer generation:** 5–30 seconds (CPU-bound; use GPU for faster inference)
- **Model download:** First run automatically fetches ~300 MB (Qwen/Qwen3-0.6B)

---

## License & Attribution

42 Curriculum Project | 2024

                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐       ┌────────────────────┐
   │ BM25Retriever   │       │  AnswerGenerator   │
   │ (retrieval.py)  │       │  (generation.py,   │
   │                 │       │   Qwen3-0.6B)      │
   └────────┬────────┘       └─────────┬──────────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
                ┌──────────────┐
                │ CLI (Fire)   │  index / search / search_dataset /
                │  cli.py      │  answer / answer_dataset / evaluate
                └──────────────┘
                       ▲
                       │
                ┌──────────────┐
                │ evaluation.py│  recall@k vs. AnsweredQuestion sources
                └──────────────┘
```

All command outputs adhere to the spec’s pydantic models
(`StudentSearchResults`, `StudentSearchResultsAndAnswer`).

## Chunking strategy

* **Python files (`.py`)** – parsed with `ast`. Each top-level function or
  class becomes one chunk; module-level header (imports, constants) becomes
  its own chunk. Bodies larger than `--max_chunk_size` are sub-split on
  newline boundaries to avoid breaking tokens. Files that fail to parse fall
  back to fixed-window chunking.
* **Markdown / RST / plain text** – split on blank-line paragraph breaks,
  then greedily merged into chunks of at most `--max_chunk_size`
  characters. Oversized paragraphs are sub-split on newline boundaries.
* **Other config-like files** (`.yaml`, `.toml`, `.json`, …) are processed
  with the text strategy because they are typically short.

Characters offsets are recorded against the *original* file content so the
emitted `MinimalSource` entries point to exact byte ranges.

## Retrieval method

* **Backbone** – BM25 via the `bm25s` library.
* **Tokenisation** – a custom tokenizer that lowercases, drops a small
  stoplist and additionally **splits identifiers** on `snake_case` and
  `CamelCase` while keeping the compound form. This single change is what
  unlocks recall on questions like *“What method needs to be overridden in
  BaseProcessingInfo…”* whose answer mentions `get_supported_mm_limits`.
* **Ranking** – pure BM25 scores (ascending k applied per query).

The system is intentionally a strong sparse baseline: it loads in
sub-second time on a cold start (after the model is cached), satisfies the
≤ 60 s cold-start latency target, and scores well above the 80% docs /
50% code `recall@5` thresholds in our experiments.

## Performance analysis

| Step | Latency target | Notes |
|------|----------------|-------|
| Indexing the full vLLM repo | ≤ 5 min | Single-process; dominated by tokenisation. |
| Cold-start retrieval | ≤ 60 s | Loading bm25s index is ~1 s; LLM load is the long pole when answering. |
| Warm retrieval, 1000 queries | ≤ 90 s | bm25s batch ranking is well under target. |

`recall@5` on the public datasets exceeds the spec floor (80% docs / 50%
code) in our runs. Higher-recall configurations (more candidates, hybrid
retrieval) are easy to bolt on as bonus work.

## Design decisions

* **Sparse-only retrieval first.** BM25 with identifier-aware tokenisation
  gets us above the recall floor without any embedding model, which keeps
  the cold-start time low and avoids GPU dependencies during retrieval.
* **AST-based code chunking.** Function/class boundaries make each chunk
  semantically meaningful, which improves both retrieval precision and the
  quality of the LLM context window.
* **Strict, source-grounded prompt.** The generator’s system prompt
  forbids reasoning beyond the supplied context, mitigating hallucination
  on questions the corpus cannot answer.
* **Greedy paragraph merging for text.** Keeps Markdown sections coherent
  while staying within the `--max_chunk_size` budget.
* **Type-safe data flow.** Every persisted artefact is validated by a
  Pydantic v2 model defined in `src/student/models.py`.

## Challenges faced

* **bm25s API surface.** Tokenisation expects either an internal
  `Tokenized` object or list-of-list inputs – we standardise on the latter
  for clarity.
* **Character offsets across encodings.** All file reads are normalised to
  UTF-8 (with replacement on errors) so the indices we store match what the
  evaluator sees.
* **Qwen3 “thinking” traces.** Qwen3 may emit `<think>...</think>` blocks
  even with `enable_thinking=False`. The generator strips them defensively.

## Example usage

```bash
$ uv run python -m student answer "What method needs to be overridden in BaseProcessingInfo?"
{
  "question": "What method needs to be overridden in BaseProcessingInfo?",
  "answer": "Override the abstract method `get_supported_mm_limits` ...",
  "retrieved_sources": [
    {
      "file_path": "docs/contributing/model/multimodal.md",
      "first_character_index": 12034,
      "last_character_index": 13892
    }
  ],
  "k": 10
}
```

## Resources

* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020).
* Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009).
* `bm25s` documentation – <https://github.com/xhluca/bm25s>
* `transformers` documentation – <https://huggingface.co/docs/transformers>
* Qwen3 model card – <https://huggingface.co/Qwen/Qwen3-0.6B>
* `pydantic` v2 docs – <https://docs.pydantic.dev/latest/>
* Python Fire docs – <https://github.com/google/python-fire>

### Use of AI

AI assistants were used to:

* draft the initial skeleton of the CLI and Pydantic models,
* brainstorm chunking heuristics (AST-based vs. paragraph-based),
* discuss BM25 tokenisation strategies for identifiers,
* sanity-check the recall@k overlap rule.

All AI-generated suggestions were reviewed, edited, and validated by the
author. The final code, tests, and design decisions are owned and
understood by the author.
