*This project has been created as part of the 42 curriculum by vlorenzo.*

# RAG against the machine

Retrieval-Augmented Generation over the [vLLM](https://github.com/vllm-project/vllm) codebase.
The project keeps BM25 retrieval local and delegates answer generation to OpenRouter.

## What changed

- Local inference with `transformers`/`torch` was removed.
- Generation now uses OpenRouter Chat Completions (`/chat/completions`).
- Models are selectable per request (`--model` in CLI and dropdown in frontend).
- API keys stay server-side (`OPENROUTER_API_KEY`), never in browser code.

## Requirements

- Python 3.10–3.12
- Indexed corpus in `data/processed` (run `python -m student index` first)
- OpenRouter key in the backend environment

## Installation

```bash
uv sync
```

## CLI quick start

```bash
# 1) Build BM25 index (one-time per corpus)
python -m student index

# 2) Search
python -m student search "How to configure OpenAI server?" --k 10

# 3) Answer with OpenRouter (default model from OPENROUTER_DEFAULT_MODEL)
OPENROUTER_API_KEY=... python -m student answer "How to configure OpenAI server?" --k 10

# 4) Override model per request
OPENROUTER_API_KEY=... python -m student answer "Explain paged attention" --model "openai/gpt-4o-mini"
```

## Backend API

Start API locally:

```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_DEFAULT_MODEL="openai/gpt-4o-mini"
uv run uvicorn student.oracle_api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /api/models` → filtered OpenRouter models for dropdown (`id`, `name`, `context_length`, prices)
- `POST /api/oracle/ask` → RAG query with optional `model`
- `GET /frontend` → simple frontend with model selector

Example request:

```bash
curl -X POST http://localhost:8000/api/oracle/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does vLLM handle batching?","k":10,"model":"openai/gpt-4o-mini"}'
```

## Docker

Build image:

```bash
docker build -t ragagainsthemachine:latest .
```

Run backend:

```bash
docker run --rm -p 8000:8000 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e OPENROUTER_DEFAULT_MODEL="openai/gpt-4o-mini" \
  -v "$(pwd)/data/raw:/app/data/raw:ro" \
  -v "$(pwd)/data/processed:/app/data/processed" \
  -v "$(pwd)/data/output:/app/data/output" \
  ragagainsthemachine:latest \
  uvicorn student.oracle_api:app --host 0.0.0.0 --port 8000
```

Use compose:

```bash
# .env must define OPENROUTER_API_KEY
cp .env.example .env  # create it manually if needed

docker-compose up --build backend
```

## Security notes

- Never hardcode `OPENROUTER_API_KEY` in source code.
- Never expose API keys to frontend JavaScript.
- Backend validates selected model id and rejects unsupported values.
- OpenRouter failures return controlled HTTP/CLI errors.

## Architecture (RAG)

1. **Ingestion/indexing**: chunk source files and build BM25 index.
2. **Retrieval**: top-k BM25 snippets.
3. **Augmentation**: snippets assembled into bounded context.
4. **Generation**: OpenRouter model answers using only retrieved context.

## Testing

```bash
uv run pytest -q
```
