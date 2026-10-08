# Deployment Guide (OpenRouter)

This project now runs retrieval locally and delegates generation to OpenRouter.
No local Hugging Face model download is required in runtime containers.

## 1) Required environment variables

Backend/container only:

- `OPENROUTER_API_KEY` (**required**)
- `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`)
- `OPENROUTER_DEFAULT_MODEL` (optional, default `openai/gpt-4o-mini`)

> Do not expose these values in frontend code.

## 2) Local run (without Docker)

```bash
uv sync
python -m student index

export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_DEFAULT_MODEL="openai/gpt-4o-mini"

uv run uvicorn student.oracle_api:app --host 0.0.0.0 --port 8000
```

Open:

- Frontend: `http://localhost:8000/frontend`
- Models API: `http://localhost:8000/api/models`
- Ask API: `http://localhost:8000/api/oracle/ask`

## 3) Docker run

```bash
docker build -t ragagainsthemachine:latest .

docker run --rm -p 8000:8000 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e OPENROUTER_BASE_URL="https://openrouter.ai/api/v1" \
  -e OPENROUTER_DEFAULT_MODEL="openai/gpt-4o-mini" \
  -v "$(pwd)/data/raw:/app/data/raw:ro" \
  -v "$(pwd)/data/processed:/app/data/processed" \
  -v "$(pwd)/data/output:/app/data/output" \
  ragagainsthemachine:latest \
  uvicorn student.oracle_api:app --host 0.0.0.0 --port 8000
```

## 4) Docker Compose

Create `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_DEFAULT_MODEL=openai/gpt-4o-mini
```

Development:

```bash
docker-compose up --build backend
```

Production:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## 5) Validation checklist

- `GET /health` returns `{"status":"ok"}`.
- `GET /api/models` returns model list for dropdown.
- `POST /api/oracle/ask` answers using selected model.
- Frontend dropdown loads models from backend and persists choice during the session.
- No key appears in frontend source, API payloads, or normal logs.

## 6) Troubleshooting

- **`OPENROUTER_API_KEY is required`**: set it in environment/.env for backend service.
- **`Unsupported model id`**: choose a model returned by `/api/models`.
- **OpenRouter timeout/error**: check network connectivity and OpenRouter status.
- **Index missing**: run `python -m student index` and mount `data/processed`.
