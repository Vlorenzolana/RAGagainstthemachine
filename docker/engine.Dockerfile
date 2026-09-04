FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY src /app/src

EXPOSE 8001

CMD ["uvicorn", "src.student.llm_engine_stub:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
