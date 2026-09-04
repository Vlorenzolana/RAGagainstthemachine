FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY . /app

EXPOSE 8001

CMD ["uvicorn", "src.student.llm_engine_stub:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
