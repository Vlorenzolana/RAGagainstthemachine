FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn requests

COPY src /app/src
COPY data /app/data
COPY web /app/web

EXPOSE 8000

CMD ["uvicorn", "src.student.oracle_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
