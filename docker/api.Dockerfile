FROM python:3.10-slim

WORKDIR /app

# Instalación runtime mínima
RUN pip install --no-cache-dir fastapi uvicorn[standard] requests

# Copiar código
COPY . /app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Ejecutar con 1 worker (consumo mínimo)
CMD ["uvicorn", "src.student.oracle_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
