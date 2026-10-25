FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Устанавливаем curl для healthcheck и dumb-init для корректного graceful shutdown
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Устанавливаем легкий CPU-only PyTorch и зависимости
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .

EXPOSE 8082

ENTRYPOINT ["dumb-init", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]
