# 🧠 DLP LaBSE Model Embedding Service (CPU)

Микросервис инференса дообученной модели `FamilFG/dlp-labse` на CPU.

## Запуск контейнера отдельно:

```bash
# Сборка
docker build -t dlp-labse-model .

# Запуск в фоне на порту 8082
docker run -d --name dlp-labse \
  --restart unless-stopped \
  -p 8082:8082 \
  dlp-labse-model
```

## Проверка работоспособности:

```bash
# Healthcheck
curl http://localhost:8082/health

# Тест генерации вектора
curl -X POST http://localhost:8082/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "uşaqlar üçün pambıq corablar"}'
```
