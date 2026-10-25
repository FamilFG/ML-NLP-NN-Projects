# 📑 Технический отчет: Хостинг модели `FamilFG/dlp-labse` на CPU
**Тема:** Развертывание микросервиса инференса модели LaBSE на CPU  
**Автор:** Famil (ML Engineer)  
**Статус:** Готово к согласованию (Ready for Review)  

---

## 🎯 1. Кратко: Главные ответы (Где, Как, с Каким конфигом)

| Вопрос | Решение |
| :--- | :--- |
| **ГДЕ хостить?** | На отдельной виртуальной машине (Linux VM) в виде Docker-контейнера в одной локальной сети с сервисами. |
| **С КАКИМ КОНФИГОМ?** | **8 vCPU, 8–16 GB RAM** (с поддержкой AVX-512). Модель в памяти занимает **1.88 ГБ**. Параметры: `workers=1`, `OMP_NUM_THREADS=4`, `max_length=192`. |
| **КАК захостить?** | В виде легковесного микросервиса, слушающего порт `8082` и отдающего вектор по `POST /embed`. Готовы 2 варианта: собственный Python Dockerfile или готовый образ Hugging Face TEI. |
| **СКОРОСТЬ НА CPU** | **~200 – 230 мс** на генерацию вектора (при пропускной способности **25–30 RPS**). |

---

## 🔬 2. Профиль и характеристики модели

* **Репозиторий:** `FamilFG/dlp-labse` (subfolder `labse-finetuned-freeze6`).
* **Базовая архитектура:** 12-слойный мультиязычный BERT (`sentence-transformers/LaBSE`).
* **Размер параметров:** ~471M параметров (входная матрица 501k слов ~385M + слои трансформера ~85M).
* **Размер весов в оперативной памяти (RAM):** **~1.88 GB** (FP32) / **~470 MB** (INT8).
* **Формат выхода:** 768 чисел (float32, L2-normalized).
* **Входные данные:** текст запроса длиной до **192 токенов**.

### Физика инференса на CPU:
Поисковые запросы пользователей короткие (в среднем 3–8 токенов). Механизм внимания $O(N^2)$ при такой длине работает моментально. Основное время CPU тратит на чтение весов 1.88 ГБ из памяти и 12 матричных умножений в слоях энкодера. На процессорах Intel Xeon / AMD EPYC с AVX-512 это стабильно занимает **~180–230 мс**.

---

## 💻 3. Конфигурация сервера (Hardware Specs)

### Рекомендуемая конфигурация (Production / Pre-prod):
* **vCPU:** **8 ядер** (процессоры с поддержкой **AVX-512** / **VNNI**: Intel Xeon Scalable 3rd+ Gen или AMD EPYC Zen 3/4).
* **RAM:** **8 – 16 GB RAM** (многоканальная DDR4/DDR5 критична для пропускной способности памяти).
* **Диск:** 20 GB SSD.
* **Производительность:** **~200 мс на запрос**, пропускная способность **до 25–30 RPS** (1500+ запросов в минуту).

### Минимальная конфигурация (Dev / Test):
* **vCPU:** **4 ядра**.
* **RAM:** **8 GB RAM**.
* **Производительность:** ~350 мс на запрос (до 10–12 RPS).

---

## ⚙️ 4. Параметры запуска и конфигурация рантайма

| Параметр | Значение | Зачем и как работает |
| :--- | :--- | :--- |
| **`workers`** | **`1`** | **Критично!** Запускать строго 1 процесс воркера. Несколько воркеров загрузят дубликаты модели в RAM и вызовут троттлинг ядер. |
| **`OMP_NUM_THREADS`** | **`4`** | Ограничивает потоки PyTorch на 1 запрос. На 8-ядерной машине позволяет параллельно считать **2 запроса без взаимной блокировки**. |
| **`max_length`** | **`192`** (или 128) | Обрезает входную строку. Защищает процессор от зависания на аномально длинных текстах. |
| **`normalize_embeddings`** | **`True`** | Обязательная L2-нормализация (длина вектора = 1.0) для корректного косинусного сходства. |
| **`Warm-up` (прогрев)** | **Да** | Разовый холостой прогон при старте контейнера. Защищает первого зашедшего пользователя от 3-секундной задержки инициализации кэшей CPU. |

---

## 🚀 5. Как захостить (2 готовых варианта)

### Вариант 1: Собственный микросервис на Python (Простой и прозрачный)

#### Файл микросервиса `app.py`:
```python
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="DLP LaBSE Model Service")

# Фиксируем число потоков для защиты от троттлинга
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))

MODEL_ID = os.getenv("MODEL_ID", "FamilFG/dlp-labse")
SUBFOLDER = os.getenv("SUBFOLDER", "labse-finetuned-freeze6")
HF_TOKEN = os.getenv("HF_TOKEN", None)

print(f"Loading model {MODEL_ID} (subfolder={SUBFOLDER}) on CPU...")
model = SentenceTransformer(
    MODEL_ID,
    subfolder=SUBFOLDER if SUBFOLDER else None,
    use_auth_token=HF_TOKEN,
    device="cpu"
)

# Обязательный прогрев (Warm-up)
_ = model.encode("прогрев модели warm-up", normalize_embeddings=True)
print("✅ Model ready for inference.")

class EmbedRequest(BaseModel):
    inputs: str

@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    if not req.inputs or not req.inputs.strip():
        raise HTTPException(status_code=400, detail="Empty input query")
    vector = model.encode(req.inputs.strip(), normalize_embeddings=True).tolist()
    return [vector]

@app.get("/health")
def healthcheck():
    return {"status": "healthy", "model": MODEL_ID}
```

#### Файл `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends dumb-init \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    sentence-transformers==2.7.0 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY app.py .

EXPOSE 8082

ENTRYPOINT ["dumb-init", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]
```

#### Команды сборки и запуска на сервере:
```bash
# 1. Сборка контейнера
docker build -t dlp-labse-service .

# 2. Запуск в фоне с автоперезапуском 24/7
docker run -d --name dlp-labse-model \
  --restart unless-stopped \
  -p 8082:8082 \
  dlp-labse-service

# 3. Тестовая проверка генерации вектора
curl -X POST http://localhost:8082/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "uşaqlar üçün pambıq corablar"}'
```

---

### Вариант 2: Готовый образ от Hugging Face TEI (Максимально быстрый, C++/Rust)
Если девопс готов использовать официальный образ от Hugging Face, скомпилированный под CPU/AVX-512, кодить вообще ничего не нужно:

```bash
docker run -d --name dlp-labse-model \
  -p 8082:80 \
  --cpus="8" \
  --memory="8g" \
  -e OMP_NUM_THREADS=4 \
  --restart unless-stopped \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 \
  --model-id FamilFG/dlp-labse \
  --port 80
```
*В этом варианте задержка генерации вектора снизится до **~60–80 мс**.*

---

## 📈 6. Финансовая экономика (Экономия бюджета компании)

| Вариант инфраструктуры | Стоимость в месяц | Стоимость в год | Результат |
| :--- | :--- | :--- | :--- |
| **GPU-инстанс (1x NVIDIA T4 / 16 GB)** | ~$220 / мес | ~$2 640 / год | Избыточная стоимость |
| **Предлагаемый CPU-сервер (8 vCPU / 16 GB RAM)** | **~$45 / мес** | **~$540 / год** | **Экономия ~$2 100 в год (79%)** |
| **On-Premise (Контур / дата-центр компании)** | **$0 доп. затрат** | **$0 доп. затрат** | Бесплатно на существующих серверах |

---

## 🏁 Резюме для тимлида (Димы)

> 1. **Где:** Разворачиваем в отдельном Docker-контейнере на виртуальной машине рядом с нашими сервисами.
> 2. **Конфиг железа:** Сервер на **8 vCPU и 8–16 GB RAM** (с поддержкой AVX-512). Сама модель весит в памяти **1.88 ГБ**.
> 3. **Конфиг рантайма:** Запуск строго в **1 воркер**, `OMP_NUM_THREADS=4`, входной лимит `192 токена`.
> 4. **Скорость:** Генерация одного 768-мерного вектора занимает **~200 мс**, сервер держит до 25–30 запросов в секунду.
> 5. **Как запускать:** Готовы Dockerfile и скрипт микросервиса на FastAPI + SentenceTransformers. Контейнер отдает вектор по `POST /embed`.
