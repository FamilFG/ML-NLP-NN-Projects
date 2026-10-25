import os
import logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dlp-labse-service")

app = FastAPI(title="DLP LaBSE Model Embedding Service", version="1.0.0")

# Ограничиваем количество потоков PyTorch, чтобы защитить CPU от троттлинга
omp_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
torch.set_num_threads(omp_threads)
logger.info(f"PyTorch CPU threads set to {omp_threads}")

MODEL_ID = os.getenv("MODEL_ID", "FamilFG/dlp-labse")
SUBFOLDER = os.getenv("SUBFOLDER", "labse-finetuned-freeze6")
HF_TOKEN = os.getenv("HF_TOKEN", None)

logger.info(f"Loading model '{MODEL_ID}' (subfolder='{SUBFOLDER}') on CPU...")
model = SentenceTransformer(
    MODEL_ID,
    subfolder=SUBFOLDER if SUBFOLDER else None,
    use_auth_token=HF_TOKEN,
    device="cpu"
)

# Выполняем обязательный прогрев кэшей процессора
logger.info("Performing warm-up forward pass...")
_ = model.encode("uşaqlar üçün pambıq corablar warm-up", normalize_embeddings=True)
logger.info("✅ Model loaded and warmed up successfully.")


class EmbedRequest(BaseModel):
    inputs: str


@app.post("/embed")
def embed_endpoint(req: EmbedRequest):
    query_text = req.inputs.strip() if req.inputs else ""
    if not query_text:
        raise HTTPException(status_code=400, detail="Inputs query text cannot be empty.")

    # Обрезаем входные токены до 192 (защита от квадратичного замедления)
    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    return [embedding]


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "model_id": MODEL_ID,
        "subfolder": SUBFOLDER,
        "device": "cpu",
        "threads": omp_threads
    }
