from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import shutil

from app.pdf_reader import extract_text_from_pdf
from app.preprocess import clean_text
from app.skills_db import load_skills
from app.ranker import rank_candidates
from app.predict_role import predict_role

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

skills_list = load_skills("data/skills_list.txt")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rank")
async def rank_resumes(
    job_description: str = Form(...),
    file: Optional[UploadFile] = File(None),
    files: Optional[UploadFile] = File(None),
):
    selected_file = file or files
    if selected_file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_path = os.path.join(UPLOAD_DIR, selected_file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(selected_file.file, buffer)

    cv_text = extract_text_from_pdf(file_path)
    cv_text = clean_text(cv_text)

    ranked_candidates = rank_candidates(
        [cv_text],
        [selected_file.filename],
        job_description,
        skills_list,
    )

    rank_data = ranked_candidates[0] if ranked_candidates else {"filename": selected_file.filename}

    role, confidence = predict_role(cv_text)

    rank_data["predicted_role"] = role
    rank_data["role_confidence"] = confidence

    return {"ranked_candidates": [rank_data]}