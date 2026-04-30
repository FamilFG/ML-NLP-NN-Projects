# Job CV Ranker (ATS-Style Resume Ranking System)

This project is a mini ATS (Applicant Tracking System) that ranks candidate resumes against a given Job Description (JD) using a hybrid AI approach.

It supports:
- PDF resume parsing
- OCR for scanned/image-based PDFs
- Resume text preprocessing
- Skill extraction using a skills database
- Experience estimation
- TF-IDF relevance scoring (resume vs JD)
- Resume quality scoring
- ML-based role classification (trained from your dataset)
- FastAPI backend with Swagger UI

---

## What is JD Text?

**JD = Job Description**

It is the vacancy description text, for example:

> "We are looking for a Python Developer with FastAPI, Docker, PostgreSQL, and Linux experience."

You provide JD text and upload multiple resumes, and the system ranks candidates by match score.

---

## How It Works (Pipeline)

For each uploaded resume PDF:

1. Extract text from PDF
2. If PDF has no selectable text → OCR is applied
3. Clean and normalize extracted text
4. Extract skills from the resume using `skills_list.txt`
5. Extract skills from the Job Description
6. Compute:
   - Skill match score
   - Experience score (estimated years)
   - Relevance score (TF-IDF similarity between CV and JD)
   - Resume quality score (structure + action verbs + richness)
7. Predict candidate role using trained ML model (optional)
8. Combine all scores into a final score
9. Rank candidates from highest to lowest score

---

## Output Example

```json
{
  "ranked_candidates": [
    {
      "filename": "resume_1.pdf",
      "final_score": 91.23,
      "skill_match_score": 88.0,
      "relevance_score": 85.5,
      "experience_years": 3,
      "experience_score": 80.0,
      "cv_quality_score": 75.0,
      "matched_skills": ["python", "docker", "fastapi"],
      "missing_skills": ["kubernetes"],
      "predicted_role": "Python Developer",
      "role_confidence": 0.92
    }
  ]
}


Project Structure
job-cv-ranker/
│
├── app/
│   ├── main.py                 # FastAPI backend (API endpoints)
│   ├── pdf_reader.py           # PDF text extraction + OCR fallback
│   ├── preprocess.py           # text cleaning & normalization
│   ├── skills_db.py            # loads skills list from file
│   ├── feature_engineering.py  # extracts skills, experience, quality score
│   ├── ranker.py               # ranking logic and scoring formula
│   ├── train_role_model.py     # trains ML classifier from dataset folders
│   ├── predict_role.py         # predicts role from trained model
│
├── data/
│   ├── resumes/                # training dataset (PDFs grouped by label)
│   │   ├── Banking/
│   │   ├── Data Science/
│   │   ├── Designer/
│   │   ├── DevOps Engineer/
│   │   ├── Python Developer/
│   │
│   ├── skills_list.txt         # database of technical skills
│
├── models/
│   ├── role_model.pkl          # trained ML model output
│   ├── role_vectorizer.pkl     # TF-IDF vectorizer output
│
├── uploads/                    # uploaded resumes for ranking
├── requirements.txt
└── README.md