from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.feature_engineering import extract_skills, estimate_experience_years, cv_quality_score

def experience_score(years):
    if years >= 5:
        return 1.0
    if years >= 3:
        return 0.8
    if years >= 2:
        return 0.6
    if years >= 1:
        return 0.4
    return 0.2

def skill_match_score(cv_skills, jd_skills):
    if not jd_skills:
        return 0.0

    matched = set(cv_skills) & set(jd_skills)
    return len(matched) / len(jd_skills)

def rank_candidates(cv_texts, filenames, jd_text, skills_list):
    jd_skills = extract_skills(jd_text, skills_list)

    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = [jd_text] + cv_texts
    vectors = vectorizer.fit_transform(all_texts)

    jd_vec = vectors[0]
    cv_vecs = vectors[1:]

    similarities = cosine_similarity(cv_vecs, jd_vec)

    results = []

    for i in range(len(cv_texts)):
        cv_text = cv_texts[i]
        filename = filenames[i]

        cv_skills = extract_skills(cv_text, skills_list)

        skill_match = skill_match_score(cv_skills, jd_skills)
        years = estimate_experience_years(cv_text)
        exp_score = experience_score(years)

        relevance = float(similarities[i][0])
        quality = cv_quality_score(cv_text) / 100.0

        final = (
            0.50 * skill_match +
            0.20 * exp_score +
            0.20 * relevance +
            0.10 * quality
        )

        missing = sorted(list(set(jd_skills) - set(cv_skills)))
        matched = sorted(list(set(jd_skills) & set(cv_skills)))

        results.append({
            "filename": filename,
            "final_score": round(final * 100, 2),
            "skill_match_score": round(skill_match * 100, 2),
            "relevance_score": round(relevance * 100, 2),
            "experience_years": years,
            "experience_score": round(exp_score * 100, 2),
            "cv_quality_score": round(quality * 100, 2),
            "matched_skills": matched,
            "missing_skills": missing
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results