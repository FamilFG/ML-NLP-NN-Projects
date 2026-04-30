import joblib
import os

MODEL_PATH = "models/role_model.pkl"
VECTORIZER_PATH = "models/role_vectorizer.pkl"

def role_model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

def predict_role(text):
    if not role_model_exists():
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = max(model.predict_proba(vec)[0])
    
    text_lower = text.lower()
    if pred == "JavaDeveloper" and any(keyword in text_lower for keyword in ["java developer", "java backend", "java engineer"]):
        prob = min(prob + 0.20, 1.0)
    
    return pred, float(prob)    