import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pdf_reader import extract_text_from_pdf
from preprocess import clean_text

DATASET_PATH = "data"

def train_role_classifier():
    X = []
    y = []

    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if file.endswith(".pdf"):
                path = os.path.join(folder, file)

                text = extract_text_from_pdf(path)
                text = clean_text(text)

                if len(text) < 50:
                    continue

                X.append(text)
                y.append(label)
                print("Processing:", path)

    print("Total samples:", len(X))

    if len(X) < 10:
        print("Not enough resumes to train model")
        return
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_vectorized = vectorizer.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200, 
        criterion='gini',
        bootstrap=True,
        n_jobs=-1, 
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_vectorized, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/role_model.pkl")
    joblib.dump(vectorizer, "models/role_vectorizer.pkl")
    print("Role classifier trained and saved successfully")

if __name__ == "__main__":
    train_role_classifier()