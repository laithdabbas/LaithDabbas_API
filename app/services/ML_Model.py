import os
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def _runtime_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODEL_PATH = os.path.join(_runtime_base_dir(), "spam_model.pkl")

def _train_model():
    data = {
        "text": [
            "Win money now",
            "Claim your free prize",
            "Limited time offer",
            "Hello how are you",
            "Let's meet tomorrow",
            "Project meeting at 10"
        ],
        "label": [
            "spam",
            "spam",
            "spam",
            "ham",
            "ham",
            "ham"
        ]
    }
    df = pd.DataFrame(data)
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])
    pipeline.fit(df["text"], df["label"])
    try:
        joblib.dump(pipeline, MODEL_PATH)
    except Exception:
        # If the app folder is not writable, keep the in-memory model.
        pass
    return pipeline

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = _train_model()

def predict_text(text):
    prediction = model.predict([text])[0]
    return prediction
