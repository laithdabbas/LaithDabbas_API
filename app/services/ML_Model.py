import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "spam_model.pkl")
SPAM_KEYWORDS = {
    "win", "winner", "prize", "free", "offer", "urgent", "claim", "bonus",
    "cash", "money", "lottery", "click", "limited", "congratulations",
    "won", "gift", "reward", "promo", "investment", "discount", "selected",
    "exclusive", "guaranteed",
}
SPAM_PHRASES = (
    "act now",
    "call now",
    "click here",
    "free gift",
    "free trial",
    "limited time",
    "claim now",
    "you won",
    "gift card",
)

def _train_model():
    data = {
        "text": [
            "Win money now",
            "Claim your free prize",
            "Limited time offer",
            "Congratulations you won a cash reward",
            "Click this link to claim your bonus",
            "Urgent action required to collect prize",
            "You have been selected as a winner",
            "Get free gift card now",
            "Hello how are you",
            "Let's meet tomorrow",
            "Project meeting at 10",
            "Can we reschedule the team call",
            "Please review the attached report",
            "Lunch at 1 pm with the client",
            "Thanks for your message",
            "See you in class tomorrow",
        ],
        "label": [
            "spam",
            "spam",
            "spam",
            "spam",
            "spam",
            "spam",
            "spam",
            "spam",
            "ham",
            "ham",
            "ham",
            "ham",
            "ham",
            "ham",
            "ham",
            "ham",
        ]
    }
    df = pd.DataFrame(data)
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])
    pipeline.fit(df["text"], df["label"])
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = _train_model()


def _model_is_valid(loaded_model) -> bool:
    try:
        spam_check = loaded_model.predict(["Win money now"])[0]
        ham_check = loaded_model.predict(["Project meeting at 10"])[0]
        return spam_check == "spam" and ham_check == "ham"
    except Exception:
        return False


if not _model_is_valid(model):
    model = _train_model()


def predict_text(text):
    normalized = (text or "").strip()
    if not normalized:
        return "ham"

    prediction = model.predict([normalized])[0]

    # Keyword safety-net: reduce obvious spam false negatives on tiny training sets.
    lowered = normalized.lower()
    keyword_hit = any(
        re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in SPAM_KEYWORDS
    )
    phrase_hit = any(phrase in lowered for phrase in SPAM_PHRASES)
    if prediction == "ham" and (keyword_hit or phrase_hit):
        return "spam"

    return prediction
