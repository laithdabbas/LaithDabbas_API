import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# simple dataset
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

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(df["text"], df["label"])

joblib.dump(model, "spam_model.pkl")

print("Model saved!")


import joblib

model = joblib.load("spam_model.pkl")

def predict_text(text):
    prediction = model.predict([text])[0]
    return prediction
