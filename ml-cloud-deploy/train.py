import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("data.csv")

X, y = df["text"], df["label"]

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model + vectorizer
joblib.dump({"model": model, "vectorizer": vectorizer}, "model.joblib")
print("✅ Model trained and saved as model.joblib")
