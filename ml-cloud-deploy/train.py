from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
categories = ['rec.autos', 'rec.sport.hockey']
data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump({
    "model": pipeline,
    "target_names": data.target_names
}, "model.joblib")

print("Model saved as model.joblib")
