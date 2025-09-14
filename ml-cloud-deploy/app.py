from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles

class Request(BaseModel):
    text: str

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, vectorizer
    obj = joblib.load("model.joblib")
    model = obj["model"]
    vectorizer = obj["vectorizer"]

@app.post("/predict")
def predict(req: Request):
    X = vectorizer.transform([req.text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return {"label": pred, "confidence": float(proba)}

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
