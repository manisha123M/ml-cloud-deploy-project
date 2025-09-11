from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class Request(BaseModel):
    text: str

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, target_names
    obj = joblib.load("model.joblib")
    model = obj["model"]
    target_names = obj["target_names"]

@app.post("/predict")
def predict(req: Request):
    pred = model.predict([req.text])[0]
    proba = model.predict_proba([req.text]).max()
    return {"label": target_names[pred], "confidence": float(proba)}
