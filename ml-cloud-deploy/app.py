from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles
import os

class RequestModel(BaseModel):
    text: str

# Create the app
app = FastAPI()

# Mount static files at /static (not /)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index.html"))

# Load ML model on startup
@app.on_event("startup")
def load_model():
    global model, target_names
    obj = joblib.load("model.joblib")
    model = obj["model"]
    target_names = obj["target_names"]

# Prediction endpoint
@app.post("/predict")
def predict(req: RequestModel):
    pred = model.predict([req.text])[0]
    proba = model.predict_proba([req.text]).max()
    return {"label": target_names[pred], "confidence": float(proba)}
