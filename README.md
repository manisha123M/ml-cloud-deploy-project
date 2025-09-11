# 🚀 ML Cloud Deploy Demo

**Mini Project for Cloud Computing (7th Semester, 4th Year Engineering)**

This project demonstrates how to **train a Machine Learning model** and **deploy it on the cloud** with a **modern web interface**, using **FastAPI**, **Docker**, and a stylish **HTML/CSS frontend**. Users can type text and get live predictions from the ML model directly on the page.

---

## ✨ Features
- **Train a text classification ML model** using **scikit-learn**  
- **Serve predictions via a REST API** using **FastAPI** (`/predict`)  
- **Containerize the app** using **Docker**  
- **Modern UI/UX**: interactive web page with live predictions  
- **Swagger API docs** available at `/docs`  

---

## 🛠 Tech Stack
- **Python 3.11**  
- **FastAPI** (REST API framework)  
- **scikit-learn** (ML pipeline)  
- **joblib** (model serialization)  
- **Docker** (containerization)  
- **HTML + CSS** (modern UI frontend)  

---

## ⚡ Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/ml-cloud-deploy.git
cd ml-cloud-deploy

2️⃣ Install dependencies
python -m venv venv
source venv/bin/activate       # Linux/Mac
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt

3️⃣ Train the ML model
python train.py


This generates model.joblib

4️⃣ Run the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8080

5️⃣ Access the project

Web UI → Open your browser to http://127.0.0.1:8080 (or Codespaces forwarded URL)

Swagger Docs → http://127.0.0.1:8080/docs

6️⃣ Test API with curl
curl -X POST "http://127.0.0.1:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"I love hockey"}'


Expected output:

{"label":"rec.sport.hockey","confidence":0.94}

📊 Project Flow
User Input → HTML/CSS Web UI → FastAPI /predict → ML Model → Prediction Displayed

🎨 UI/UX Features

Gradient background with glassmorphism container

Rounded input textarea & buttons with hover animation

Live prediction displayed directly below input

Optional button to open Swagger /docs

ne Learning Model on Cloud using FastAPI, Docker, and AWS Elastic Beanstalk.
