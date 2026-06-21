from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ==================================================
# FastAPI App
# ==================================================

app = FastAPI(
    title="Fake News Detection API",
    description="BERT-powered Fake News Classifier",
    version="1.0.0"
)

# ==================================================
# Load Saved Model
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "notebooks" / "saved_model"

print(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.eval()

print("Model loaded successfully!")

# ==================================================
# Request Schema
# ==================================================

class PredictRequest(BaseModel):
    text: str

# ==================================================
# Health Check Endpoint
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# ==================================================
# Prediction Endpoint
# ==================================================

@app.post("/predict")
def predict(request: PredictRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    fake_prob = probabilities[0].item()
    real_prob = probabilities[1].item()

    label = "REAL" if real_prob > fake_prob else "FAKE"

    confidence = round(
        max(fake_prob, real_prob),
        4
    )

    return {
        "label": label,
        "confidence": confidence
    }

# ==================================================
# Local Run
# ==================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )