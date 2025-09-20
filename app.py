from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Load models ----------------
model = joblib.load("models/mello_v1.pkl")
scaler = joblib.load("models/structured_scaler.pkl")
# Load Hugging Face model directly instead of pickled transformer
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- FastAPI setup ----------------
app = FastAPI(title="Mental Illness Detection API")

# ---------------- Request model ----------------
class PredictionRequest(BaseModel):
    text: str
    structured: conlist(float, min_length=25, max_length=25)  # exactly 25 numbers

# ---------------- Health check ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Predict endpoint ----------------
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # Encode text
        X_embed = text_encoder.encode([req.text], show_progress_bar=False)
        # Scale structured features
        X_struct_scaled = scaler.transform([req.structured])
        # Combine features
        X_combined = np.hstack([X_embed, X_struct_scaled])

        # Predict probability
        prob = model.predict_proba(X_combined)[0][1]
        pred = int(prob >= 0.6)

        return {
            "probability": float(prob),
            "prediction": "Has Mental Illness" if pred else "No Illness"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
