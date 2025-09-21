from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Load models ----------------
model = joblib.load("models/mellov2.pkl")
scaler = joblib.load("models/structured_scaler.pkl")
vectorizer = joblib.load("models/text_vectorizer.pkl")  # TF-IDF vectorizer

# ---------------- FastAPI setup ----------------
app = FastAPI(title="Mental Illness Detection API (Lightweight)")

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
        # ---------------- Vectorize text ----------------
        X_text = vectorizer.transform([req.text]).toarray()

        # ---------------- Scale structured features ----------------
        X_struct_scaled = scaler.transform([req.structured])

        # ---------------- Combine features ----------------
        X_combined = np.hstack([X_text, X_struct_scaled])

        # ---------------- Predict ----------------
        prob = model.predict_proba(X_combined)[0][1]
        pred = int(prob >= 0.6)

        return {
            "probability": float(prob),
            "prediction": "Has Mental Illness" if pred else "No Illness"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
