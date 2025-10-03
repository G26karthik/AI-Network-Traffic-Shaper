from __future__ import annotations

"""
FastAPI service exposing POST /predict_traffic.

Accepts JSON body with features: protocol, length, src_port, dst_port.
Loads either a PyTorch model (deep_model.pt) or a scikit-learn pipeline (traffic_model.pkl) as fallback.
"""

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import joblib
except Exception:
    joblib = None  # type: ignore

try:
    from deep.infer import TorchPredictor
except Exception:
    TorchPredictor = None  # type: ignore


class TrafficFeatures(BaseModel):
    protocol: str = Field(..., description="Highest protocol layer string, e.g., TCP, UDP, HTTP")
    length: int = Field(..., ge=0)
    src_port: int = Field(..., ge=0)
    # FIXED: Removed dst_port to prevent label leakage


MODEL_PATH = os.environ.get("TRAFFIC_TORCH_MODEL", "deep_model.pt")
SK_MODEL_PATH = os.environ.get("TRAFFIC_SK_MODEL", "traffic_model.pkl")

app = FastAPI(title="AI Traffic Classifier API")


class PredictorWrapper:
    def __init__(self) -> None:
        self.torch: Optional[object] = None
        self.sklearn: Optional[object] = None
        if TorchPredictor and os.path.exists(MODEL_PATH):
            try:
                self.torch = TorchPredictor(MODEL_PATH)
                print(f"[i] Loaded Torch model: {MODEL_PATH}")
            except Exception as e:
                print("[!] Torch model load failed:", e)
        if joblib and os.path.exists(SK_MODEL_PATH):
            try:
                self.sklearn = joblib.load(SK_MODEL_PATH)
                print(f"[i] Loaded sklearn pipeline: {SK_MODEL_PATH}")
            except Exception as e:
                print("[!] sklearn pipeline load failed:", e)
        if not self.torch and not self.sklearn:
            raise RuntimeError("No model available. Provide deep_model.pt or traffic_model.pkl")

    def predict(self, feats: dict) -> str:
        if self.torch:
            return self.torch.predict_label(feats)  # type: ignore[attr-defined]
        # Fallback to sklearn
        X = pd.DataFrame([feats])
        return str(self.sklearn.predict(X)[0])  # type: ignore[call-arg]


_predictor: Optional[PredictorWrapper] = None


@app.on_event("startup")
def _startup() -> None:
    global _predictor
    _predictor = PredictorWrapper()


@app.post("/predict_traffic")
def predict_traffic(body: TrafficFeatures):
    global _predictor
    if _predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    feats = body.model_dump()
    try:
        label = _predictor.predict(feats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"label": label}
