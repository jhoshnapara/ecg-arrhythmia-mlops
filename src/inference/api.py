"""FastAPI inference service for ECG arrhythmia classifier."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.cnn import ECGCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusion)", "Q (Unclassified)"]
WINDOW_SIZE = 360
MODEL_PATH = "models/best.pt"

state = {"model": None, "version": "v1-tuned"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading model from {MODEL_PATH}...")
    model = ECGCNN(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    state["model"] = model
    log.info(f"✅ Loaded model {state['version']}")
    yield
    log.info("Shutting down")


app = FastAPI(title="ECG Arrhythmia Classifier", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., description="ECG signal samples (length 360)")

    def to_tensor(self) -> torch.Tensor:
        arr = np.array(self.signal, dtype=np.float32)
        if arr.shape != (WINDOW_SIZE,):
            raise ValueError(f"Expected length {WINDOW_SIZE}, got {arr.shape}")
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return torch.from_numpy(arr).unsqueeze(0)


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_idx: int
    confidence: float
    class_probabilities: dict[str, float]
    inference_ms: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info():
    return {"model_name": "ecg-arrhythmia-cnn", "version": state["version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded")
    try:
        x = req.to_tensor()
    except ValueError as e:
        raise HTTPException(400, str(e))

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = state["model"](x)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000

    idx = int(np.argmax(probs))
    log.info(f"prediction idx={idx} confidence={float(probs[idx]):.3f} latency_ms={latency_ms:.2f}")

    return PredictResponse(
        predicted_class=CLASS_NAMES[idx],
        predicted_idx=idx,
        confidence=float(probs[idx]),
        class_probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        inference_ms=latency_ms,
        model_version=state["version"],
    )
