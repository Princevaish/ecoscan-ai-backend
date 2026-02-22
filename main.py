"""
🧹 Garbage vs Clean Classifier — FastAPI Backend
Production-ready API for image classification using TensorFlow CNN model.
"""

import io
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── TensorFlow import (lazy, to keep startup fast in some envs) ──────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    raise RuntimeError("TensorFlow is not installed. Run: pip install tensorflow")


# ══════════════════════════════════════════════════════════════════════════════
# ── Lifespan: load model once at startup, release at shutdown ─────────────────
# ══════════════════════════════════════════════════════════════════════════════
ml_model: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model when the server starts, clean up on shutdown."""
    print("🚀 Loading garbage classifier model...")
    try:
        ml_model["classifier"] = load_model("garbage_classifier_model.h5")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model could not be loaded: {e}")
    yield
    # ── Cleanup ──────────────────────────────────────────────────────────────
    ml_model.clear()
    print("🛑 Model unloaded. Server shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
# ── App Initialization ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Garbage vs Clean Classifier API",
    description="Upload images and classify them as GARBAGE or CLEAN.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS: allow the frontend (any origin during dev; restrict in prod) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ⚠️ Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ── Image Preprocessing ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a model-ready NumPy array.

    Steps:
        1. Open image with Pillow.
        2. Convert to RGB (drops alpha channel / handles grayscale).
        3. Resize to 128 × 128.
        4. Normalise pixel values to [0, 1].
        5. Add batch dimension → shape (1, 128, 128, 3).
    """
    image = Image.open(io.BytesIO(image_bytes))

    # ── Remove alpha channel or convert grayscale → RGB ──────────────────────
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ── Resize ───────────────────────────────────────────────────────────────
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    # ── Normalise ─────────────────────────────────────────────────────────────
    array = np.array(image, dtype=np.float32) / 255.0  # shape: (128, 128, 3)

    # ── Batch dimension ───────────────────────────────────────────────────────
    array = np.expand_dims(array, axis=0)               # shape: (1, 128, 128, 3)
    return array


# ══════════════════════════════════════════════════════════════════════════════
# ── Prediction Logic ──────────────────────────────────════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction(image_array: np.ndarray) -> tuple[str, float]:
    """
    Run model inference and decode sigmoid output.

    The model outputs a single sigmoid value in [0, 1]:
        < 0.5  → GARBAGE  (confidence = (1 - prob) × 100)
        ≥ 0.5  → CLEAN    (confidence = prob × 100)

    Returns:
        (label, confidence_percentage)
    """
    preds = ml_model["classifier"].predict(image_array, verbose=0)
    prob: float = float(preds[0][0])

    if prob < 0.5:
        label = "GARBAGE"
        confidence = (1.0 - prob) * 100.0
    else:
        label = "CLEAN"
        confidence = prob * 100.0

    return label, round(confidence, 2)


# ══════════════════════════════════════════════════════════════════════════════
# ── Routes ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Garbage Classifier API is running 🚀"}


@app.post("/predict/", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image as GARBAGE or CLEAN.

    - **file**: JPG / JPEG / PNG image file.

    Returns JSON:
    ```json
    {
        "filename": "photo.jpg",
        "prediction": "GARBAGE",
        "confidence": 87.45
    }
    ```
    """
    # ── Validate content type ─────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: JPEG, PNG, WEBP.",
        )

    # ── Read raw bytes ────────────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # ── Preprocess ───────────────────────────────────────────────────────────
    try:
        image_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Image preprocessing failed: {e}"
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        prediction, confidence = run_prediction(image_array)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {e}"
        )

    return JSONResponse(
        content={
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Dev Entry Point ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
