"""
api.py — FastAPI backend for the Deepfake Audio Detection System.

Endpoints
---------
  POST /detect          — upload an audio file, get real/fake/uncertain label
  POST /detect/batch    — upload multiple files
  GET  /health          — liveness probe
  GET  /model/info      — model metadata

Run
---
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  # or
  python api.py
"""

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Deepfake Audio Detector API",
    description=(
        "Detect AI-generated (fake) vs real human speech using a Wav2Vec2-based classifier. "
        "Upload audio files and receive a real/fake/uncertain label with confidence scores."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
#  Global detector (lazy-loaded on first request)
# ──────────────────────────────────────────────────────────

_detector = None
_model_dir: Optional[str] = os.getenv("MODEL_DIR", str(cfg.CHECKPOINTS / "best_model"))


def get_detector():
    global _detector
    if _detector is None:
        from inference import DeepfakeDetector
        log.info(f"Loading model from: {_model_dir}")
        _detector = DeepfakeDetector(
            model_dir=_model_dir,
            threshold=cfg.CONFIDENCE_THRESHOLD,
        )
    return _detector


# ──────────────────────────────────────────────────────────
#  Response schemas
# ──────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    chunk_index: int
    label:       str = Field(..., example="fake")
    confidence:  float = Field(..., example=0.94)
    real_prob:   float
    fake_prob:   float


class DetectionResult(BaseModel):
    filename:    str
    label:       str  = Field(..., example="fake")
    confidence:  float = Field(..., ge=0.0, le=1.0)
    real_prob:   float
    fake_prob:   float
    num_chunks:  int
    chunks:      List[ChunkResult]
    latency_ms:  float


class BatchResult(BaseModel):
    results:     List[DetectionResult]
    total_files: int
    latency_ms:  float


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    device:      str


class ModelInfo(BaseModel):
    model_name:  str
    model_dir:   str
    threshold:   float
    sample_rate: int
    chunk_sec:   float


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

MAX_BYTES = cfg.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


async def _save_upload_tmp(file: UploadFile) -> str:
    """Save uploaded file to a temp file and return the path."""
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {cfg.MAX_UPLOAD_MB} MB.",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


def _build_detection_result(filename: str, raw: dict, latency_ms: float) -> DetectionResult:
    chunks = [
        ChunkResult(
            chunk_index=i,
            label=c["label"],
            confidence=c["confidence"],
            real_prob=c["real_prob"],
            fake_prob=c["fake_prob"],
        )
        for i, c in enumerate(raw.get("chunks", []))
    ]
    return DetectionResult(
        filename=filename,
        label=raw["label"],
        confidence=raw["confidence"],
        real_prob=raw["real_prob"],
        fake_prob=raw["fake_prob"],
        num_chunks=len(chunks),
        chunks=chunks,
        latency_ms=round(latency_ms, 1),
    )


# ──────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness probe — returns 200 when the service is running."""
    global _detector
    loaded = _detector is not None
    device = str(_detector.device) if loaded else "not_loaded"
    return HealthResponse(status="ok", model_loaded=loaded, device=device)


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
def model_info():
    """Return metadata about the loaded model."""
    det = get_detector()
    return ModelInfo(
        model_name=cfg.MODEL_NAME,
        model_dir=_model_dir,
        threshold=det.threshold,
        sample_rate=cfg.SAMPLE_RATE,
        chunk_sec=det.chunk_sec,
    )


@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="Audio file to analyse"),
    threshold: float = Query(
        default=cfg.CONFIDENCE_THRESHOLD,
        ge=0.0, le=1.0,
        description="Confidence threshold; below this → 'uncertain'",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Analyse a single audio file for deepfake detection.

    Returns a JSON with:
    - `label`:      **real** | **fake** | **uncertain**
    - `confidence`: float (0–1)
    - `real_prob`, `fake_prob`: class probabilities
    - `chunks`:     per-chunk breakdown
    """
    tmp_path = await _save_upload_tmp(file)
    background_tasks.add_task(lambda p: Path(p).unlink(missing_ok=True), tmp_path)

    det = get_detector()
    det.threshold = threshold   # honour per-request threshold

    t0     = time.perf_counter()
    result = det.predict(tmp_path)
    ms     = (time.perf_counter() - t0) * 1000

    return _build_detection_result(file.filename or "audio", result, ms)


@app.post("/detect/batch", response_model=BatchResult, tags=["Detection"])
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files"),
    threshold: float = Query(default=cfg.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Analyse multiple audio files in one request.
    Returns a list of results in the same order as the uploaded files.
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 files per batch request.")

    det = get_detector()
    det.threshold = threshold

    results  = []
    t0_total = time.perf_counter()

    for f in files:
        tmp_path = await _save_upload_tmp(f)
        background_tasks.add_task(lambda p: Path(p).unlink(missing_ok=True), tmp_path)

        t0     = time.perf_counter()
        result = det.predict(tmp_path)
        ms     = (time.perf_counter() - t0) * 1000
        results.append(_build_detection_result(f.filename or "audio", result, ms))

    total_ms = (time.perf_counter() - t0_total) * 1000
    return BatchResult(results=results, total_files=len(files), latency_ms=round(total_ms, 1))


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=False,
        workers=1,           # single worker — model stays in memory
    )
