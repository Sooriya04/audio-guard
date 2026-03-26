"""
config.py — Central configuration for the Deepfake Audio Detection System.
All hyperparameters, paths, and settings live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REAL_RAW      = RAW_DIR  / "real"
FAKE_RAW      = RAW_DIR  / "fake"
REAL_PROC     = PROCESSED_DIR / "real"
FAKE_PROC     = PROCESSED_DIR / "fake"
CHECKPOINTS   = BASE_DIR / "checkpoints"
LOGS_DIR      = BASE_DIR / "logs"
RESULTS_DIR   = BASE_DIR / "results"

for _p in [REAL_RAW, FAKE_RAW, REAL_PROC, FAKE_PROC, CHECKPOINTS, LOGS_DIR, RESULTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  Audio
# ─────────────────────────────────────────────
SAMPLE_RATE      = 16_000          # Hz — Wav2Vec2 expects 16kHz
MAX_DURATION_SEC = 10.0            # clip longer audio
MIN_DURATION_SEC = 1.0             # discard shorter clips
CHUNK_DURATION   = 4.0             # seconds per chunk when splitting
TARGET_DB        = -20.0           # amplitude normalisation target (dBFS)
TOP_DB           = 30              # librosa silence-trim threshold


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
MODEL_NAME        = "facebook/wav2vec2-base"   # swap to "facebook/wav2vec2-large" for higher accuracy
NUM_LABELS        = 2                           # real=0, fake=1
HIDDEN_DROPOUT    = 0.1
CLASSIFIER_DROPOUT = 0.3
FREEZE_FEATURE_EXTRACTOR = True    # freeze CNN feature extractor; fine-tune transformer


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS     = 6
WARMUP_RATIO   = 0.1               # fraction of total steps for LR warmup
GRAD_CLIP      = 1.0               # gradient clipping max-norm
VAL_SPLIT      = 0.15
TEST_SPLIT     = 0.10
SEED           = 42
FP16           = True              # mixed-precision (set False on CPU)
SAVE_STEPS     = 500
EVAL_STEPS     = 500
LOGGING_STEPS  = 50
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST = "f1"


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80        # below this → "uncertain"
LABEL_MAP = {0: "real", 1: "fake"}
INV_LABEL_MAP = {"real": 0, "fake": 1}


# ─────────────────────────────────────────────
#  Augmentation (optional, training only)
# ─────────────────────────────────────────────
AUGMENT = True
AUG_NOISE_PROB    = 0.3
AUG_REVERB_PROB   = 0.2
AUG_COMPRESS_PROB = 0.3            # simulate YouTube/MP3 compression
AUG_PITCH_PROB    = 0.2
AUG_PITCH_RANGE   = (-2, 2)       # semitones


# ─────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_MB = 50
