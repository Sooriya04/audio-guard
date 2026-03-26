"""
inference.py — Inference engine for deepfake audio detection.

Returns:
  {
    "label":      "real" | "fake" | "uncertain",
    "confidence": float,          # max class probability
    "real_prob":  float,
    "fake_prob":  float,
    "chunks":     [...]           # per-chunk results (if audio was split)
  }

Usage
-----
  from inference import DeepfakeDetector
  detector = DeepfakeDetector("checkpoints/best_model")
  result   = detector.predict("audio.wav")

  # CLI
  python inference.py --model checkpoints/best_model --audio speech.wav
  python inference.py --model checkpoints/best_model --audio speech.wav --verbose
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor

import config as cfg
from model import load_model
from preprocess import (
    chunk_audio,
    normalize_amplitude,
    to_mono_16k,
    trim_silence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Detector class
# ──────────────────────────────────────────────────────────

class DeepfakeDetector:
    """
    High-level inference wrapper.

    Parameters
    ----------
    model_dir  : path to the HuggingFace checkpoint directory
    device     : 'cpu' | 'cuda' | None (auto-select)
    threshold  : confidence threshold below which label becomes 'uncertain'
    chunk_sec  : duration of each audio chunk in seconds
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        threshold: float = cfg.CONFIDENCE_THRESHOLD,
        chunk_sec: float = cfg.CHUNK_DURATION,
    ):
        self.threshold = threshold
        self.chunk_sec = chunk_sec

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        log.info(f"Loading model from '{model_dir}' on {self.device} …")
        self.model = load_model(str(model_dir), self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(model_dir),
            sampling_rate=cfg.SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        log.info("Model ready.")

    # ── public API ─────────────────────────────────────────

    def predict(self, audio_input: Union[str, Path, np.ndarray]) -> Dict:
        """
        Predict whether the audio is real or fake.

        Parameters
        ----------
        audio_input : file path  OR  raw numpy waveform (float32, 16kHz mono)

        Returns
        -------
        dict with keys: label, confidence, real_prob, fake_prob, chunks
        """
        waveform = self._load_and_preprocess(audio_input)

        chunks      = chunk_audio(waveform, chunk_sec=self.chunk_sec)
        if not chunks:
            return self._make_result(real_prob=0.5, fake_prob=0.5, chunk_results=[])

        chunk_results = [self._predict_chunk(c) for c in chunks]

        # Aggregate: average probabilities across chunks
        avg_real = float(np.mean([r["real_prob"] for r in chunk_results]))
        avg_fake = float(np.mean([r["fake_prob"] for r in chunk_results]))

        return self._make_result(avg_real, avg_fake, chunk_results)

    def predict_batch(self, paths: List[Union[str, Path]]) -> List[Dict]:
        """Run predict() on a list of audio files."""
        return [self.predict(p) for p in paths]

    # ── internals ──────────────────────────────────────────

    def _load_and_preprocess(self, audio_input) -> np.ndarray:
        if isinstance(audio_input, (str, Path)):
            try:
                waveform, sr = sf.read(str(audio_input), dtype="float32", always_2d=False)
            except Exception:
                waveform, sr = librosa.load(str(audio_input), sr=None, mono=True)
                waveform = waveform.astype(np.float32)
            waveform = to_mono_16k(waveform, sr)
        else:
            waveform = np.asarray(audio_input, dtype=np.float32)

        waveform = trim_silence(waveform)
        waveform = normalize_amplitude(waveform)
        return waveform

    @torch.no_grad()
    def _predict_chunk(self, chunk: np.ndarray) -> Dict:
        """Run the model on a single audio chunk."""
        max_len = int(self.chunk_sec * cfg.SAMPLE_RATE)
        encoding = self.feature_extractor(
            chunk,
            sampling_rate=cfg.SAMPLE_RATE,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values   = encoding.input_values.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        logits  = outputs.logits.float().cpu().numpy()[0]           # (2,)
        probs   = _softmax(logits)

        real_p = float(probs[0])
        fake_p = float(probs[1])
        conf   = float(probs.max())
        pred   = int(probs.argmax())
        label  = cfg.LABEL_MAP[pred]

        if conf < self.threshold:
            label = "uncertain"

        return {
            "label":      label,
            "confidence": round(conf, 4),
            "real_prob":  round(real_p, 4),
            "fake_prob":  round(fake_p, 4),
        }

    def _make_result(
        self,
        real_prob: float,
        fake_prob: float,
        chunk_results: List[Dict],
    ) -> Dict:
        conf      = max(real_prob, fake_prob)
        pred_idx  = 1 if fake_prob > real_prob else 0
        label     = cfg.LABEL_MAP[pred_idx] if conf >= self.threshold else "uncertain"

        return {
            "label":      label,
            "confidence": round(conf, 4),
            "real_prob":  round(real_prob, 4),
            "fake_prob":  round(fake_prob, 4),
            "chunks":     chunk_results,
        }


# ──────────────────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

def _print_result(result: Dict, verbose: bool = False):
    label = result["label"]
    conf  = result["confidence"]
    emoji = {"real": "✅", "fake": "🚨", "uncertain": "❓"}.get(label, "?")

    print(f"\n  {emoji}  Label      : {label.upper()}")
    print(f"     Confidence : {conf:.2%}")
    print(f"     Real prob  : {result['real_prob']:.4f}")
    print(f"     Fake prob  : {result['fake_prob']:.4f}")

    if verbose and result.get("chunks"):
        print(f"\n  Per-chunk breakdown ({len(result['chunks'])} chunk(s)):")
        for i, c in enumerate(result["chunks"], 1):
            print(
                f"    Chunk {i:02d}: {c['label']:<10}  "
                f"conf={c['confidence']:.2%}  "
                f"fake_p={c['fake_prob']:.4f}"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake audio inference.")
    parser.add_argument("--model",   type=Path, required=True, help="Model checkpoint directory")
    parser.add_argument("--audio",   type=Path, required=True, help="Audio file to analyse")
    parser.add_argument("--threshold", type=float, default=cfg.CONFIDENCE_THRESHOLD)
    parser.add_argument("--device",  type=str,  default=None,  help="cpu | cuda")
    parser.add_argument("--verbose", action="store_true",      help="Show per-chunk breakdown")
    parser.add_argument("--json",    action="store_true",      help="Output raw JSON")
    args = parser.parse_args()

    detector = DeepfakeDetector(
        model_dir=args.model,
        device=args.device,
        threshold=args.threshold,
    )

    result = detector.predict(args.audio)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n── Deepfake Audio Analysis ────────────────")
        print(f"  File: {args.audio.name}")
        _print_result(result, verbose=args.verbose)
