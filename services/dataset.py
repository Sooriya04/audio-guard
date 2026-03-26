"""
dataset.py — PyTorch Dataset + DataLoader factory for deepfake audio detection.

The dataset reads from either:
  (a) a metadata CSV with columns: path, label, label_id
  (b) two directories:  processed/real/*.wav  and  processed/fake/*.wav
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import Wav2Vec2FeatureExtractor

import config as cfg


# ──────────────────────────────────────────────────────────
#  Augmentation helpers
# ──────────────────────────────────────────────────────────

def _add_gaussian_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(waveform ** 2))
    rms_noise  = rms_signal / (10 ** (snr_db / 20.0))
    noise      = np.random.randn(len(waveform)).astype(np.float32) * rms_noise
    return np.clip(waveform + noise, -1.0, 1.0)


def _add_reverb(waveform: np.ndarray) -> np.ndarray:
    """Simple FIR reverb simulation via random IR convolution."""
    ir_len  = random.randint(800, 3200)            # 50–200 ms at 16kHz
    ir      = np.random.exponential(1.0, ir_len).astype(np.float32)
    ir     /= (ir.sum() + 1e-8)
    reverbed = np.convolve(waveform, ir, mode="full")[: len(waveform)]
    return np.clip(reverbed, -1.0, 1.0)


def _simulate_compression(waveform: np.ndarray) -> np.ndarray:
    """
    Simulate MP3/YouTube compression artefacts by quantising to lower
    bit depth (dithered), which introduces characteristic compression noise.
    """
    bits   = random.choice([8, 10, 12])
    levels = 2 ** bits
    quantised = np.round(waveform * levels) / levels
    return np.clip(quantised, -1.0, 1.0)


def _pitch_shift(waveform: np.ndarray) -> np.ndarray:
    semitones = random.uniform(*cfg.AUG_PITCH_RANGE)
    return librosa.effects.pitch_shift(
        waveform, sr=cfg.SAMPLE_RATE, n_steps=semitones
    ).astype(np.float32)


def augment_waveform(waveform: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a waveform during training."""
    if random.random() < cfg.AUG_NOISE_PROB:
        snr = random.uniform(10, 30)
        waveform = _add_gaussian_noise(waveform, snr)
    if random.random() < cfg.AUG_REVERB_PROB:
        waveform = _add_reverb(waveform)
    if random.random() < cfg.AUG_COMPRESS_PROB:
        waveform = _simulate_compression(waveform)
    if random.random() < cfg.AUG_PITCH_PROB:
        waveform = _pitch_shift(waveform)
    return waveform


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────

class DeepfakeAudioDataset(Dataset):
    """
    Parameters
    ----------
    records : list of dicts with keys ``path`` (str) and ``label_id`` (int)
    feature_extractor : Wav2Vec2FeatureExtractor
    augment : bool — apply augmentation (training only)
    max_length_sec : float — hard cap on clip duration
    """

    def __init__(
        self,
        records: List[Dict],
        feature_extractor: Wav2Vec2FeatureExtractor,
        augment: bool = False,
        max_length_sec: float = cfg.CHUNK_DURATION,
    ):
        self.records           = records
        self.feature_extractor = feature_extractor
        self.augment           = augment
        self.max_samples       = int(max_length_sec * cfg.SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec   = self.records[idx]
        path  = rec["path"]
        label = int(rec["label_id"])

        waveform = self._load(path)

        if self.augment and cfg.AUGMENT:
            waveform = augment_waveform(waveform)

        # Wav2Vec2FeatureExtractor → input_values + attention_mask
        encoding = self.feature_extractor(
            waveform,
            sampling_rate=cfg.SAMPLE_RATE,
            max_length=self.max_samples,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_values":  encoding.input_values.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels":        torch.tensor(label, dtype=torch.long),
        }

    def _load(self, path: str) -> np.ndarray:
        try:
            waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            waveform, sr = librosa.load(path, sr=None, mono=True)
            waveform = waveform.astype(np.float32)

        # Resample if needed
        if sr != cfg.SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)

        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Truncate / pad
        if len(waveform) > self.max_samples:
            # Random crop during training, centre crop during eval
            if self.augment:
                start    = random.randint(0, len(waveform) - self.max_samples)
                waveform = waveform[start : start + self.max_samples]
            else:
                mid   = len(waveform) // 2
                half  = self.max_samples // 2
                waveform = waveform[mid - half : mid - half + self.max_samples]
        elif len(waveform) < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        return waveform


# ──────────────────────────────────────────────────────────
#  Factory: build datasets + dataloaders from CSV
# ──────────────────────────────────────────────────────────

def load_records_from_csv(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"path", "label_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[["path", "label_id", "label"]].to_dict("records")


def load_records_from_dirs(
    real_dir: Path = cfg.REAL_PROC,
    fake_dir: Path = cfg.FAKE_PROC,
) -> List[Dict]:
    records = []
    for wav in sorted(real_dir.glob("*.wav")):
        records.append({"path": str(wav), "label": "real", "label_id": 0})
    for wav in sorted(fake_dir.glob("*.wav")):
        records.append({"path": str(wav), "label": "fake", "label_id": 1})
    return records


def split_records(
    records: List[Dict],
    val_split: float = cfg.VAL_SPLIT,
    test_split: float = cfg.TEST_SPLIT,
    seed: int = cfg.SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split

    labels = [r["label_id"] for r in records]

    train_val, test = train_test_split(
        records,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )
    tv_labels = [r["label_id"] for r in train_val]
    rel_val   = val_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        stratify=tv_labels,
        random_state=seed,
    )
    return train, val, test


def make_weighted_sampler(records: List[Dict]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class imbalance during training.
    Each sample's weight is inversely proportional to its class frequency.
    """
    from collections import Counter
    counts  = Counter(r["label_id"] for r in records)
    total   = len(records)
    weights = [total / counts[r["label_id"]] for r in records]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(
    feature_extractor: Wav2Vec2FeatureExtractor,
    csv_path: Optional[Path] = None,
    real_dir: Optional[Path] = None,
    fake_dir: Optional[Path] = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Priority: csv_path > (real_dir + fake_dir) > default config dirs.
    """
    if csv_path and csv_path.exists():
        records = load_records_from_csv(csv_path)
    else:
        r = real_dir or cfg.REAL_PROC
        f = fake_dir or cfg.FAKE_PROC
        records = load_records_from_dirs(r, f)

    if not records:
        raise RuntimeError("No records found — run preprocess.py first.")

    train_recs, val_recs, test_recs = split_records(records)

    print(f"\n  Dataset split — train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}\n")

    train_ds = DeepfakeAudioDataset(train_recs, feature_extractor, augment=True)
    val_ds   = DeepfakeAudioDataset(val_recs,   feature_extractor, augment=False)
    test_ds  = DeepfakeAudioDataset(test_recs,  feature_extractor, augment=False)

    sampler = make_weighted_sampler(train_recs) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
