"""
preprocess.py — Audio preprocessing pipeline for deepfake detection.

Usage
-----
# Preprocess a folder of real audio:
python preprocess.py --src data/raw/real --dst data/processed/real --label real

# Preprocess a folder of fake audio:
python preprocess.py --src data/raw/fake --dst data/processed/fake --label fake

# Dry-run (shows stats without writing files):
python preprocess.py --src data/raw/real --dst data/processed/real --label real --dry-run
"""

import argparse
import logging
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ──────────────────────────────────────────────────────────
#  Core helpers
# ──────────────────────────────────────────────────────────

def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load any audio file → (waveform float32, sample_rate)."""
    try:
        waveform, sr = librosa.load(str(path), sr=None, mono=True)
        return waveform.astype(np.float32), sr
    except Exception as exc:
        raise RuntimeError(f"Cannot load {path}: {exc}") from exc


def to_mono_16k(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz (waveform is already mono from librosa.load)."""
    if sr != cfg.SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
    return waveform


def trim_silence(waveform: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence using librosa."""
    trimmed, _ = librosa.effects.trim(waveform, top_db=cfg.TOP_DB)
    return trimmed


def normalize_amplitude(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalise to ±1, then RMS-normalise to TARGET_DB dBFS."""
    # Peak normalise first to avoid clipping
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    # RMS normalise
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        target_linear = 10 ** (cfg.TARGET_DB / 20.0)
        gain = target_linear / rms
        waveform = waveform * gain
        # Hard-clip to [-1, 1] as safety net
        waveform = np.clip(waveform, -1.0, 1.0)

    return waveform


def chunk_audio(
    waveform: np.ndarray,
    chunk_sec: float = cfg.CHUNK_DURATION,
    min_sec: float = cfg.MIN_DURATION_SEC,
) -> List[np.ndarray]:
    """
    Split a waveform into chunks of `chunk_sec` seconds.
    The last chunk is only kept if it is ≥ min_sec.
    """
    sr = cfg.SAMPLE_RATE
    chunk_len = int(chunk_sec * sr)
    min_len   = int(min_sec  * sr)
    total     = len(waveform)

    if total < min_len:
        return []          # too short — discard entirely

    if total <= chunk_len:
        return [waveform]  # fits in a single chunk

    chunks = []
    for start in range(0, total, chunk_len):
        segment = waveform[start : start + chunk_len]
        if len(segment) >= min_len:
            # Zero-pad last segment to full chunk length for uniformity
            if len(segment) < chunk_len:
                segment = np.pad(segment, (0, chunk_len - len(segment)))
            chunks.append(segment)

    return chunks


def process_single_file(
    path: Path,
    dst_dir: Path,
    stem_prefix: str = "",
    dry_run: bool = False,
) -> int:
    """
    Full pipeline for one audio file.
    Returns the number of chunks written (0 on failure / too short).
    """
    try:
        waveform, sr = load_audio(path)
    except RuntimeError as exc:
        log.warning(str(exc))
        return 0

    waveform = to_mono_16k(waveform, sr)
    waveform = trim_silence(waveform)
    waveform = normalize_amplitude(waveform)

    # Hard limit on total duration (discard if beyond MAX_DURATION * 2)
    max_samples = int(cfg.MAX_DURATION_SEC * 2 * cfg.SAMPLE_RATE)
    waveform = waveform[:max_samples]

    chunks = chunk_audio(waveform)
    if not chunks:
        log.debug(f"Skipped (too short): {path.name}")
        return 0

    if dry_run:
        return len(chunks)

    dst_dir.mkdir(parents=True, exist_ok=True)
    prefix = stem_prefix or path.stem
    written = 0
    for idx, chunk in enumerate(chunks):
        out_path = dst_dir / f"{prefix}_chunk{idx:04d}.wav"
        sf.write(str(out_path), chunk, cfg.SAMPLE_RATE, subtype="PCM_16")
        written += 1

    return written


def collect_audio_files(src_dir: Path) -> List[Path]:
    """Recursively collect all supported audio files under src_dir."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(src_dir.rglob(f"*{ext}"))
    return sorted(files)


# ──────────────────────────────────────────────────────────
#  Main batch processor
# ──────────────────────────────────────────────────────────

def preprocess_directory(
    src_dir: Path,
    dst_dir: Path,
    label: str,
    dry_run: bool = False,
) -> dict:
    """
    Process all audio files in src_dir, save chunks to dst_dir.
    Returns a stats dict.
    """
    files = collect_audio_files(src_dir)
    if not files:
        log.warning(f"No audio files found in {src_dir}")
        return {"files_found": 0, "chunks_written": 0, "skipped": 0}

    log.info(f"Found {len(files)} audio file(s) in '{src_dir}' [{label}]")

    total_chunks = 0
    skipped = 0

    for file_path in tqdm(files, desc=f"Processing [{label}]", unit="file"):
        n = process_single_file(
            path=file_path,
            dst_dir=dst_dir,
            stem_prefix=f"{label}_{file_path.stem}",
            dry_run=dry_run,
        )
        if n == 0:
            skipped += 1
        total_chunks += n

    stats = {
        "files_found":    len(files),
        "chunks_written": total_chunks,
        "skipped":        skipped,
        "dry_run":        dry_run,
    }
    log.info(
        f"[{label}] Done — {total_chunks} chunks from {len(files)-skipped} files "
        f"({skipped} skipped)."
    )
    return stats


def build_metadata_csv(processed_dir: Path, output_path: Optional[Path] = None) -> None:
    """
    Scan the processed directory structure (real/ and fake/) and
    emit a CSV manifest:  path, label, duration_sec
    """
    import pandas as pd

    rows = []
    for label_name, label_id in cfg.INV_LABEL_MAP.items():
        label_dir = processed_dir / label_name
        if not label_dir.exists():
            continue
        for f in sorted(label_dir.glob("*.wav")):
            try:
                info = sf.info(str(f))
                rows.append({
                    "path":         str(f),
                    "label":        label_name,
                    "label_id":     label_id,
                    "duration_sec": round(info.duration, 3),
                })
            except Exception:
                pass

    if not rows:
        log.warning("No processed .wav files found — CSV not written.")
        return

    df = pd.DataFrame(rows)
    out = output_path or processed_dir / "metadata.csv"
    df.to_csv(out, index=False)
    log.info(
        f"Metadata CSV written: {out}  |  "
        f"real={len(df[df.label=='real'])}  fake={len(df[df.label=='fake'])}"
    )


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio for deepfake detection training."
    )
    parser.add_argument("--src",     type=Path, required=True, help="Source directory of raw audio")
    parser.add_argument("--dst",     type=Path, required=True, help="Destination directory for processed audio")
    parser.add_argument("--label",   type=str,  required=True, choices=["real", "fake"])
    parser.add_argument("--dry-run", action="store_true",      help="Count chunks without writing")
    parser.add_argument(
        "--build-csv",
        action="store_true",
        help="After processing, build metadata.csv in the processed root",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = preprocess_directory(
        src_dir=args.src,
        dst_dir=args.dst,
        label=args.label,
        dry_run=args.dry_run,
    )
    print("\n── Preprocessing Stats ──────────────────")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    if args.build_csv and not args.dry_run:
        processed_root = args.dst.parent
        build_metadata_csv(processed_root)
