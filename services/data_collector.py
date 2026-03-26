"""
data_collector.py — Helpers to build a labelled training dataset.

This module provides:
  1. Guides / manifests for free-to-use datasets
  2. A TTS fake audio generator using gTTS / pyttsx3 (offline)
  3. A HuggingFace dataset downloader (ASVspoof, FakeOrReal, etc.)
  4. A YouTube audio downloader wrapper (yt-dlp)
  5. Dataset balancing utility

Usage
-----
  # Generate fake audio samples from text
  python data_collector.py generate-fake --text-file prompts.txt --out data/raw/fake --n 200

  # Download a HuggingFace dataset
  python data_collector.py hf-download --dataset "dkounadis/artificial-speech" --out data/raw/fake

  # Balance the dataset
  python data_collector.py balance --real data/raw/real --fake data/raw/fake
"""

import argparse
import logging
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Free public datasets reference
# ──────────────────────────────────────────────────────────

PUBLIC_DATASETS = {
    "real": [
        {
            "name":   "LibriSpeech (test-clean)",
            "hf_id":  "librispeech_asr",
            "split":  "test.clean",
            "label":  "real",
            "notes":  "~5h clean read speech, English. Free, CC BY 4.0.",
        },
        {
            "name":   "Common Voice (en)",
            "hf_id":  "mozilla-foundation/common_voice_13_0",
            "split":  "test",
            "label":  "real",
            "notes":  "Crowd-sourced, multiple accents including Tamil (ta).",
        },
        {
            "name":   "VoxCeleb1",
            "hf_id":  None,
            "url":    "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
            "label":  "real",
            "notes":  "Celebrity speech from YouTube. Requires registration.",
        },
        {
            "name":   "FLEURS (Tamil)",
            "hf_id":  "google/fleurs",
            "split":  "ta_in.test",
            "label":  "real",
            "notes":  "High-quality Tamil read speech.",
        },
    ],
    "fake": [
        {
            "name":   "ASVspoof 2019 LA",
            "hf_id":  None,
            "url":    "https://www.asvspoof.org/",
            "label":  "fake",
            "notes":  "Benchmark dataset with multiple TTS + VC systems.",
        },
        {
            "name":   "FakeOrReal",
            "hf_id":  "dkounadis/artificial-speech",
            "split":  "train",
            "label":  "fake",
            "notes":  "AI-generated vs real speech. HuggingFace hosted.",
        },
        {
            "name":   "WaveFake",
            "hf_id":  None,
            "url":    "https://github.com/RUB-SysSec/WaveFake",
            "label":  "fake",
            "notes":  "7 vocoder types, large scale.",
        },
        {
            "name":   "In-The-Wild",
            "hf_id":  None,
            "url":    "https://deepfake-total.com/in_the_wild",
            "label":  "fake",
            "notes":  "Real-world deepfakes. ~38k samples.",
        },
    ],
}


def print_dataset_guide():
    print("\n── Recommended Datasets ─────────────────────────────────────────")
    for split, datasets in PUBLIC_DATASETS.items():
        print(f"\n  [{split.upper()}]")
        for d in datasets:
            hf = d.get("hf_id") or d.get("url", "—")
            print(f"    • {d['name']}")
            print(f"      Source : {hf}")
            print(f"      Notes  : {d['notes']}")
    print()


# ──────────────────────────────────────────────────────────
#  HuggingFace dataset downloader
# ──────────────────────────────────────────────────────────

def download_hf_dataset(
    dataset_id: str,
    split: str = "train",
    output_dir: Path = cfg.FAKE_RAW,
    label: str = "fake",
    max_samples: int = 5000,
):
    """
    Stream audio from a HuggingFace dataset and save as .wav files.
    Works for datasets with an 'audio' column (AudioFolder compatible).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install `datasets`: pip install datasets")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {dataset_id} ({split}) → {output_dir}")

    ds = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)
    # Disable automatic decoding to avoid torchcodec/ffmpeg issues
    if "audio" in ds.column_names:
        from datasets import Audio
        ds = ds.cast_column("audio", Audio(decode=False))

    saved = 0
    for i, sample in enumerate(tqdm(ds, desc=f"Downloading [{label}]", total=max_samples)):
        if saved >= max_samples:
            break

        audio_data = sample.get("audio") or sample.get("speech")
        if audio_data is None:
            continue

        # Decode manually from bytes/path
        if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
            import io
            import soundfile as sf
            array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        elif isinstance(audio_data, dict) and "path" in audio_data and audio_data["path"]:
            import librosa
            array, sr = librosa.load(audio_data["path"], sr=None)
        else:
            # Fallback for already decoded or other formats
            array = np.array(audio_data["array"], dtype=np.float32)
            sr    = audio_data["sampling_rate"]

        # Resample if needed
        if sr != cfg.SAMPLE_RATE:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)

        out_path = output_dir / f"{label}_{i:06d}.wav"
        sf.write(str(out_path), array, cfg.SAMPLE_RATE, subtype="PCM_16")
        saved += 1

    log.info(f"Saved {saved} samples to {output_dir}")


# ──────────────────────────────────────────────────────────
#  Offline TTS fake audio generator
# ──────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work and communicate.",
    "Please verify your identity before proceeding with the transaction.",
    "The weather forecast shows rain throughout the week.",
    "She sells seashells by the seashore.",
    "To be or not to be, that is the question.",
    "The stock market closed higher today driven by technology stocks.",
    "நான் தமிழ் பேசுகிறேன். இது ஒரு பரிசோதனை.",   # Tamil
    "Emergency services have been dispatched to the scene.",
    "Your package has been shipped and will arrive in three days.",
]


def generate_fake_with_gtts(
    prompts: List[str],
    output_dir: Path = cfg.FAKE_RAW,
    n: int = 100,
    lang: str = "en",
):
    """Generate fake audio using gTTS (Google Text-to-Speech — requires internet)."""
    try:
        from gtts import gTTS
    except ImportError:
        log.error("Install gTTS: pip install gtts")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for i in tqdm(range(n), desc="gTTS generation"):
        prompt = random.choice(prompts)
        tts    = gTTS(text=prompt, lang=lang, slow=False)
        mp3    = output_dir / f"gtts_{i:06d}.mp3"
        tts.save(str(mp3))
        generated += 1

    log.info(f"Generated {generated} gTTS samples in {output_dir}")


def generate_fake_with_pyttsx3(
    prompts: List[str],
    output_dir: Path = cfg.FAKE_RAW,
    n: int = 100,
):
    """Generate fake audio using pyttsx3 (fully offline TTS)."""
    try:
        import pyttsx3
    except ImportError:
        log.error("Install pyttsx3: pip install pyttsx3")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()

    # Vary voice properties slightly for diversity
    voices = engine.getProperty("voices")
    generated = 0

    for i in tqdm(range(n), desc="pyttsx3 generation"):
        prompt = random.choice(prompts)
        # Alternate between available voices
        if voices:
            engine.setProperty("voice", voices[i % len(voices)].id)
        rate = random.randint(140, 200)
        engine.setProperty("rate", rate)

        out_path = str(output_dir / f"pyttsx3_{i:06d}.wav")
        engine.save_to_file(prompt, out_path)
        engine.runAndWait()
        generated += 1

    log.info(f"Generated {generated} pyttsx3 samples in {output_dir}")


# ──────────────────────────────────────────────────────────
#  YouTube downloader (real speech)
# ──────────────────────────────────────────────────────────

def download_youtube_audio(
    urls: List[str],
    output_dir: Path = cfg.REAL_RAW,
    start_sec: int = 0,
    duration_sec: int = 30,
):
    """
    Download audio from YouTube URLs using yt-dlp.
    Requires: pip install yt-dlp
    """
    if not shutil.which("yt-dlp"):
        log.error("yt-dlp not found. Install: pip install yt-dlp")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(tqdm(urls, desc="YouTube download")):
        out_template = str(output_dir / f"yt_{i:04d}.%(ext)s")
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ss {start_sec} -t {duration_sec}",
            "-o", out_template,
            url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            log.warning(f"Failed to download {url}: {e.stderr.decode()[:200]}")

    log.info(f"YouTube downloads saved to {output_dir}")


# ──────────────────────────────────────────────────────────
#  Dataset balancing
# ──────────────────────────────────────────────────────────

def balance_dataset(
    real_dir: Path = cfg.REAL_RAW,
    fake_dir: Path = cfg.FAKE_RAW,
    strategy: str = "undersample",
):
    """
    Balance real/fake audio counts.
    strategy: 'undersample' | 'oversample'
    """
    real_files = list(real_dir.glob("*.*"))
    fake_files = list(fake_dir.glob("*.*"))

    n_real = len(real_files)
    n_fake = len(fake_files)
    log.info(f"Before balancing — real: {n_real}, fake: {n_fake}")

    if n_real == n_fake:
        log.info("Already balanced.")
        return

    minority_n = min(n_real, n_fake)
    majority_n = max(n_real, n_fake)

    if strategy == "undersample":
        majority_files = real_files if n_real > n_fake else fake_files
        to_remove = random.sample(majority_files, majority_n - minority_n)
        for f in tqdm(to_remove, desc="Removing excess files"):
            f.unlink()
        log.info(f"Removed {len(to_remove)} files — now balanced at {minority_n} each.")

    elif strategy == "oversample":
        minority_files = fake_files if n_real > n_fake else real_files
        minority_dir   = fake_dir   if n_real > n_fake else real_dir
        needed = majority_n - minority_n
        for i in range(needed):
            src = random.choice(minority_files)
            dst = minority_dir / f"oversample_{i:06d}{src.suffix}"
            shutil.copy2(src, dst)
        log.info(f"Duplicated {needed} files for oversampling — now balanced at {majority_n} each.")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset collection utilities.")
    sub    = parser.add_subparsers(dest="command")

    # guide
    sub.add_parser("guide", help="Print a guide to recommended public datasets.")

    # generate-fake
    gen = sub.add_parser("generate-fake", help="Generate fake TTS audio.")
    gen.add_argument("--text-file", type=Path, default=None)
    gen.add_argument("--out",       type=Path, default=cfg.FAKE_RAW)
    gen.add_argument("--n",         type=int,  default=200)
    gen.add_argument("--engine",    choices=["gtts", "pyttsx3"], default="pyttsx3")
    gen.add_argument("--lang",      default="en")

    # hf-download
    hf = sub.add_parser("hf-download", help="Download from a HuggingFace dataset.")
    hf.add_argument("--dataset",  required=True)
    hf.add_argument("--split",    default="train")
    hf.add_argument("--out",      type=Path, default=cfg.FAKE_RAW)
    hf.add_argument("--label",    default="fake", choices=["real", "fake"])
    hf.add_argument("--max",      type=int, default=5000)

    # balance
    bal = sub.add_parser("balance", help="Balance real/fake class counts.")
    bal.add_argument("--real",     type=Path, default=cfg.REAL_RAW)
    bal.add_argument("--fake",     type=Path, default=cfg.FAKE_RAW)
    bal.add_argument("--strategy", choices=["undersample", "oversample"], default="undersample")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "guide":
        print_dataset_guide()

    elif args.command == "generate-fake":
        prompts = SAMPLE_PROMPTS
        if args.text_file and args.text_file.exists():
            prompts = args.text_file.read_text().splitlines()
            prompts = [p.strip() for p in prompts if p.strip()]
        if args.engine == "gtts":
            generate_fake_with_gtts(prompts, args.out, args.n, args.lang)
        else:
            generate_fake_with_pyttsx3(prompts, args.out, args.n)

    elif args.command == "hf-download":
        download_hf_dataset(args.dataset, args.split, args.out, args.label, args.max)

    elif args.command == "balance":
        balance_dataset(args.real, args.fake, args.strategy)

    else:
        print("Run with --help for usage.")
