# 🎙️ Deepfake Audio Detection System - Setup Guide

This guide explains how to set up, train, and run the Deepfake Audio Detection system.

## 🛠️ Step 1: Environment Setup
First, install the core dependencies. Note that `torch` and `cuda` are large (500MB+); ensure you have a stable connection.

```bash
pip install -r requirements.txt
```

> [!TIP]
> If you only want to handle data/audio collection for now, you can install just these libs first:
> `pip install datasets librosa soundfile tqdm pyttsx3`

## 🧪 Minimal Test (10 Real + 10 Fake)
To quickly verify the pipeline with a small dataset:

### 1. Download 10 Real Samples
```bash
python data_collector.py hf-download --dataset "librispeech_asr" --split "test.clean" --label real --out data/raw/real --max 10
```

### 2. Generate 10 Fake Samples (Offline)
```bash
python data_collector.py generate-fake --n 10 --engine pyttsx3 --out data/raw/fake
```

## 🧹 Step 2: Preprocessing
Normalize audio to 16kHz mono and generate the metadata mapping.

```bash
# Preprocess real samples
python preprocess.py --src data/raw/real --dst data/processed/real --label real

# Preprocess fake samples and build the metadata file
python preprocess.py --src data/raw/fake --dst data/processed/fake --label fake --build-csv
```

## 🚂 Step 3: Training
Fine-tune the Wav2Vec2 model.

```bash
python train.py --output checkpoints/detection_model
```

## 🚀 Step 4: Running Inference
Use your trained model to detect deepfakes.

### Using the CLI
**IMPORTANT**: Always point to the `best_model` subfolder.
```bash
python cli.py detect path/to/your/audio.wav --model checkpoints/detection_model/best_model
```

### Starting the Web API
```bash
MODEL_DIR=checkpoints/detection_model/best_model uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🌐 Offline Usage
*   **First Run**: `train.py` will download `wav2vec2-base` (~380MB) once.
*   **Later Runs**: Files are cached locally and used automatically.
*   **Force Offline**:
    ```bash
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ```

> [!NOTE]
> Training is significantly faster if you have a CUDA-enabled GPU.
