# 🎙️ Deepfake Audio Detection System

A production-grade binary classifier that distinguishes **real human speech** from **AI-generated (fake) audio** using a fine-tuned **Wav2Vec2** model. Designed for real-world conditions: YouTube audio, phone recordings, noisy environments, multiple accents and languages.

---

## 📁 Project Structure

```
deepfake_audio_detector/
├── config.py            # All hyperparameters and paths
├── preprocess.py        # Audio preprocessing pipeline
├── dataset.py           # PyTorch Dataset + DataLoader factory
├── model.py             # Wav2Vec2 + classification head
├── train.py             # Training loop (mixed precision, TensorBoard)
├── evaluate.py          # Metrics, confusion matrix, ROC, P-R curves
├── inference.py         # Inference engine (file → real/fake/uncertain)
├── api.py               # FastAPI REST backend
├── cli.py               # Rich CLI tool
├── data_collector.py    # Dataset collection utilities
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect data
```bash
# See recommended datasets
python data_collector.py guide

# Download FakeOrReal from HuggingFace (fake samples)
python data_collector.py hf-download \
    --dataset "dkounadis/artificial-speech" --label fake --max 3000

# Download LibriSpeech test-clean (real samples)
python data_collector.py hf-download \
    --dataset "librispeech_asr" --split "test.clean" --label real --out data/raw/real --max 3000

# Generate fake samples offline with pyttsx3
python data_collector.py generate-fake --n 500 --engine pyttsx3
```

### 3. Preprocess audio
```bash
python preprocess.py --src data/raw/real --dst data/processed/real --label real
python preprocess.py --src data/raw/fake --dst data/processed/fake --label fake --build-csv
```

### 4. Train
```bash
python train.py --output checkpoints/run1
```

### 5. Evaluate
```bash
python evaluate.py --model checkpoints/best_model --csv data/processed/metadata.csv
```

### 6. Inference
```bash
# Python API
python inference.py --model checkpoints/best_model --audio speech.wav

# Rich CLI
python cli.py detect speech.wav --model checkpoints/best_model --verbose

# Batch
python cli.py batch /path/to/audio_folder --model checkpoints/best_model
```

### 7. REST API
```bash
MODEL_DIR=checkpoints/best_model uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🏗️ Model Architecture

```
Input Audio (16kHz mono)
        │
        ▼
Wav2Vec2FeatureExtractor  (normalize + pad/truncate)
        │
        ▼
Wav2Vec2 CNN Feature Extractor  ← FROZEN
        │
        ▼
Wav2Vec2 Transformer Layers     ← FINE-TUNED
        │  last_hidden_state: (B, T', 768)
        ▼
Attention-Weighted Mean Pool    → (B, 768)
        │
        ▼
LayerNorm → Linear(768→256) → GELU → Dropout(0.3)
        │
        ▼
Linear(256→2)
        │
        ▼
Softmax → [P(real), P(fake)]
        │
        ▼
Threshold:
  P(fake) > 0.80  → "fake"
  P(real) > 0.80  → "real"
  else            → "uncertain"
```

---

## 📊 Inference Response

```json
{
  "label": "fake",
  "confidence": 0.9731,
  "real_prob": 0.0269,
  "fake_prob": 0.9731,
  "chunks": [
    { "label": "fake", "confidence": 0.9812, "real_prob": 0.0188, "fake_prob": 0.9812 },
    { "label": "fake", "confidence": 0.9650, "real_prob": 0.0350, "fake_prob": 0.9650 }
  ]
}
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect` | Analyse a single audio file |
| `POST` | `/detect/batch` | Analyse up to 20 files |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/model/info` | Model metadata |

**Example cURL:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@speech.wav" \
  -F "threshold=0.80"
```

---

## 🔧 Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `facebook/wav2vec2-base` | Base model |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHUNK_DURATION` | `4.0` | Chunk size (seconds) |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `BATCH_SIZE` | `8` | Training batch size |
| `NUM_EPOCHS` | `6` | Training epochs |
| `CONFIDENCE_THRESHOLD` | `0.80` | Uncertainty threshold |
| `FREEZE_FEATURE_EXTRACTOR` | `True` | Freeze CNN layers |
| `AUGMENT` | `True` | Enable data augmentation |

**Switch to larger model:**
```python
# In config.py:
MODEL_NAME = "facebook/wav2vec2-large"
```

---

## 📈 Recommended Datasets

### Real speech
| Dataset | Source | Notes |
|---------|--------|-------|
| LibriSpeech | HuggingFace | Clean English read speech |
| Common Voice | Mozilla / HF | Multi-accent, incl. Tamil |
| VoxCeleb1 | Oxford VGG | Celebrity YouTube speech |
| FLEURS | Google / HF | High-quality multilingual |

### Fake (AI-generated) speech
| Dataset | Source | Notes |
|---------|--------|-------|
| ASVspoof 2019 LA | asvspoof.org | Benchmark, multiple TTS systems |
| WaveFake | GitHub | 7 vocoder types |
| FakeOrReal | HuggingFace | AI vs real speech |
| In-The-Wild | deepfake-total.com | Real-world deepfakes ~38k samples |

---

## 🔊 Augmentation Pipeline

During training, the following augmentations are randomly applied:

| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| Gaussian noise | 30% | Simulate noisy environments |
| Reverb (FIR) | 20% | Simulate room acoustics |
| MP3 compression | 30% | Simulate YouTube / phone quality |
| Pitch shift ±2 st | 20% | Handle varied vocal pitches |

---

## 🧪 Evaluation Outputs

Running `evaluate.py` generates:
- `test_report.json` — full metrics + per-class report
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_precision_recall.png`
- `test_confidence_hist.png` — P(fake) distribution for real vs fake

---

## 🚀 Deployment Tips

### CPU-only server
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""
MODEL_DIR=checkpoints/best_model uvicorn api:app --workers 1
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV MODEL_DIR=/app/checkpoints/best_model
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Quantisation (faster CPU inference)
```python
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## ⚠️ Known Limitations

- Performance may degrade on very short clips (<1s) — pad to at least 1s
- Highly compressed audio (128kbps MP3) may reduce confidence
- Cross-lingual generalisation requires multilingual training data
- Voice cloning of a specific target speaker is harder to detect without speaker-specific training

---

## 📄 Citing

If you use this system in research, please cite:

- **Wav2Vec 2.0**: Baevski et al., 2020 — *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
- **ASVspoof 2019**: Todisco et al. — *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*
