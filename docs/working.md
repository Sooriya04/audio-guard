# Audio-Guard System Architecture

The Audio-Guard application uses a modern microservices architecture designed to decouple user management from computationally intensive machine learning inference. The system is split into two primary components: the **Backend (Node.js/Express)** and the **AI Services (Python/FastAPI)**.

## 1. Backend (Node.js/Express)
The backend is the central API gateway that manages user authentication, state, and rate-limiting. It is built using **Node.js** and **Express.js**, and interacts with a **MongoDB** database. 

### Core Components
- **Authentication System**: Secures the application with JWT-based session management (`/api/auth/signup`, `/api/auth/login`, `/api/auth/logout`).
- **Profile Management**: Handles user profiles (`/api/profile`), storing relevant user data and metadata in MongoDB.
- **Middleware & Security**: 
  - Uses `cors` for cross-origin resource sharing.
  - Implements API rate limiting (`apiLimiter`, `authLimiter`) to prevent brute force attacks and overloading. 
  - Standardized error handling (`errorHandler`).
- **Database Architecture**: Connects to MongoDB (configured via Mongoose), with an organized schema model pattern (e.g., `User`, `JwtSession`).

### Request Flow
1. Client sends requests (e.g., sign up, fetch profile) to the Node.js API.
2. The Node.js application validates requests through rate-limiting and JWT authentication middleware.
3. Upon validation, the controller executes the business logic, interacting with the MongoDB database using Mongoose schemas.
4. An HTTP response is sent back to the client.

---

## 2. AI Services (Python/FastAPI)
The service layer forms a complete Machine Learning lifecycle pipeline—it covers data gathering, preprocessing, training, evaluation, and finally, inference via a REST API. 

Here is the full flow and breakdown of each script in the `services/` directory:

### 1. `config.py` (Configuration Hub)
* **Flow**: Acts as the single source of truth for the entire pipeline. It defines hyperparameters (learning rates, clip norms), file paths (raw data, processed data, checkpoints), and audio constants (16kHz sample rate, target RMS). All other scripts import and rely on these constants.

### 2. `data_collector.py` (Data Gathering)
* **Flow**: The entry point for building a dataset. 
  - It can download open-source HuggingFace datasets (`hf-download`).
  - It uses offline/online Text-to-Speech engines like `gTTS` or `pyttsx3` to artificially fabricate deepfake audio clips (`generate-fake`).
  - It provides wrappers to download clean "real" audio from YouTube via `yt-dlp`. 
  - It balances the gathered datasets with an `oversample` or `undersample` strategy to prevent class imbalance.

### 3. `preprocess.py` (Cleaning & Chunking)
* **Flow**: Takes the raw `.wav`, `.mp3`, or `.flac` files and normalizes them for the neural network. 
  - Audio is resampled strictly to mono 16kHz (`to_mono_16k`).
  - Librosa trims leading or trailing silence.
  - The amplitude is normalized (RMS-normalized to a target dB).
  - Crucially, it splits long audios into fixed-length "chunks" (e.g., 3-second or 5-second segments) to standardise input sizes for the model. 
  - Results are written out to a `/processed` directory and mapped in a `metadata.csv`.

### 4. `dataset.py` (PyTorch Data Loading & Augmentation)
* **Flow**: Creates PyTorch `Dataset` and `DataLoader` objects that feed the model during training. 
  - It parses the processed audio directories or the `metadata.csv`.
  - Applies **on-the-fly random augmentations** (adding Gaussian noise, FIR reverb, bitrate compression simulations, and pitch shifting) to make the model robust against real-world degraded audio.
  - Prepares the chunks into PyTorch tensors and batches them up.

### 5. `model.py` (Neural Network Architecture)
* **Flow**: Defines the heart of the deepfake detector (`DeepfakeAudioDetector`). 
  - Uses a pretrained **Wav2Vec2** model from HuggingFace. The CNN feature extractor reads the raw audio waveforms, passing them through transformer layers to get hidden states.
  - Sits a custom PyTorch `ClassificationHead` on top, which performs weighted mean-pooling, passes the pooled states through a 2-layer MLP (LayerNorm → Linear → GELU → Dropout → Linear), and outputs logits to two dimensions: **real** and **fake**.

### 6. `train.py` & `evaluate.py` (Model Training)
* **Flow**: The automated ML training loop.
  - Loads the data loaders and the model. 
  - Runs batches through the `AdamW` optimizer using FP16 mixed precision.
  - Uses cosine learning rate scheduling.
  - Calculates accuracy, F1 scores, and ROC-AUC via `evaluate.py`.
  - Automatically snapshots the best-performing `Wav2Vec2` model checkpoint to the disk (`/checkpoints/best_model`).

### 7. `inference.py` (Inference Engine)
* **Flow**: The standalone predictor script.
  - Instantiates `DeepfakeDetector()`, which points to a saved checkpoint.
  - Given an audio file, it applies the exact same preprocessing and chunking used in `preprocess.py`.
  - Runs the chunks through the model to get class logits, applies softmax, and aggregates chunk confidence probabilities to decide if an audio file is real, fake, or "uncertain" (if it falls beneath a confidence threshold).

### 8. `api.py` (FastAPI Server)
* **Flow**: The HTTP wrapper around `inference.py`.
  - Starts a Uvicorn server.
  - Accepts multipart `File` uploads on `/detect` or `/detect/batch`.
  - Flushes the files to a temporary disk location.
  - Invokes `inference.py` to analyze the audio and returns JSON responses specifying class labels, specific piece-by-piece chunk breakdowns, probabilities, and latency metadata.
