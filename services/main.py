import os
import time
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import pipeline
import logging
from pymongo import MongoClient
from datetime import datetime, timezone

# Load from backend .env to get MONGODB if possible, else local
load_dotenv()
load_dotenv("../backend/.env")

# Setup Logging
class MongoHandler(logging.Handler):
    def __init__(self, uri, database, collection):
        logging.Handler.__init__(self)
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # test connection
            self.client.server_info() 
            self.db = self.client[database]
            self.collection = self.db[collection]
        except Exception as e:
            print(f"Failed to connect to MongoDB for logging: {e}")
            self.client = None

    def emit(self, record):
        if self.client is not None:
            log_document = {
                "timestamp": datetime.now(timezone.utc),
                "level": record.levelname,
                "message": self.format(record),
                "module": record.module,
                "funcName": record.funcName,
                "lineNo": record.lineno,
            }
            try:
                self.collection.insert_one(log_document)
            except Exception:
                pass

logger = logging.getLogger("VoiceSafeML")
logger.setLevel(logging.DEBUG)

# Console Handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# MongoDB Handler
mongodb_uri = os.environ.get("MONGODB", "mongodb://localhost:27017/audio_guard")
# Usually the DB is part of the URI, but we can extract or just use 'audio_guard'
mongo_handler = MongoHandler(uri=mongodb_uri, database="audio_guard", collection="python_logs")
mongo_handler.setLevel(logging.INFO)
mongo_handler.setFormatter(formatter)
logger.addHandler(mongo_handler)

app = FastAPI(title="VoiceSafe ML Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "Shanmugapriya6/voice-fake-detector-v1"
logger.info(f"Loading model {MODEL_ID} locally...")
try:
    pipe = pipeline("audio-classification", model=MODEL_ID)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    pipe = None

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    start_time = time.time()
    logger.debug(f"Processing file: {file.filename}")
    
    if pipe is None:
        return {
            "success": False,
            "error": "ModelNotLoaded",
            "message": "The ML model failed to load at startup. Check service logs."
        }
    
    try:
        audio_content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(audio_content)
            tmp_path = tmp.name
        
        try:
            logger.debug("Running local inference...")
            outputs = pipe(tmp_path)
            logger.debug(f"Model Output: {outputs}")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        latency = time.time() - start_time
        
        # Logic from result.ipynb
        # Find the label with the highest score
        if not isinstance(outputs, list) or len(outputs) == 0:
            raise HTTPException(status_code=500, detail="Invalid response from Model")
            
        best_prediction = max(outputs, key=lambda x: x['score'])
        
        # Determine if it's REAL or FAKE
        # Based on user logic: LABEL_0 is REAL, others are FAKE
        is_real = best_prediction['label'] == 'LABEL_0'
        result = 'REAL' if is_real else 'FAKE'
        confidence = best_prediction['score']

        # Extra metrics
        scores = {item['label']: item['score'] for item in outputs}
        
        return {
            "success": True,
            "prediction": {
                "result": result,
                "label": best_prediction['label'],
                "confidence": round(confidence * 100, 2),
                "is_fake": not is_real
            },
            "metrics": {
                "latency_seconds": round(latency, 4),
                "all_scores": scores,
                "model": MODEL_ID,
                "engine": "local-transformers"
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": type(e).__name__,
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
