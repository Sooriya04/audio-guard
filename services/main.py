import os
import time
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

app = FastAPI(title="VoiceSafe ML Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "Shanmugapriya6/voice-fake-detector-v1"
print(f"INFO: Loading model {MODEL_ID} locally...")
try:
    pipe = pipeline("audio-classification", model=MODEL_ID)
    print("INFO: Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    pipe = None

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    start_time = time.time()
    print(f"DEBUG: Processing file: {file.filename}")
    
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
            print(f"DEBUG: Running local inference...")
            outputs = pipe(tmp_path)
            print(f"DEBUG: Model Output: {outputs}")
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
        print(f"ERROR: {str(e)}")
        return {
            "success": False,
            "error": type(e).__name__,
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
