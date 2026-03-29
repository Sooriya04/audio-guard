# Integration Guide: Kotlin App & HuggingFace Deployment

Integrating the **Audio-Guard** services into a modern stack involves connecting the client (Kotlin Mobile Application) to the primary APIs, and effectively hosting the heavy Deepfake detection service (on HuggingFace).

---

## 1. Kotlin Mobile App Integration

Your Kotlin app must interface with both the **Node.js Backend** (for user/session state) and the **Python Service** (for deepfake analysis). Depending on architectural preferences, the Kotlin app can either talk directly to both or send all requests to the Node.js backend, which acts as a proxy to the Python service. 

### Recommended Architecture (Direct-To-Service)

**Phase 1: User Authentication (Node.js API)**
- Use Retrofit/OkHttp in Kotlin to communicate with the Node.js `/api/auth/...` endpoints. 
- Securely store the returned JWT using Android `EncryptedSharedPreferences` or the Android Keystore system.
- Attach the JWT to the `Authorization: Bearer <token>` header for subsequent profile or protected backend requests.

**Phase 2: Audio Upload & Detection (Python API)**
- Use Android's `MediaRecorder` or `AudioRecord` APIs to capture the user's audio or select files from device storage.
- Construct a Retrofit `MultipartBody.Part` to stream the audio file directly to the Python service’s `/detect` or `/detect/batch` using a `POST` request.
- Parse the returned JSON to update the Android UI (e.g., displaying "Real Audio: 94% Confidence").
- Use Kotlin Coroutines and `ViewModel` to ensure network requests happen asynchronously so the main UI thread stays unblocked.

### Example Kotlin Retrofit Setup (Audio Upload)

```kotlin
interface AudioGuardApiService {
    @Multipart
    @POST("/detect")
    suspend fun detectAudio(
        @Part file: MultipartBody.Part, 
        @Query("threshold") threshold: Float = 0.5f
    ): Response<DetectionResult>
}
```

---

## 2. Deploying the AI Service to HuggingFace (Future)

HuggingFace offers **Inference Endpoints** and **Spaces**, which are robust and auto-scaling environments specifically designed for hosting ML inference models like your PyTorch `Wav2Vec2` detector.

### Approach 1: HuggingFace Spaces (Docker)
Since you already have a `FastAPI` application (`api.py`), the simplest and most customizable approach is to deploy it using a **HuggingFace Docker Space**.

1. **Prepare a `Dockerfile`**: In your `services/` directory, create a Docker image that leverages an official Python environment or PyTorch CUDA image.
2. **Install Dependencies**: Copy over the your `requirements.txt` and install all necessary pip packages. 
3. **Include the Checkpoint**: Ensure that the `checkpoints/best_model` (your compiled model weights) is either downloaded dynamically via a script during the build or pushed to the Space's Git LFS. 
4. **Boot the FastAPI Server**: Your Dockerfile's `CMD` command should look like this:
   ```dockerfile
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
   *(Note: HuggingFace Spaces expose port 7860 by default).*

### Approach 2: HuggingFace Inference Endpoints
If this application must scale to thousands of users, deploy it via **Inference Endpoints**.

1. **Upload the Model Repository**: Create a new Model repository on HuggingFace and commit your compiled custom `Wav2Vec2` weights into it (using `git lfs`).
2. **Custom Handler**: Write a custom `handler.py` script. The Inference Endpoint infrastructure expects an initialized pipeline. You will integrate the custom `ClassificationHead` defined in your `model.py` to be invoked during the Endpoint's inference route.
3. **Deploy & Integrate**: Set up an Inference Endpoint in HuggingFace (choosing CPU or GPU instance sizing). HuggingFace will provide a secure HTTPS endpoint `https://api-inference.huggingface.co/models/<username>/<repo>`. 
4. **Kotlin Update**: Point your Kotlin Retrofit client directly to the provided Inference URL and attach your HuggingFace API key in the Authorization headers.

By decoupling the Node.js management API (which can reside cheaply on standard VPS services like Render or AWS EC2) and pushing the heavy ML logic to HuggingFace, your Kotlin application achieves an optimal, highly scalable data flow.
