### Initial Commit

initializied with fogoe

### Issue 2: Added Global Error Handlers

Implemented a centralized global error handling mechanism in the Express.js application to handle both synchronous and asynchronous errors consistently. All async route errors are now captured using a wrapper function and forwarded to a single error-handling middleware, ensuring proper HTTP status codes and standardized error responses. This improves code readability, prevents server crashes caused by unhandled promise rejections, and makes the application more reliable and easier to maintain.

### Issue 3: Finalized Auth Backend & Implemented Rate Limiting

Wired up the complete authentication backend (signup, login, logout) with secure JWT session tracking. Fixed underlying database queries for user validation and established a protected profile endpoint. To secure the application, a custom rate-limiting middleware was deployed across all API and authentication routes to prevent abuse and brute-force attacks.

### Issue 4: Deepfake Audio Detection Service
Developed a production-ready binary classifier for detecting real human speech versus AI-generated (fake) audio. The system leverages a fine-tuned Wav2Vec2 model and includes a high-performance FastAPI backend supporting both single and batch audio detection. The service is optimized for real-world environmental noise and varied vocal samples, featuring a complete pipeline for audio preprocessing, multi-epoch training with mixed precision, and detailed evaluation metrics.
