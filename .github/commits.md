### Initial Commit
initializied with fogoe

### Issue 2: Added Global Error Handlers
Implemented a centralized global error handling mechanism in the Express.js application to handle both synchronous and asynchronous errors consistently. All async route errors are now captured using a wrapper function and forwarded to a single error-handling middleware, ensuring proper HTTP status codes and standardized error responses. This improves code readability, prevents server crashes caused by unhandled promise rejections, and makes the application more reliable and easier to maintain.