const ErrorLog = require('../models/mongodb/ErrorLog');

const errorHandler = async (err, req, res, next) => {
  console.error('Global Error Handler:', err);

  const statusCode = err.statusCode || 500;
  const message = err.isOperational
    ? err.message
    : 'Internal Server Error';

  const errorDetails = {
    message: err.message,
    stack: err.stack,
    statusCode,
    route: req.originalUrl,
    method: req.method,
    ip: req.ip,
    isOperational: err.isOperational || false,
  };

  try {
    await ErrorLog.create(errorDetails);
  } catch (dbErr) {
    console.error('Error log save failed:', dbErr.message);
  }

  if (res.headersSent) {
    return next(err);
  }

  res.status(statusCode).json({
    success: false,
    message,
  });
};

module.exports = errorHandler;