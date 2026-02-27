const AppError = require("../utils/AppError");

module.exports = (err, req, res, next) => {
  console.error("🔥 ERROR:", err);

  const statusCode = err.statusCode || 500;
  const message =
    err.message || "Something went wrong. Please try again later.";

  res.status(statusCode).json({
    success: false,
    status: err.status || "error",
    message
  });
};