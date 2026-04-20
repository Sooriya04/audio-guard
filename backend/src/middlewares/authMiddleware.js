// middlewares/auth.middleware.js
const jwt = require("jsonwebtoken");
const JwtSession = require("../models/mongodb/JwtSession");
const AppError = require("../utils/AppError");

module.exports = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return next(new AppError("Unauthorized", 401));
    }

    const token = authHeader.split(" ")[1];

    // 1️⃣ Find session
    const session = await JwtSession.findOne({
      accessToken: token,
      isValid: true
    });

    if (!session) {
      return next(new AppError("Session expired or revoked", 401));
    }

    // 2️⃣ Verify JWT using stored secret
    const decoded = jwt.verify(token, session.accessSecret);

    // 3️⃣ Attach user to request
    req.user = {
      id: decoded.userId,
      email: decoded.email
    };

    next();
  } catch (e) {
    return next(new AppError("Invalid or expired token", 401));
  }
};