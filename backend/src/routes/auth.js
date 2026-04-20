const express = require("express");
const router = express.Router();
const signupController = require("../controllers/auth/signup");
const loginController = require("../controllers/auth/login");
const logoutController = require("../controllers/auth/logout");
const verifyOtpController = require("../controllers/auth/verifyOtp");
const resendOtpController = require("../controllers/auth/resendOtp");
const { authLimiter } = require("../middlewares/rateLimiter");

// Apply authLimiter to all auth routes
router.use(authLimiter);

router.post("/signup", signupController.signup);
router.post("/login", loginController.login);
router.post("/logout", logoutController.logout);
router.post("/verify-otp", verifyOtpController.verifyOtp);
router.post("/resend-verification", resendOtpController.resendOtp);

module.exports = router;
