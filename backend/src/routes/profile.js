const express = require("express");
const router = express.Router();
const profileController = require("../controllers/user/profile");
const authMiddleware = require("../middlewares/authMiddleware");

// Protected routes
router.get("/", authMiddleware, profileController.getProfile);
router.patch("/", authMiddleware, profileController.updateProfile);

module.exports = router;
