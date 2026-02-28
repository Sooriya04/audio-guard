const express = require("express");
const router = express.Router();
const profileController = require("../controllers/user/profile");
const authMiddleware = require("../middlewares/authMiddleware");

// Protected route
router.get("/", authMiddleware, profileController.getProfile);

module.exports = router;
