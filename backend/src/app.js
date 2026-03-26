require("dotenv").config();

const express = require("express");
const cors = require("cors");
const errorHandler = require("./middlewares/errorHandler");
const app = express();
app.use(cors());

require("./config/db");
require("./config/mongoose");
app.use(express.json());

const { apiLimiter } = require("./middlewares/rateLimiter");
app.use("/api", apiLimiter);

// Routes
const authRoutes = require("./routes/auth");
const profileRoutes = require("./routes/profile");

app.use("/api/auth", authRoutes);
app.use("/api/profile", profileRoutes);

app.use(errorHandler)
module.exports = app;