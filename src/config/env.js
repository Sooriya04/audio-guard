require("dotenv").config();

module.exports = {
  PORT: process.env.PORT || 3000,
  JWT_SECRET: process.env.JWT_SECRET || "dev-secret",
  DATABASE_URL: process.env.DATABASE_URL || ""
};