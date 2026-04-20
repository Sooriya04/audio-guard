require("dotenv").config({ path: "../../.env" });


module.exports = {
  PORT: process.env.PORT || 3000,
  JWT_SECRET: process.env.JWT_SECRET || "dev-secret",
  POSTGRESQL: process.env.POSTGRESQL,
  MONGODB : process.env.MONGODB,
  MAILER_USER: process.env.MAILER_USER,
  MAILER_PASS: process.env.MAILER_PASS,
  MAILER_HOST: process.env.MAILER_HOST,
  MAILER_PORT: process.env.MAILER_PORT
};