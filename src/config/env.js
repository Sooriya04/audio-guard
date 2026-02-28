require("dotenv").config({ path: "../../.env" });


module.exports = {
  PORT: process.env.PORT || 3000,
  JWT_SECRET: process.env.JWT_SECRET || "dev-secret",
  POSTGRESQL: process.env.POSTGRESQL,
  MONGODB : process.env.MONGODB
};