// models/jwtSession.model.js
const mongoose = require("mongoose");

const JwtSessionSchema = new mongoose.Schema(
  {
    userId: {
      type: Number, // PostgreSQL user id
      required: true
    },
    email: {
      type: String,
      required: true
    },

    accessToken: {
      type: String,
      required: true
    },

    accessSecret: {
      type: String,
      required: true
    },

    refreshToken: {
      type: String,
      required: true
    },

    isValid: {
      type: Boolean,
      default: true
    },

    expiresAt: {
      type: Date,
      required: true
    }
  },
  { timestamps: true }
);

module.exports = mongoose.model("JwtSession", JwtSessionSchema);