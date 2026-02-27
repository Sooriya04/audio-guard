const mongoose = require("mongoose");

const JwtSessionSchema = new mongoose.Schema({
  userId: {
    type: Number, 
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
}, {
  timestamps: true
});

module.exports = mongoose.model("JwtSession", JwtSessionSchema);