const mongoose = require("mongoose");

const AuthLogSchema = new mongoose.Schema({
  email: String,

  action: {
    type: String,
    enum: [
      "SIGNUP",
      "LOGIN_SUCCESS",
      "LOGIN_FAIL",
      "OTP_SENT",
      "EMAIL_VERIFIED",
      "LOGOUT"
    ]
  },

  ip: String,
  userAgent: String
}, {
  timestamps: true
});

module.exports = mongoose.model("AuthLog", AuthLogSchema);