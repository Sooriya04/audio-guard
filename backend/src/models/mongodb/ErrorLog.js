// models/errorLog.model.js
const mongoose = require('mongoose');

const errorLogSchema = new mongoose.Schema(
  {
    message: {
      type: String,
      required: true,
    },
    stack: String,
    statusCode: {
      type: Number,
      default: 500,
    },
    route: String,
    method: String,
    ip: String,
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
    },
    isOperational: {
      type: Boolean,
      default: false,
    },
  },
  { timestamps: true }
);

errorLogSchema.index({ createdAt: -1 });

module.exports = mongoose.model('ErrorLog', errorLogSchema);