const crypto = require("crypto");

/**
 * Generates a 6-digit numeric OTP
 * @returns {string} 6-digit OTP
 */
const generateOTP = () => {
  return crypto.randomInt(100000, 999999).toString();
};

module.exports = { generateOTP };
