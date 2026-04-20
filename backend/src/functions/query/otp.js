const pool = require("../../config/db");

/**
 * Saves an OTP to the database
 */
const saveOTP = async (email, otp, purpose, expiresAt) => {
  const query = `
    INSERT INTO otp_codes (email, otp, purpose, expires_at)
    VALUES ($1, $2, $3, $4)
    RETURNING *;
  `;
  const values = [email, otp, purpose, expiresAt];
  const { rows } = await pool.query(query, values);
  return rows[0];
};

/**
 * Verifies if an OTP exists and is valid
 */
const verifyOTP = async (email, otp, purpose) => {
  const query = `
    SELECT * FROM otp_codes
    WHERE email = $1 AND otp = $2 AND purpose = $3 
    AND is_used = FALSE 
    AND expires_at > CURRENT_TIMESTAMP
    ORDER BY created_at DESC
    LIMIT 1;
  `;
  const values = [email, otp, purpose];
  const { rows } = await pool.query(query, values);
  return rows[0];
};

/**
 * Marks an OTP as used
 */
const markOTPAsUsed = async (id) => {
  const query = `
    UPDATE otp_codes
    SET is_used = TRUE
    WHERE id = $1;
  `;
  await pool.query(query, [id]);
};

/**
 * Deletes expired or unused OTPs for an email (cleanup)
 */
const deleteExistingOTPs = async (email, purpose) => {
  const query = `
    DELETE FROM otp_codes
    WHERE email = $1 AND purpose = $2;
  `;
  await pool.query(query, [email, purpose]);
};

module.exports = {
  saveOTP,
  verifyOTP,
  markOTPAsUsed,
  deleteExistingOTPs,
};
