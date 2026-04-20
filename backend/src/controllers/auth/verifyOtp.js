const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const otpQuery = require("../../functions/query/otp");
const pool = require("../../config/db");

exports.verifyOtp = catchAsync(async (req, res, next) => {
  const { email, otp } = req.body;

  if (!email || !otp) {
    return next(new AppError("Email and OTP are required", 400));
  }

  const otpRecord = await otpQuery.verifyOTP(email, otp, 'VERIFY_EMAIL');

  if (!otpRecord) {
    return next(new AppError("Invalid or expired verification code", 400));
  }

  // Mark OTP as used and update user verification status
  await otpQuery.markOTPAsUsed(otpRecord.id);

  const updateQuery = `
    UPDATE users 
    SET email_verified = TRUE 
    WHERE email = $1;
  `;
  await pool.query(updateQuery, [email]);

  // Also update user profile if it exists
  const updateProfileQuery = `
    UPDATE user_profiles
    SET is_verified = TRUE
    WHERE email = $1;
  `;
  await pool.query(updateProfileQuery, [email]);

  res.status(200).json({
    success: true,
    message: "Email successfully verified",
  });
});
