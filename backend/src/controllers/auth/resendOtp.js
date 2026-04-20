const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const otpQuery = require("../../functions/query/otp");
const userQuery = require("../../functions/query/users");
const { generateOTP } = require("../../utils/otp");
const { sendOTPEmail } = require("../../utils/mailer");

exports.resendOtp = catchAsync(async (req, res, next) => {
  const { email } = req.body;

  if (!email) {
    return next(new AppError("Email is required", 400));
  }

  // Check if user exists
  const user = await userQuery.existingEmail(email);
  if (!user || user.length === 0) {
    return next(new AppError("User with this email does not exist", 404));
  }

  // Cleanup old OTPs for this email and purpose
  await otpQuery.deleteExistingOTPs(email, 'VERIFY_EMAIL');

  // Generate and send new OTP
  const otp = generateOTP();
  const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

  await otpQuery.saveOTP(email, otp, 'VERIFY_EMAIL', expiresAt);

  try {
    await sendOTPEmail(email, otp);
  } catch (mailError) {
    return next(new AppError("Failed to send verification email. Please try again later.", 500));
  }

  res.status(200).json({
    success: true,
    message: "New verification code sent to your email",
  });
});
