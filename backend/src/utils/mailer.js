const nodemailer = require("nodemailer");
const { MAILER_USER, MAILER_PASS, MAILER_HOST, MAILER_PORT } = require("../config/env");

const transporter = nodemailer.createTransport({
  host: MAILER_HOST || "smtp.gmail.com",
  port: MAILER_PORT || 587,
  secure: false, // true for 465, false for other ports
  auth: {
    user: MAILER_USER,
    pass: MAILER_PASS,
  },
});

/**
 * Sends a verification OTP email
 * @param {string} email Target email address
 * @param {string} otp 6-digit OTP
 */
const sendOTPEmail = async (email, otp) => {
  const mailOptions = {
    from: `"VoiceSafe Security" <${MAILER_USER}>`,
    to: email,
    subject: "Verify Your VoiceSafe Account",
    html: `
      <div style="font-family: sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;">
        <h2 style="color: #4f46e5; text-align: center;">VoiceSafe</h2>
        <p>Hello,</p>
        <p>Your verification code for VoiceSafe is:</p>
        <div style="background: #f1f5f9; padding: 20px; text-align: center; border-radius: 8px; margin: 24px 0;">
          <h1 style="letter-spacing: 8px; color: #0f172a; margin: 0;">${otp}</h1>
        </div>
        <p>This code is used for account verification and will expire in 10 minutes.</p>
        <p>If you didn't request this code, please ignore this email.</p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />
        <p style="color: #64748b; font-size: 12px; text-align: center;">&copy; 2026 VoiceSafe Security. All rights reserved.</p>
      </div>
    `,
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log(`OTP sent to ${email}`);
  } catch (error) {
    console.error("Error sending email:", error);
    throw new Error("Failed to send verification email");
  }
};

module.exports = { sendOTPEmail };
