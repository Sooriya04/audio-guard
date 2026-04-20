const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const userQuery = require("../../functions/query/users");
const otpQuery = require("../../functions/query/otp");
const hashing = require("../../utils/hashing");
const { generateOTP } = require("../../utils/otp");
const { sendOTPEmail } = require("../../utils/mailer");

exports.signup = catchAsync(async(req, res, next)=>{
    const {
        email,
        password
    } = req.body;

    if(!email || !password){
        return next(new AppError("All fields are required", 400))
    }
    const existingEmail = await userQuery.existingEmail(email);

    if(existingEmail.length > 0){
        return next(new AppError('Email already exist', 409));
    }
    const hash_password = await hashing.hashPassword(password)
    const user = await userQuery.insertUser({
        email, hash_password
    })

    // --- Verification Flow ---
    const otp = generateOTP();
    const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    await otpQuery.saveOTP(email, otp, 'VERIFY_EMAIL', expiresAt);
    
    try {
        await sendOTPEmail(email, otp);
    } catch (mailError) {
        console.error("Failed to send initial welcome email:", mailError);
        // We still created the user, they can 'resend' later
    }

    res.status(201).json({
        success : true,
        message : "User created successfully. Please verify your email.",
        data : {
            id : user.id,
            email : user.email
        }
    })
})