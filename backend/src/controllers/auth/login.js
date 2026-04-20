const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const userQuery = require("../../functions/query/users");
const hashing = require("../../utils/hashing");
const tokens = require("../../utils/token");
const JwtSession = require("../../models/mongodb/JwtSession");


exports.login = catchAsync(async(req, res, next)=>{
    const {
        email,
        password
    } = req.body;

    if(!email || !password){
        return next(new AppError("All fields are required", 400));
    }

    const existingEmail = await userQuery.login(email);
    
    if(!existingEmail){
        return next(new AppError("Invaild credentials", 401));
    }

    const isMatch = await hashing.comparePassword(password, existingEmail.password);

    if(!isMatch){
        return next(new AppError("Invaild credentials", 401));
    }

    const accessSecret = tokens.generateAccessSecret();
    const refreshToken = tokens.generateRefreshToken();

    const accessToken = tokens.signAccessToken(
        {
            userId : existingEmail.id,
            email : existingEmail.email
        },
        accessSecret
    );

    await JwtSession.create({
        userId : existingEmail.id,
        email : existingEmail.email,
        accessToken : accessToken,
        accessSecret : accessSecret,
        refreshToken : refreshToken,
        expiresAt : new Date(Date.now() + 20 * 365 * 24 * 60 * 60 * 1000)
    });

    res.status(200)
    .json({
        success : true,
        message: "Logged in successfully",
        accessToken,
        refreshToken,
        email : existingEmail.email,
        is_verified: existingEmail.email_verified
    })
})