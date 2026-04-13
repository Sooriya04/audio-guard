const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const userQuery = require("../../functions/query/users");
const hashing = require("../../utils/hashing");

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

    res.status(201).json({
        success : true,
        message : "User created successfully",
        data : {
            id : user.id,
            email : user.email
        }
    })
})