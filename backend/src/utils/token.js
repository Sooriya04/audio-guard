const jwt = require("jsonwebtoken");
const crypto = require("crypto");

exports.generateAccessSecret = () => {
    return crypto.randomBytes(64).toString("hex");
}

exports.generateRefreshToken = ()=> {
    return crypto.randomBytes(64).toString("hex");
}

exports.signAccessToken = (payload, secret) => {
    return jwt.sign(payload, secret , {
        expiresIn : "7d"
    });
}