const jwt = require("jsonwebtoken");
const crypto = require("crypto");

exports.generateAccessSecret = () => {
    crypto.randomBytes(64).toString("hex");
}

exports.generateRefreshToken = ()=> {
    crypto.randomBytes(64).toString("hex");
}

exports.signAccessToken = (payload, secret) => {
    jwt.sign(payload, secret , {
        expiresIn : "7d"
    });
}