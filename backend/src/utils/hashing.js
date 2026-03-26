const bcrypt = require("bcrypt");

exports.hashPassword = async(password) => {
    return await bcrypt.hash(password, 12);
}

exports.comparePassword = async(hashed_password, password) => {
    return await bcrypt.compare(hashed_password, password)
}