const bcrypt = require("bcrypt");

exports.hashPassword = async(password) => {
    return await bcrypt.hash(password, 12);
}

exports.comparePassword = async(plaintextPassword, hashedPassword) => {
    return await bcrypt.compare(plaintextPassword, hashedPassword);
}