const pool = require("../../config/db");

exports.existingEmail = async (email) => {
    const {
        rows 
    } = 
        await pool.query(
            `SELECT id FROM users WHERE email = $1`,
            [email]
        )
    return rows;
}

exports.insertUser = async({email, hash_password}) => {
    const {
        rows 
    } = await pool.query(
        `INSERT INTO users (email, password)
         VALUES($1, $2) 
         RETURNING id, email, created_at
        `,
        [email, hash_password]
    );
    return rows[0];
}

exports.login = async(email) => {
    const { rows } = await pool.query(
        `
        SELECT id, email, password FROM users WHERE email = $1
        `,
        [email]
    )
    return rows[0];
}