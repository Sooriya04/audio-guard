const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const pool = require("../../config/db");

exports.getProfile = catchAsync(async (req, res, next) => {
    const userId = req.user.id; // Added by authMiddleware
    
    const { rows } = await pool.query(
        `SELECT id, email, created_at FROM users WHERE id = $1`,
        [userId]
    );

    if (rows.length === 0) {
        return next(new AppError("User not found", 404));
    }

    res.status(200).json({
        success: true,
        data: rows[0]
    });
});
