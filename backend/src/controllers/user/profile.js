const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const pool = require("../../config/db");

/**
 * Get complete normalized user profile
 */
exports.getProfile = catchAsync(async (req, res, next) => {
    const userId = req.user.id;
    
    // Join users and user_profiles to get all information
    const { rows } = await pool.query(
        `SELECT 
            u.id, u.email, u.email_verified AS is_verified, u.created_at as account_created,
            p.full_name, p.username, p.phone_number, p.country, p.profile_image,
            p.account_type, p.total_scans, p.fake_detected, 
            p.storage_used_mb, p.notifications_enabled, p.theme
         FROM users u
         LEFT JOIN user_profiles p ON u.id = p.user_id
         WHERE u.id = $1`,
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

/**
 * Update or Initialize user profile (Normalized Corporate Style)
 */
exports.updateProfile = catchAsync(async (req, res, next) => {
    const userId = req.user.id;
    const { 
        full_name, username, phone_number, country, 
        profile_image, notifications_enabled, theme 
    } = req.body;

    // UPSERT logic: Insert if not exists, otherwise update
    const query = `
        INSERT INTO user_profiles (
            user_id, full_name, username, phone_number, country, 
            profile_image, notifications_enabled, theme, updated_at
        ) 
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id) 
        DO UPDATE SET 
            full_name = EXCLUDED.full_name,
            username = EXCLUDED.username,
            phone_number = EXCLUDED.phone_number,
            country = EXCLUDED.country,
            profile_image = EXCLUDED.profile_image,
            notifications_enabled = EXCLUDED.notifications_enabled,
            theme = EXCLUDED.theme,
            updated_at = CURRENT_TIMESTAMP
        RETURNING *;
    `;

    const { rows } = await pool.query(query, [
        userId, full_name, username, phone_number, country, 
        profile_image, notifications_enabled, theme
    ]);

    res.status(200).json({
        success: true,
        message: "Profile updated successfully",
        data: rows[0]
    });
});
