--------------------------------------------------
-- USERS TABLE (Core Authentication)
--------------------------------------------------
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--------------------------------------------------
-- USER PROFILES TABLE (Normalized Identity & Stats)
--------------------------------------------------
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    
    -- Basic Info
    full_name VARCHAR(255),
    username VARCHAR(100) UNIQUE,
    email VARCHAR(255), -- Denormalized email for faster profile lookups
    
    -- Contact
    phone_number VARCHAR(20),
    country VARCHAR(100),
    profile_image TEXT, -- Local path or URL
    
    -- System / App Related
    account_type VARCHAR(50) DEFAULT 'free',
    is_verified BOOLEAN DEFAULT FALSE,
    
    -- Usage Stats
    total_scans INTEGER DEFAULT 0,
    fake_detected INTEGER DEFAULT 0,
    storage_used_mb INTEGER DEFAULT 0,
    
    -- Preferences
    notifications_enabled BOOLEAN DEFAULT TRUE,
    theme VARCHAR(20) DEFAULT 'light',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--------------------------------------------------
-- OTP CODES TABLE
--------------------------------------------------
CREATE TABLE otp_codes (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    otp VARCHAR(6) NOT NULL,
    purpose VARCHAR(20) NOT NULL
        CHECK (purpose IN ('LOGIN', 'VERIFY_EMAIL')),
    expires_at TIMESTAMP NOT NULL,
    is_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_otp_email_purpose ON otp_codes(email, purpose);