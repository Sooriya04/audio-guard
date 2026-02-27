--------------------------------------------------
-- USERS TABLE
--------------------------------------------------
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
CREATE INDEX idx_otp_email_purpose ON otp_codes(email, purpose);