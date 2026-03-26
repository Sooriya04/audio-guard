// Custom in-memory rate limiter to bypass npm issues
const createRateLimiter = (options) => {
    const hits = new Map();
    
    return (req, res, next) => {
        const ip = req.ip || req.connection.remoteAddress;
        const now = Date.now();
        
        let record = hits.get(ip);
        if (!record) {
            record = { count: 1, resetTime: now + options.windowMs };
            hits.set(ip, record);
        } else {
            if (now > record.resetTime) {
                record.count = 1;
                record.resetTime = now + options.windowMs;
            } else {
                record.count++;
            }
        }

        if (record.count > options.limit) {
            return res.status(429).json({
                success: false,
                message: options.message?.message || "Too many requests"
            });
        }
        
        // Cleanup old entries randomly to prevent memory leak
        if (Math.random() < 0.01) {
            const cleanupTime = Date.now();
            for (const [key, value] of hits.entries()) {
                if (cleanupTime > value.resetTime) hits.delete(key);
            }
        }
        
        next();
    };
};

// General API Rate Limiter
exports.apiLimiter = createRateLimiter({
    windowMs: 15 * 60 * 1000, // 15 minutes
    limit: 100,
    message: { message: "Too many requests from this IP, please try again after 15 minutes" }
});

// Stricter Auth Rate Limiter
exports.authLimiter = createRateLimiter({
    windowMs: 60 * 60 * 1000, // 1 hour
    limit: 10,
    message: { message: "Too many authentication attempts from this IP, please try again after an hour" }
});
