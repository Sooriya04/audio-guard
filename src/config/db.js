const { Pool } = require("pg");
const { DATABASE_URL } = require("./env");

const pool = new Pool({ connectionString: DATABASE_URL });

pool.on("connect", () => console.log("PostgreSQL connected"));

module.exports = pool;