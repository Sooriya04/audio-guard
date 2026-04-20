const { Pool } = require("pg");
const { POSTGRESQL } = require("./env");

const pool = new Pool({ connectionString: POSTGRESQL });

pool.on("connect", () => 
    console.log("PostgreSQL connected")
);

// DEBUG 
// console.log("POSTGRESQL =", process.env.POSTGRESQL);
// (async () => {
//   try {
//     const client = await pool.connect();
//     console.log("PostgreSQL connected");
//     client.release();
//   } catch (err) {
//     console.error("PostgreSQL connection failed:", err.message);
//   }
// })();

module.exports = pool;