const express = require("express");
const cors = require("cors");
const homeRoutes = require("./routes/home");

const app = express();

app.use(cors());
app.use(express.json());
app.use("/", homeRoutes);

module.exports = app;