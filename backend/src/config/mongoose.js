const mongoose = require("mongoose");
const { MONGODB } = require("./env")
mongoose
    .connect(MONGODB)
    .then(()=>{
        console.log("MONGODB is connected")
    }).catch((e)=>{
        console.error("MONGODB connection failed", e.message)
    })


