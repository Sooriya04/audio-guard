const express = require('express');
const router = express.Router();
const multer = require('multer');
const analysisController = require('../controllers/analysisController');

// Multer setup for handling audio file upload
const upload = multer({ dest: 'uploads/' });

router.post('/detect', upload.single('audio'), analysisController.detectDeepfake);

module.exports = router;
