const fs = require('fs');

exports.detectDeepfake = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: 'No audio file provided' });
        }

        const filePath = req.file.path;
        const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000/analyze';

        // Read the file as a Buffer
        const fileBuffer = fs.readFileSync(filePath);

        const formData = new FormData();
        const fileContent = new File([fileBuffer], req.file.originalname, { type: req.file.mimetype });
        formData.append('file', fileContent);

        // Call the Python Microservice using native fetch
        const response = await fetch(mlServiceUrl, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        // Cleanup: remove temporary upload file
        fs.unlinkSync(filePath);

        if (!response.ok) {
            return res.status(response.status).json({
                success: false,
                message: 'ML Service Error',
                error: data.error || 'Unknown error'
            });
        }

        return res.status(200).json(data);

    } catch (error) {
        console.error('Analysis error:', error);
        
        // Cleanup on error
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        return res.status(500).json({
            success: false,
            message: 'Internal server error during analysis',
            error: error.message
        });
    }
};
