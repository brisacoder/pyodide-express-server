let dotenvLoaded = false;
try {
  require('dotenv').config();
  dotenvLoaded = true;
} catch (err) {
  // dotenv is optional; ignore if not installed
}

const config = {
  port: process.env.PORT || 3000,
  maxFileSize: parseInt(process.env.MAX_FILE_SIZE, 10) || 10 * 1024 * 1024,
  uploadDir: process.env.UPLOAD_DIR || 'uploads',
  corsOrigin: process.env.CORS_ORIGIN || '*',
  dotenvLoaded,
};

module.exports = config;
