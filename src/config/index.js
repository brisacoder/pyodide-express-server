let dotenvLoaded = false;
try {
  require('dotenv').config();
  dotenvLoaded = true;
} catch {
  // dotenv is optional; ignore if not installed
}

const constants = require('./constants');

const config = {
  port: process.env.PORT || constants.SERVER.DEFAULT_PORT,
  maxFileSize: parseInt(process.env.MAX_FILE_SIZE, 10) || constants.NETWORK.MAX_UPLOAD_SIZE,
  uploadDir: process.env.UPLOAD_DIR || constants.NETWORK.DEFAULT_UPLOAD_DIR,
  corsOrigin: process.env.CORS_ORIGIN || constants.NETWORK.DEFAULT_CORS_ORIGIN,
  dotenvLoaded,
};

module.exports = config;
