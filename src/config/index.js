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
  allowedExts: constants.SECURITY.allowedExts,
  pyodideDataDir: process.env.PYODIDE_DATA_DIR || constants.PYODIDE.DEFAULT_PYODIDE_DATA_DIR,
  pyodideBases: (() => {
    try {
      const bases = JSON.parse(process.env.PYODIDE_BASES || '{}');
      // Merge with defaults
      return { ...constants.PYODIDE.DEFAULT_BASES, ...bases };
    } catch {
      return constants.PYODIDE.DEFAULT_BASES;
    }
  })(),
  dotenvLoaded,
};
module.exports = config;
