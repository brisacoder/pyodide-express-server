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
  processPool: {
    poolSize: parseInt(process.env.PYODIDE_POOL_SIZE, 10) || constants.PROCESS_POOL.DEFAULT_POOL_SIZE,
    maxExecutions: parseInt(process.env.PYODIDE_MAX_EXECUTIONS, 10) || constants.PROCESS_POOL.MAX_EXECUTIONS_PER_PROCESS,
    processInitTimeout: parseInt(process.env.PYODIDE_INIT_TIMEOUT, 10) || constants.PROCESS_POOL.PROCESS_INIT_TIMEOUT,
    idleTimeout: parseInt(process.env.PYODIDE_IDLE_TIMEOUT, 10) || constants.PROCESS_POOL.PROCESS_IDLE_TIMEOUT,
    healthCheckInterval: parseInt(process.env.PYODIDE_HEALTH_CHECK_INTERVAL, 10) || constants.PROCESS_POOL.POOL_HEALTH_CHECK_INTERVAL,
  },
  dotenvLoaded,
};
module.exports = config;
