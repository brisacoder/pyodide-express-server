/**
 * Pyodide Express Server - Main Server Entry Point
 * 
 * Orchestrates server startup, Pyodide initialization, and graceful shutdown.
 * Provides production-ready server lifecycle management.
 */

const app = require('./app');
const config = require('./config');
const pyodideService = require('./services/pyodide-service');
const logger = require('./utils/logger');
const CrashReporter = require('./utils/crashReporter');

// Initialize crash reporter
const crashReporter = new CrashReporter({
    crashDir: './crash-reports',
    maxCrashFiles: 50,
    includeEnvironment: true,
    includeMemoryInfo: true,
    includePyodideState: true
});

// Make pyodideService globally accessible for crash reporting
global.pyodideService = pyodideService;

/**
 * Starts the Express server with Pyodide initialization and signal handling.
 * Handles the complete server startup sequence including dependency initialization.
 * 
 * @returns {Promise<Object>} Promise resolving to the HTTP server instance
 * 
 * @example
 * // Direct server startup
 * const { startServer } = require('./server');
 * startServer()
 *   .then(server => console.log('Server started:', server.address()))
 *   .catch(error => console.error('Startup failed:', error));
 * 
 * // In testing environment
 * const server = await startServer();
 * // ... run tests
 * server.close();
 * 
 * @description
 * Startup Sequence:
 * 1. Logs startup initiation
 * 2. Initializes Pyodide WebAssembly runtime
 * 3. Starts Express HTTP server on configured port
 * 4. Logs server information and endpoints
 * 5. Sets up graceful shutdown signal handlers
 * 6. Returns server instance for external control
 * 
 * Signal Handling:
 * - SIGTERM: Graceful shutdown (production deployment)
 * - SIGINT: Graceful shutdown (Ctrl+C in development)
 * - Both signals close server and exit with code 0
 * 
 * Error Handling:
 * - Logs startup failures with full error details
 * - Exits with code 1 on initialization failure
 * - Prevents hanging processes on startup errors
 * 
 * Logging Output:
 * - Server port and URLs
 * - API documentation links
 * - Log file locations
 * - Health check endpoints
 * - File management interfaces
 * 
 * Production Considerations:
 * - Supports process managers (PM2, systemd)
 * - Handles container orchestration signals
 * - Provides clear startup/shutdown logging
 * - Enables monitoring and health checks
 * 
 * @throws {Error} Initialization errors (Pyodide, port binding, etc.)
 */
async function startServer() {
  try {
    logger.info('Starting Pyodide Express Server...');
    logger.info('Initializing Pyodide...');
    await pyodideService.initialize();
    logger.info('Pyodide initialization completed!');

    const server = app.listen(config.port, () => {
      const logInfo = logger.getLogInfo();
      logger.info(`🚀 Server running on port ${config.port}`);
      logger.info(`📖 Web interface: http://localhost:${config.port}`);
      logger.info(`📚 API Documentation: http://localhost:${config.port}/docs`);
      logger.info(`🔧 API base URL: http://localhost:${config.port}/api`);
      logger.info(`📊 Health check: http://localhost:${config.port}/health`);
      logger.info(`📁 File management: http://localhost:${config.port}/api/uploaded-files`);

      if (logInfo.isFileLoggingEnabled) {
        logger.info(`📝 Logs writing to: ${logInfo.logFile}`);
        logger.info(`📁 Log directory: ${logInfo.logDirectory}`);
      } else {
        logger.info('📺 Console-only logging (no file output)');
      }
    });

    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      logger.info('SIGINT received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    return server;
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  startServer();
}

module.exports = { app, startServer };
