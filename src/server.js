const app = require('./app');
const config = require('./config');
const pyodideService = require('./services/pyodide-service');
const logger = require('./utils/logger');

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
