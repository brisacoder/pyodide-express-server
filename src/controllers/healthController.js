const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');

function health(req, res) {
  const status = pyodideService.getStatus();
  const logInfo = logger.getLogInfo();

  res.json({
    status: 'ok',
    server: 'running',
    pyodide: status,
    logging: logInfo,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
  });
}

module.exports = { health };
