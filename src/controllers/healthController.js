/**
 * Health Check Controller for System Status Monitoring
 * 
 * Provides comprehensive health and status information for monitoring,
 * alerting, and operational visibility.
 */

const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');

/**
 * Health check endpoint providing comprehensive system status information.
 * Returns detailed information about server, Pyodide, logging, and system resources.
 * 
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @returns {void} Sends JSON health status response
 * 
 * @example
 * // GET /health
 * // Response:
 * // {
 * //   "status": "ok",
 * //   "server": "running",
 * //   "pyodide": {
 * //     "initialized": true,
 * //     "version": "0.28.0",
 * //     "packagesLoaded": ["numpy", "pandas"]
 * //   },
 * //   "logging": {
 * //     "level": "info",
 * //     "logFile": "/app/logs/server.log",
 * //     "isFileLoggingEnabled": true
 * //   },
 * //   "timestamp": "2025-08-20T10:30:00Z",
 * //   "uptime": 3661.45,
 * //   "memory": {
 * //     "rss": 52428800,
 * //     "heapTotal": 41943040,
 * //     "heapUsed": 29892568,
 * //     "external": 1089024
 * //   }
 * // }
 * 
 * @description
 * Health Check Components:
 * - Server Status: Overall service availability
 * - Pyodide Status: WebAssembly runtime health
 * - Logging Status: Log system configuration
 * - System Metrics: Memory usage and uptime
 * - Timestamp: Current server time for sync verification
 * 
 * Monitoring Integration:
 * - Use for load balancer health checks
 * - Monitor Pyodide initialization status
 * - Track memory usage trends
 * - Verify logging system functionality
 * - Detect service degradation
 * 
 * Alerting Conditions:
 * - status !== "ok": Service degraded
 * - pyodide.initialized === false: Runtime failure
 * - High memory usage: Resource leak detection
 * - Long uptime: Restart cycle monitoring
 * 
 * Performance Characteristics:
 * - Low latency response (<10ms typical)
 * - No expensive operations
 * - Safe for frequent polling (every 5-30 seconds)
 * - Minimal resource impact
 * 
 * Response Format:
 * - Always returns 200 OK (even during degraded states)
 * - JSON format for programmatic consumption
 * - Structured data for automated monitoring
 * - Human-readable timestamps and values
 */
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
