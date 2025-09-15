/**
 * Health Check Controller for System Status Monitoring
 *
 * Provides comprehensive health and status information for monitoring,
 * alerting, and operational visibility with AWS deployment support.
 */
const os = require('os');
const fs = require('fs').promises;
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
/**
 * Basic health check endpoint for load balancers and simple monitoring.
 * Returns minimal status information for quick availability checks.
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
const health = async (req, res) => {
  try {
    const healthStatus = await getBasicHealthStatus();
    res.status(200).json({
      success: true,
      data: healthStatus,
      error: null,
      meta: {
        timestamp: healthStatus.timestamp
      }
    });
  } catch (error) {
    logger.error('Health check failed', { error: error.message, stack: error.stack });
    res.status(503).json({
      success: false,
      data: null,
      error: 'Service temporarily unavailable',
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
};
/**
 * Detailed health check endpoint for comprehensive monitoring.
 * Returns extensive system information for operational visibility.
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
const healthcheck = async (req, res) => {
  try {
    const healthStatus = await getDetailedHealthStatus();
    res.status(200).json({
      success: true,
      data: healthStatus,
      error: null,
      meta: {
        timestamp: healthStatus.timestamp
      }
    });
  } catch (error) {
    logger.error('Detailed health check failed', { error: error.message, stack: error.stack });
    res.status(503).json({
      success: false,
      data: null,
      error: `Service health check failed: ${error.message}`,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
};
/**
 * Kubernetes-style readiness probe.
 * Returns 200 when service is ready to receive traffic.
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 */
const readiness = async (req, res) => {
  try {
    const isReady = await checkServiceReadiness();
    const timestamp = new Date().toISOString();
    if (isReady) {
      res.status(200).json({
        success: true,
        data: {
          status: 'ready'
        },
        error: null,
        meta: {
          timestamp: timestamp
        }
      });
    } else {
      res.status(503).json({
        success: false,
        data: {
          status: 'not_ready'
        },
        error: 'Service is not ready to accept traffic',
        meta: {
          timestamp: timestamp
        }
      });
    }
  } catch (error) {
    logger.error('Readiness check failed', { error: error.message });
    res.status(503).json({
      success: false,
      data: {
        status: 'not_ready'
      },
      error: error.message,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
};
/**
 * Get basic health status for load balancer checks
 * @returns {Promise<Object>} Basic health status object
 */
async function getBasicHealthStatus() {
  const startTime = Date.now();
  // Test Pyodide availability
  const pyodideTest = await testPyodideBasic();
  return {
    status: pyodideTest.success ? 'ok' : 'degraded',
    server: 'running',
    pyodide: pyodideTest.success ? 'available' : 'unavailable',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    responseTime: Date.now() - startTime,
  };
}
/**
 * Get detailed health status for comprehensive monitoring
 * @returns {Promise<Object>} Detailed health status object
 */
async function getDetailedHealthStatus() {
  const startTime = Date.now();
  // Gather system information
  const [pyodideStatus, systemInfo, diskInfo, directoryInfo] = await Promise.allSettled([
    getPyodideStatus(),
    getSystemInfo(),
    getDiskUsage(),
    getDirectoryStatus(),
  ]);
  return {
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    server: {
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
      pid: process.pid,
      environment: process.env.NODE_ENV || 'development',
    },
    pyodide:
      pyodideStatus.status === 'fulfilled'
        ? pyodideStatus.value
        : { error: pyodideStatus.reason?.message },
    system:
      systemInfo.status === 'fulfilled' ? systemInfo.value : { error: systemInfo.reason?.message },
    disk: diskInfo.status === 'fulfilled' ? diskInfo.value : { error: diskInfo.reason?.message },
    directories:
      directoryInfo.status === 'fulfilled'
        ? directoryInfo.value
        : { error: directoryInfo.reason?.message },
    memory: process.memoryUsage(),
    logging: {
      level: process.env.LOG_LEVEL || 'info',
      logFile: process.env.LOG_FILE_PATH || './logs/server.log',
      isFileLoggingEnabled: true,
    },
    responseTime: Date.now() - startTime,
  };
}
/**
 * Check if service is ready to receive traffic
 * @returns {Promise<boolean>} True if service is ready, false otherwise
 */
async function checkServiceReadiness() {
  try {
    // Check if Pyodide can execute basic Python
    const pyodideTest = await testPyodideBasic();
    if (!pyodideTest.success) {
      return false;
    }
    // Check if required directories exist
    const requiredDirs = ['./uploads', './logs', './plots'];
    for (const dir of requiredDirs) {
      try {
        await fs.access(dir);
      } catch {
        logger.warn(`Required directory missing: ${dir}`);
        return false;
      }
    }
    return true;
  } catch (error) {
    logger.error('Readiness check failed', { error: error.message });
    return false;
  }
}
/**
 * Test basic Pyodide functionality
 * @returns {Promise<Object>} Test result object with success status
 */
async function testPyodideBasic() {
  try {
    if (!pyodideService.isInitialized()) {
      return { success: false, error: 'Pyodide not initialized' };
    }
    // Simple Python execution test
    const executionResult = await pyodideService.executeCode('2 + 2', {}, 5000);
    // Check if execution was successful and result is correct
    const isSuccessful =
      executionResult.success && (executionResult.result === 4 || executionResult.result === '4');
    return {
      success: isSuccessful,
      result: executionResult,
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
    };
  }
}
/**
 * Get Pyodide status information
 * @returns {Promise<Object>} Pyodide status and diagnostics
 */
async function getPyodideStatus() {
  try {
    const isInitialized = pyodideService.isInitialized();
    let version = 'unknown';
    let packagesLoaded = [];
    let testResult = null;
    if (isInitialized) {
      try {
        // Get Pyodide version
        version = await pyodideService.executeCode('import sys; sys.version', 3000);
        // Get loaded packages
        const packageInfo = await pyodideService.executeCode(
          `
import sys
import json
packages = [name for name in sys.modules.keys() if not name.startswith('_')]
json.dumps(sorted(packages[:20]))  # First 20 packages
        `,
          5000
        );
        try {
          packagesLoaded = JSON.parse(packageInfo);
        } catch {
          packagesLoaded = ['Unable to parse package list'];
        }
        // Test execution
        testResult = await testPyodideBasic();
      } catch (error) {
        testResult = { success: false, error: error.message };
      }
    }
    return {
      initialized: isInitialized,
      version: version,
      packagesLoaded: packagesLoaded,
      test: testResult,
    };
  } catch (error) {
    return {
      initialized: false,
      error: error.message,
    };
  }
}
/**
 * Get system information
 * @returns {Object} System information object
 */
function getSystemInfo() {
  return {
    hostname: os.hostname(),
    platform: os.platform(),
    architecture: os.arch(),
    cpus: os.cpus().length,
    totalMemory: os.totalmem(),
    freeMemory: os.freemem(),
    loadAverage: os.loadavg(),
    uptime: os.uptime(),
  };
}
/**
 * Get disk usage information (simplified for cross-platform)
 * @returns {Promise<Object>} Disk usage information
 */
async function getDiskUsage() {
  try {
    await fs.stat('./');
    return {
      available: 'N/A (Cross-platform limitation)',
      note: 'Use external monitoring for disk space',
    };
  } catch (error) {
    return {
      error: error.message,
    };
  }
}
/**
 * Check status of required directories
 * @returns {Promise<Object>} Directory status information
 */
async function getDirectoryStatus() {
  const directories = ['./uploads', './logs', './plots', './plots/matplotlib', './plots/seaborn'];
  const status = {};
  for (const dir of directories) {
    try {
      const stats = await fs.stat(dir);
      const files = await fs.readdir(dir);
      status[dir] = {
        exists: true,
        isDirectory: stats.isDirectory(),
        fileCount: files.length,
        size: stats.size,
      };
    } catch (error) {
      status[dir] = {
        exists: false,
        error: error.code,
      };
    }
  }
  return status;
}
module.exports = { health, healthcheck, readiness };
