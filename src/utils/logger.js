/**
 * Structured JSON logger with basic rotation, request ID support, and security logging.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { getRequestId } = require('./requestContext');

const levels = { error: 0, warn: 1, info: 2, debug: 3 };
const currentLevel = levels[(process.env.LOG_LEVEL || 'info').toLowerCase()] ?? levels.info;

const logDir = process.env.LOG_DIR || path.join(__dirname, '../../logs');
const logFile = process.env.LOG_FILE || path.join(logDir, 'server.log');
const securityLogFile = path.join(logDir, 'security.log');
const maxSize = parseInt(process.env.LOG_MAX_SIZE || 5 * 1024 * 1024, 10); // 5MB default

// In-memory statistics storage (use Redis/DB for production)
const executionStats = {
  totalExecutions: 0,
  successCount: 0,
  errorCount: 0,
  totalExecutionTime: 0,
  ipCounts: new Map(),
  codePatterns: new Map(),
  packageInstalls: 0,
  fileUploads: 0,
  lastHourExecutions: [],
  topErrors: new Map(),
  userAgents: new Map(),
  startTime: Date.now()
};

if (logFile) {
  const dir = path.dirname(logFile);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Rotates log file if it exceeds the maximum size limit.
 * Creates a timestamped backup and starts fresh with the original filename.
 * 
 * @param {string} [fileName=logFile] - Path to the log file to check for rotation
 * @returns {void}
 * 
 * @example
 * // Rotate default log file if needed
 * rotateIfNeeded();
 * 
 * // Rotate specific log file
 * rotateIfNeeded('/path/to/custom.log');
 * 
 * @description
 * - Checks if file exists and its size
 * - If size >= maxSize, renames file with timestamp suffix
 * - Next write will create a new file with original name
 */
function rotateIfNeeded(fileName = logFile) {
  if (!fileName || !fs.existsSync(fileName)) return;
  const { size } = fs.statSync(fileName);
  if (size < maxSize) return;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rotated = `${fileName}.${timestamp}`;
  fs.renameSync(fileName, rotated);
}

/**
 * Safely writes content to a log file with automatic rotation.
 * Handles file creation, rotation checking, and error recovery.
 * 
 * @param {string} fileName - Path to the log file to write to
 * @param {string} content - Content to append to the file
 * @returns {void}
 * 
 * @example
 * // Write to default log file
 * writeToFile(logFile, '{"level":"info","message":"Server started"}');
 * 
 * // Write to security log
 * writeToFile(securityLogFile, '{"level":"security","event":"login"}');
 * 
 * @description
 * - Checks for rotation before writing
 * - Appends content with newline
 * - Gracefully handles write errors without crashing
 */
function writeToFile(fileName, content) {
  if (!fileName) return;
  try {
    rotateIfNeeded(fileName);
    fs.appendFileSync(fileName, content + '\n');
  } catch (err) {
    console.error(`Failed to write to ${fileName}:`, err.message);
  }
}

/**
 * Core logging function that formats and outputs log entries.
 * Handles console output, file writing, and request ID injection.
 * 
 * @param {string} level - Log level (error, warn, info, debug)
 * @param {number} levelNum - Numeric level for filtering
 * @param {string} message - Primary log message
 * @param {Object} [meta={}] - Additional metadata to include
 * @returns {void}
 * 
 * @example
 * // Basic logging
 * write('info', levels.info, 'Server started');
 * 
 * // Logging with metadata
 * write('error', levels.error, 'Database connection failed', {
 *   host: 'localhost',
 *   port: 5432,
 *   error: 'ECONNREFUSED'
 * });
 * 
 * @description
 * - Respects current log level filtering
 * - Automatically adds timestamp and request ID
 * - Outputs to both console and file
 * - Uses appropriate console method per level
 */
function write(level, levelNum, message, meta = {}) {
  if (levelNum > currentLevel) return;
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    ...meta,
  };
  const requestId = getRequestId();
  if (requestId) entry.requestId = requestId;
  const line = JSON.stringify(entry);

  switch (level) {
    case 'error':
      console.error(line);
      break;
    case 'warn':
      console.warn(line);
      break;
    case 'info':
      console.info(line);
      break;
    case 'debug':
      console.debug(line);
      break;
    default:
      console.log(line);
      break;
  }

  writeToFile(logFile, line);
}

/**
 * Updates execution statistics with data from code execution events.
 * Tracks performance metrics, error patterns, and usage analytics.
 * 
 * @param {Object} data - Execution data object
 * @param {string} [data.ip] - Client IP address
 * @param {boolean} data.success - Whether execution was successful
 * @param {number} [data.executionTime] - Execution time in milliseconds
 * @param {string} [data.error] - Error message if execution failed
 * @param {string} [data.userAgent] - Client user agent string
 * @param {string} [data.codeHash] - SHA-256 hash of executed code
 * @returns {void}
 * 
 * @example
 * // Track successful execution
 * updateExecutionStats({
 *   ip: '192.168.1.100',
 *   success: true,
 *   executionTime: 150,
 *   userAgent: 'Mozilla/5.0...',
 *   codeHash: 'a1b2c3d4...'
 * });
 * 
 * // Track failed execution
 * updateExecutionStats({
 *   ip: '192.168.1.101',
 *   success: false,
 *   error: 'SyntaxError: invalid syntax',
 *   userAgent: 'curl/7.68.0'
 * });
 * 
 * @description
 * - Updates counters (total, success, error)
 * - Tracks performance metrics
 * - Maintains IP and user agent statistics
 * - Extracts and categorizes error types
 * - Keeps rolling window of recent executions
 * - Generates code pattern analytics
 */
function updateExecutionStats(data) {
  const { ip, success, executionTime, error, userAgent, codeHash } = data;
  
  // Update basic counters
  executionStats.totalExecutions++;
  if (success) {
    executionStats.successCount++;
  } else {
    executionStats.errorCount++;
  }
  
  // Track execution time
  if (executionTime) {
    executionStats.totalExecutionTime += executionTime;
  }
  
  // Track IPs
  if (ip) {
    executionStats.ipCounts.set(ip, (executionStats.ipCounts.get(ip) || 0) + 1);
  }
  
  // Track User Agents
  if (userAgent) {
    const agent = userAgent.split(' ')[0]; // First part of user agent
    executionStats.userAgents.set(agent, (executionStats.userAgents.get(agent) || 0) + 1);
  }
  
  // Track errors
  if (error && !success) {
    // Better error type extraction - get just the error class name
    let errorType = error;
    if (error.includes('Error:')) {
      errorType = error.split('Error:')[0] + 'Error';
    } else if (error.includes('Exception:')) {
      errorType = error.split('Exception:')[0] + 'Exception';
    } else if (error.includes('(')) {
      // Handle cases like "invalid syntax (<string>, line 1)"
      errorType = error.split('(')[0].trim();
    }
    executionStats.topErrors.set(errorType, (executionStats.topErrors.get(errorType) || 0) + 1);
  }
  
  // Track recent executions (last hour)
  const now = Date.now();
  executionStats.lastHourExecutions.push({
    timestamp: now,
    success,
    executionTime: executionTime || 0,
    ip
  });
  
  // Clean old entries (keep only last hour)
  const oneHourAgo = now - (60 * 60 * 1000);
  executionStats.lastHourExecutions = executionStats.lastHourExecutions.filter(
    exec => exec.timestamp > oneHourAgo
  );
  
  // Track code patterns (first few characters for pattern detection)
  if (codeHash) {
    const pattern = codeHash.substring(0, 8); // First 8 chars of hash
    executionStats.codePatterns.set(pattern, (executionStats.codePatterns.get(pattern) || 0) + 1);
  }
}

const logger = {
  error(message, meta) {
    write('error', levels.error, message, meta);
  },
  warn(message, meta) {
    write('warn', levels.warn, message, meta);
  },
  info(message, meta) {
    write('info', levels.info, message, meta);
  },
  debug(message, meta) {
    write('debug', levels.debug, message, meta);
  },
  
  /**
   * Enhanced security logging for audit trails and threat detection.
   * Logs security-relevant events to both console and dedicated security log.
   * 
   * @param {string} eventType - Type of security event (code_execution, package_install, file_upload, etc.)
   * @param {Object} data - Event-specific data object
   * @param {string} [data.ip] - Client IP address
   * @param {string} [data.userAgent] - Client user agent
   * @param {boolean} [data.success] - Whether operation was successful
   * @param {string} [data.error] - Error message if operation failed
   * @param {number} [data.executionTime] - Time taken for operation
   * @param {string} [data.codeHash] - SHA-256 hash of code for code_execution events
   * @returns {void}
   * 
   * @example
   * // Log code execution
   * logger.security('code_execution', {
   *   ip: '192.168.1.100',
   *   success: true,
   *   executionTime: 250,
   *   codeHash: 'a1b2c3d4e5f6...',
   *   userAgent: 'Mozilla/5.0...'
   * });
   * 
   * // Log package installation
   * logger.security('package_install', {
   *   package: 'numpy',
   *   ip: '10.0.0.5',
   *   success: true
   * });
   * 
   * // Log suspicious activity
   * logger.security('suspicious_code', {
   *   ip: '203.0.113.1',
   *   reason: 'file_system_access_attempt',
   *   code_snippet: 'import os; os.system(...)'
   * });
   * 
   * @description
   * - Creates structured security log entries
   * - Automatically adds timestamp and request ID
   * - Updates relevant statistics based on event type
   * - Outputs to both console and security.log file
   * - Enables security monitoring and alerting
   */
  security(eventType, data) {
    const securityEntry = {
      timestamp: new Date().toISOString(),
      level: 'security',
      eventType,
      ...data,
      requestId: getRequestId()
    };
    
    const line = JSON.stringify(securityEntry);
    console.info(`[SECURITY] ${line}`);
    writeToFile(securityLogFile, line);
    
    // Update statistics for specific event types
    if (eventType === 'code_execution') {
      updateExecutionStats(data);
    } else if (eventType === 'package_install') {
      executionStats.packageInstalls++;
    } else if (eventType === 'file_upload') {
      executionStats.fileUploads++;
    }
  },
  
  /**
   * Retrieves comprehensive execution statistics and analytics.
   * Provides real-time metrics for monitoring dashboard and system health.
   * 
   * @returns {Object} Statistics object with multiple categories
   * @returns {Object} returns.overview - High-level statistics
   * @returns {number} returns.overview.totalExecutions - Total code executions since start
   * @returns {string} returns.overview.successRate - Success rate as percentage string
   * @returns {number} returns.overview.averageExecutionTime - Average execution time in ms
   * @returns {number} returns.overview.uptimeSeconds - Server uptime in seconds
   * @returns {string} returns.overview.uptimeHuman - Human-readable uptime
   * @returns {Object} returns.recent - Recent activity metrics
   * @returns {number} returns.recent.lastHourExecutions - Executions in last hour
   * @returns {string} returns.recent.recentSuccessRate - Recent success rate percentage
   * @returns {number} returns.recent.packagesInstalled - Total packages installed
   * @returns {number} returns.recent.filesUploaded - Total files uploaded
   * @returns {Array<Object>} returns.topIPs - Most active IP addresses
   * @returns {Array<Object>} returns.topErrors - Most common error types
   * @returns {Array<Object>} returns.userAgents - Most common user agents
   * @returns {Array<number>} returns.hourlyTrend - 24-hour execution trend data
   * 
   * @example
   * const stats = logger.getStats();
   * console.log(`Server uptime: ${stats.overview.uptimeHuman}`);
   * console.log(`Success rate: ${stats.overview.successRate}%`);
   * console.log(`Top error: ${stats.topErrors[0]?.error}`);
   * 
   * // Use for dashboard
   * res.json({
   *   dashboard: stats,
   *   timestamp: new Date().toISOString()
   * });
   * 
   * @description
   * - Calculates real-time statistics from in-memory data
   * - Provides both raw numbers and human-readable formats
   * - Generates trending data for charts
   * - Safe for frequent polling (no expensive operations)
   * - Returns consistent structure for API responses
   */
  getStats() {
    const now = Date.now();
    const uptimeSeconds = Math.floor((now - executionStats.startTime) / 1000);
    const recentExecutions = executionStats.lastHourExecutions;
    
    return {
      overview: {
        totalExecutions: executionStats.totalExecutions,
        successRate: executionStats.totalExecutions > 0 
          ? ((executionStats.successCount / executionStats.totalExecutions) * 100).toFixed(1)
          : 0,
        averageExecutionTime: executionStats.totalExecutions > 0
          ? Math.round(executionStats.totalExecutionTime / executionStats.totalExecutions)
          : 0,
        uptimeSeconds,
        uptimeHuman: formatUptime(uptimeSeconds)
      },
      recent: {
        lastHourExecutions: recentExecutions.length,
        recentSuccessRate: recentExecutions.length > 0
          ? ((recentExecutions.filter(e => e.success).length / recentExecutions.length) * 100).toFixed(1)
          : 0,
        packagesInstalled: executionStats.packageInstalls,
        filesUploaded: executionStats.fileUploads
      },
      topIPs: Array.from(executionStats.ipCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([ip, count]) => ({ ip, count })),
      topErrors: Array.from(executionStats.topErrors.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([error, count]) => ({ error, count })),
      userAgents: Array.from(executionStats.userAgents.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([agent, count]) => ({ agent, count })),
      hourlyTrend: generateHourlyTrend(recentExecutions)
    };
  },
  
  /**
   * Clears all collected statistics and resets counters to zero.
   * Useful for testing, debugging, or periodic statistics reset.
   * 
   * @returns {void}
   * 
   * @example
   * // Reset stats for new monitoring period
   * logger.clearStats();
   * 
   * // In test cleanup
   * afterEach(() => {
   *   logger.clearStats();
   * });
   * 
   * @description
   * - Resets all numeric counters to 0
   * - Clears all Map collections (IPs, errors, user agents)
   * - Empties recent executions array
   * - Updates start time to current moment
   * - Does not affect log files or configuration
   */
  clearStats() {
    executionStats.totalExecutions = 0;
    executionStats.successCount = 0;
    executionStats.errorCount = 0;
    executionStats.totalExecutionTime = 0;
    executionStats.ipCounts.clear();
    executionStats.codePatterns.clear();
    executionStats.packageInstalls = 0;
    executionStats.fileUploads = 0;
    executionStats.lastHourExecutions = [];
    executionStats.topErrors.clear();
    executionStats.userAgents.clear();
    executionStats.startTime = Date.now();
  },
  
  /**
   * Retrieves current logging configuration and file information.
   * Useful for debugging and system administration.
   * 
   * @returns {Object} Logging configuration object
   * @returns {string} returns.level - Current log level (error, warn, info, debug)
   * @returns {string} returns.logFile - Path to main log file
   * @returns {string} returns.securityLogFile - Path to security log file
   * @returns {boolean} returns.isFileLoggingEnabled - Whether file logging is active
   * @returns {string|null} returns.logDirectory - Directory containing log files
   * 
   * @example
   * const logInfo = logger.getLogInfo();
   * console.log(`Logging level: ${logInfo.level}`);
   * console.log(`Log files in: ${logInfo.logDirectory}`);
   * 
   * // Check if file logging is working
   * if (!logInfo.isFileLoggingEnabled) {
   *   console.warn('File logging disabled - check LOG_FILE environment variable');
   * }
   * 
   * @description
   * - Returns current configuration state
   * - Helps diagnose logging issues
   * - Useful for health checks and monitoring
   * - Safe to call frequently
   */
  getLogInfo() {
    return {
      level: Object.keys(levels).find((k) => levels[k] === currentLevel),
      logFile,
      securityLogFile,
      isFileLoggingEnabled: !!logFile,
      logDirectory: logFile ? path.dirname(logFile) : null,
    };
  },
  getLogFile() {
    return logFile;
  },
  getLevel() {
    return Object.keys(levels).find((k) => levels[k] === currentLevel);
  },
};

/**
 * Formats uptime seconds into human-readable duration string.
 * Automatically selects appropriate units for readability.
 * 
 * @param {number} seconds - Uptime duration in seconds
 * @returns {string} Formatted duration string
 * 
 * @example
 * formatUptime(45); // "45s"
 * formatUptime(150); // "2m 30s"
 * formatUptime(3665); // "1h 1m 5s"
 * formatUptime(90000); // "1d 1h 0m"
 * 
 * @description
 * - Uses most significant units (days > hours > minutes > seconds)
 * - Omits leading zero units for cleaner output
 * - Handles edge cases (0 seconds, very large values)
 * - Returns consistent format for UI display
 */
function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

/**
 * Generates 24-hour execution trend data for dashboard charts.
 * Creates array of hourly execution counts from recent execution history.
 * 
 * @param {Array<Object>} executions - Array of execution objects with timestamps
 * @param {number} executions[].timestamp - Execution timestamp in milliseconds
 * @param {boolean} executions[].success - Whether execution was successful
 * @param {number} executions[].executionTime - Execution duration in ms
 * @param {string} executions[].ip - Client IP address
 * @returns {Array<number>} Array of 24 numbers representing hourly execution counts
 * 
 * @example
 * const executions = [
 *   { timestamp: Date.now() - 3600000, success: true }, // 1 hour ago
 *   { timestamp: Date.now() - 7200000, success: false }, // 2 hours ago
 *   { timestamp: Date.now() - 86400000, success: true } // 24 hours ago (excluded)
 * ];
 * 
 * const trend = generateHourlyTrend(executions);
 * console.log(trend); // [0, 0, ..., 1, 1, 0, ...] (24 numbers)
 * 
 * // Use with Chart.js
 * const chartData = {
 *   labels: Array.from({length: 24}, (_, i) => `${i}:00`),
 *   datasets: [{
 *     label: 'Executions per Hour',
 *     data: generateHourlyTrend(recentExecutions)
 *   }]
 * };
 * 
 * @description
 * - Creates 24-element array (one per hour)
 * - Index 0 = 23 hours ago, Index 23 = current hour
 * - Only counts executions within last 24 hours
 * - Returns zero-filled array if no recent executions
 * - Perfect for time-series charts and trend analysis
 */
function generateHourlyTrend(executions) {
  const now = Date.now();
  const hourlyBuckets = new Array(24).fill(0);
  
  executions.forEach(exec => {
    const hoursAgo = Math.floor((now - exec.timestamp) / (60 * 60 * 1000));
    if (hoursAgo < 24) {
      hourlyBuckets[23 - hoursAgo]++;
    }
  });
  
  return hourlyBuckets;
}

module.exports = logger;
