/**
 * Structured JSON logger with basic rotation, request ID support, and security logging.
 */

/**
 * Execution statistics data object
 * @typedef {Object} ExecutionStatsData
 * @property {string} [ip] - Client IP address
 * @property {boolean} [success=false] - Whether execution was successful
 * @property {number} [executionTime] - Execution time in milliseconds  
 * @property {string} [error] - Error message if execution failed
 * @property {string} [userAgent] - Client user agent string
 * @property {string} [codeHash] - SHA-256 hash of executed code
 */

/**
 * Security logging event data
 * @typedef {Object} SecurityEventData
 * @property {string} type - Type of security event
 * @property {string} [ip] - Client IP address
 * @property {string} [userAgent] - Client user agent string
 * @property {string} [endpoint] - API endpoint involved
 * @property {*} [details] - Additional event-specific details
 */

/**
 * Statistics overview object
 * @typedef {Object} StatsOverview
 * @property {number} totalExecutions - Total number of code executions
 * @property {number} successCount - Number of successful executions
 * @property {number} errorCount - Number of failed executions
 * @property {number} successRate - Success rate as percentage (0-100)
 * @property {number} averageExecutionTime - Average execution time in ms
 * @property {number} uniqueIPs - Number of unique IP addresses
 * @property {string} uptime - Formatted server uptime string
 */

/**
 * Complete statistics object returned by getStats()
 * @typedef {Object} CompleteStats
 * @property {StatsOverview} overview - High-level statistics summary
 * @property {Object} recent - Recent activity metrics
 * @property {Array<Object>} topIPs - Most active IP addresses
 * @property {Array<Object>} topErrors - Most common error types
 * @property {Array<Object>} userAgents - Most common user agents
 */

/**
 * Logging configuration information
 * @typedef {Object} LoggingConfig
 * @property {string} level - Current log level
 * @property {string} logFile - Path to main log file
 * @property {string} securityLogFile - Path to security log file
 * @property {boolean} isFileLoggingEnabled - Whether file logging is enabled
 * @property {string} logDirectory - Directory containing log files
 */

/**
 * Execution object with timestamp for trend analysis
 * @typedef {Object} ExecutionRecord
 * @property {number} timestamp - Execution timestamp in milliseconds
 * @property {boolean} success - Whether execution was successful
 * @property {number} [executionTime] - Execution time in milliseconds
 * @property {string} [ip] - Client IP address
 * @property {string} [error] - Error message if failed
 */

const fs = require('fs');
const path = require('path');
const constants = require('../config/constants');
const { getRequestId } = require('./requestContext');

const levels = { error: 0, warn: 1, info: 2, debug: 3 };
const currentLevel = levels[(process.env.LOG_LEVEL || 'info').toLowerCase()] ?? levels.info;

const logDir = process.env.LOG_DIR || path.join(__dirname, '../../logs');
const logFile = process.env.LOG_FILE || path.join(logDir, 'server.log');
const securityLogFile = path.join(logDir, 'security.log');
const maxSize = parseInt(process.env.LOG_MAX_SIZE || constants.LOGGING.MAX_LOG_SIZE, 10);

/**
 * In-memory execution statistics storage
 * @typedef {Object} ExecutionStatsStorage
 * @property {number} totalExecutions - Total number of code executions
 * @property {number} successCount - Count of successful executions
 * @property {number} errorCount - Count of failed executions
 * @property {number} totalExecutionTime - Sum of all execution times in ms
 * @property {Map<string, number>} ipCounts - Map of IP addresses to request counts
 * @property {Map<string, number>} codePatterns - Map of code hash patterns to counts
 * @property {number} packageInstalls - Count of package installations
 * @property {number} fileUploads - Count of file uploads
 * @property {Array<ExecutionRecord>} lastHourExecutions - Recent execution records
 * @property {Map<string, number>} topErrors - Map of error messages to counts
 * @property {Map<string, number>} userAgents - Map of user agents to counts
 * @property {number} startTime - Server start timestamp
 */

/**
 * @type {ExecutionStatsStorage}
 */
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
 * @param {ExecutionStatsData} data - Execution data object with performance metrics
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
  const oneHourAgo = now - constants.TIME.HOUR;
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
   * @param {SecurityEventData} data - Event-specific data object with security context
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
   * @returns {CompleteStats} Statistics object with multiple categories including overview, recent metrics, top IPs, errors, and user agents
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
   * @returns {LoggingConfig} Logging configuration object with level, file paths, and settings
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
  const days = Math.floor(seconds / constants.TIME.SECONDS_PER_DAY);
  const hours = Math.floor((seconds % constants.TIME.SECONDS_PER_DAY) / constants.TIME.SECONDS_PER_HOUR);
  const minutes = Math.floor((seconds % constants.TIME.SECONDS_PER_HOUR) / constants.TIME.SECONDS_PER_MINUTE);
  const secs = seconds % constants.TIME.SECONDS_PER_MINUTE;
  
  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

/**
 * Generates 24-hour execution trend data for dashboard charts.
 * Creates array of hourly execution counts from recent execution history.
 * 
 * @param {Array<ExecutionRecord>} executions - Array of execution objects with timestamps
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
    const hoursAgo = Math.floor((now - exec.timestamp) / constants.TIME.HOUR);
    if (hoursAgo < constants.TIME.HOURS_PER_DAY) {
      hourlyBuckets[constants.TIME.HOURS_PER_DAY - 1 - hoursAgo]++;
    }
  });
  
  return hourlyBuckets;
}

module.exports = logger;
