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

function rotateIfNeeded(fileName = logFile) {
  if (!fileName || !fs.existsSync(fileName)) return;
  const { size } = fs.statSync(fileName);
  if (size < maxSize) return;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rotated = `${fileName}.${timestamp}`;
  fs.renameSync(fileName, rotated);
}

function writeToFile(fileName, content) {
  if (!fileName) return;
  try {
    rotateIfNeeded(fileName);
    fs.appendFileSync(fileName, content + '\n');
  } catch (err) {
    console.error(`Failed to write to ${fileName}:`, err.message);
  }
}

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
  
  // Enhanced security logging
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
  
  // Statistics access
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
