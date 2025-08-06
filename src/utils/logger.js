/**
 * Structured JSON logger with basic rotation and request ID support.
 */
const fs = require('fs');
const path = require('path');
const { getRequestId } = require('./requestContext');

const levels = { error: 0, warn: 1, info: 2, debug: 3 };
const currentLevel = levels[(process.env.LOG_LEVEL || 'info').toLowerCase()] ?? levels.info;

const logDir = process.env.LOG_DIR || path.join(__dirname, '../../logs');
const logFile = process.env.LOG_FILE || path.join(logDir, 'server.log');
const maxSize = parseInt(process.env.LOG_MAX_SIZE || 5 * 1024 * 1024, 10); // 5MB default

if (logFile) {
  const dir = path.dirname(logFile);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function rotateIfNeeded() {
  if (!logFile || !fs.existsSync(logFile)) return;
  const { size } = fs.statSync(logFile);
  if (size < maxSize) return;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rotated = `${logFile}.${timestamp}`;
  fs.renameSync(logFile, rotated);
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
      console.log(line);
      break;
    case 'debug':
      console.debug(line);
      break;
    default:
      console.log(line);
  }

  if (logFile) {
    try {
      rotateIfNeeded();
      fs.appendFileSync(logFile, line + '\n');
    } catch (err) {
      console.error('Failed to write to log file:', err.message);
    }
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
  getLogInfo() {
    return {
      level: Object.keys(levels).find((k) => levels[k] === currentLevel),
      logFile,
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

module.exports = logger;
