/**
 * Logger utility for the Pyodide Express Server
 *
 * Provides consistent logging across the application with different log levels
 * and optional file output for production environments. Detailed logs help
 * diagnose issues when executing Python code through Pyodide.
 */

const fs = require('fs');
const path = require('path');

class Logger {
  constructor() {
    this.levels = {
      ERROR: 0,
      WARN: 1,
      INFO: 2,
      DEBUG: 3
    };
    
    this.currentLevel = this.levels[process.env.LOG_LEVEL?.toUpperCase()] || this.levels.INFO;
    this.logFile = process.env.LOG_FILE || null;
    
    // Ensure log directory exists if log file is specified
    if (this.logFile) {
      const logDir = path.dirname(this.logFile);
      if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
      }
    }
  }

  /**
   * Format log message with timestamp and level
   * @private
   */
  _formatMessage(level, message, ...args) {
    const timestamp = new Date().toISOString();
    const formattedArgs = args.length > 0 ? ' ' + args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ') : '';
    
    return `[${timestamp}] ${level}: ${message}${formattedArgs}`;
  }

  /**
   * Write log message to console and optionally to file
   * @private
   */
  _writeLog(level, levelNum, message, ...args) {
    if (levelNum > this.currentLevel) return;

    const formattedMessage = this._formatMessage(level, message, ...args);
    
    // Console output with colors
    switch (level) {
      case 'ERROR':
        console.error('\x1b[31m%s\x1b[0m', formattedMessage); // Red
        break;
      case 'WARN':
        console.warn('\x1b[33m%s\x1b[0m', formattedMessage); // Yellow
        break;
      case 'INFO':
        console.info('\x1b[36m%s\x1b[0m', formattedMessage); // Cyan
        break;
      case 'DEBUG':
        console.debug('\x1b[35m%s\x1b[0m', formattedMessage); // Magenta
        break;
      default:
        console.log(formattedMessage);
    }
    
    // File output (without colors)
    if (this.logFile) {
      try {
        fs.appendFileSync(this.logFile, formattedMessage + '\n');
      } catch (error) {
        console.error('Failed to write to log file:', error.message);
      }
    }
  }

  /**
   * Log error messages
   */
  error(message, ...args) {
    this._writeLog('ERROR', this.levels.ERROR, message, ...args);
  }

  /**
   * Log warning messages
   */
  warn(message, ...args) {
    this._writeLog('WARN', this.levels.WARN, message, ...args);
  }

  /**
   * Log info messages
   */
  info(message, ...args) {
    this._writeLog('INFO', this.levels.INFO, message, ...args);
  }

  /**
   * Log debug messages
   */
  debug(message, ...args) {
    this._writeLog('DEBUG', this.levels.DEBUG, message, ...args);
  }

  /**
   * Set log level
   */
  setLevel(level) {
    const upperLevel = level.toUpperCase();
    if (this.levels[upperLevel] !== undefined) {
      this.currentLevel = this.levels[upperLevel];
      this.info(`Log level set to ${upperLevel}`);
    } else {
      this.warn(`Invalid log level: ${level}. Valid levels: ${Object.keys(this.levels).join(', ')}`);
    }
  }

  /**
   * Get current log level
   */
  getLevel() {
    return Object.keys(this.levels).find(key => this.levels[key] === this.currentLevel);
  }
  /**
   * Get log file path
   */
  getLogFile() {
    return this.logFile;
  }

  /**
   * Get log info for status checks
   */
  getLogInfo() {
    return {
      level: this.getLevel(),
      logFile: this.logFile,
      isFileLoggingEnabled: !!this.logFile,
      logDirectory: this.logFile ? require('path').dirname(this.logFile) : null
    };
  }
}

module.exports = new Logger();