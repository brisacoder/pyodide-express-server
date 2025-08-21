/**
 * @file Type definitions for Pyodide Express Server
 * @description Centralized JSDoc type definitions for better IDE support and type checking
 */
/**
 * Express Request object with extended properties
 * @typedef {Object} ExpressRequest
 * @property {string} method - HTTP method (GET, POST, etc.)
 * @property {string} url - Request URL
 * @property {Object} headers - Request headers
 * @property {Object} params - Route parameters
 * @property {Object} query - Query string parameters
 * @property {*} body - Request body (parsed)
 * @property {string} ip - Client IP address
 * @property {string} requestId - Unique request identifier (added by middleware)
 * @property {Object} file - Uploaded file information (from multer)
 * @property {Array<Object>} files - Multiple uploaded files (from multer)
 */
/**
 * Express Response object with extended properties
 * @typedef {Object} ExpressResponse
 * @property {number} statusCode - HTTP status code
 * @property {Function} status - Set status code
 * @property {Function} json - Send JSON response
 * @property {Function} send - Send response
 * @property {Function} set - Set response header
 * @property {Object} locals - Response local variables
 */
/**
 * Express Next function
 * @typedef {Function} ExpressNext
 * @param {Error} [error] - Optional error to pass to error handler
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
 * Recent activity metrics
 * @typedef {Object} RecentMetrics
 * @property {number} lastHour - Executions in the last hour
 * @property {number} lastMinute - Executions in the last minute
 * @property {Array<number>} hourlyTrend - 24-hour execution trend data
 */
/**
 * Top IP address statistics
 * @typedef {Object} TopIPStats
 * @property {string} ip - IP address
 * @property {number} count - Number of requests from this IP
 * @property {number} percentage - Percentage of total requests
 */
/**
 * Top error statistics
 * @typedef {Object} TopErrorStats
 * @property {string} error - Error message or type
 * @property {number} count - Number of occurrences
 * @property {number} percentage - Percentage of total errors
 */
/**
 * User agent statistics
 * @typedef {Object} UserAgentStats
 * @property {string} userAgent - User agent string
 * @property {number} count - Number of requests from this user agent
 * @property {number} percentage - Percentage of total requests
 */
/**
 * Complete statistics object returned by getStats()
 * @typedef {Object} CompleteStats
 * @property {StatsOverview} overview - High-level statistics summary
 * @property {RecentMetrics} recent - Recent activity metrics
 * @property {Array<TopIPStats>} topIPs - Most active IP addresses
 * @property {Array<TopErrorStats>} topErrors - Most common error types
 * @property {Array<UserAgentStats>} userAgents - Most common user agents
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
/**
 * Python code execution context variables
 * @typedef {Object} PythonContext
 * @property {*} [key] - Any variables to make available in Python scope
 */
/**
 * Python code execution result
 * @typedef {Object} PythonExecutionResult
 * @property {boolean} success - Whether execution completed successfully
 * @property {*} [result] - Return value from Python code (if any)
 * @property {string} [error] - Error message if execution failed
 * @property {string} [stdout] - Standard output from Python execution
 * @property {string} [stderr] - Standard error from Python execution
 * @property {number} executionTime - Time taken to execute in milliseconds
 * @property {Object} [metadata] - Additional execution metadata
 */
/**
 * Package installation result
 * @typedef {Object} PackageInstallResult
 * @property {boolean} success - Whether installation was successful
 * @property {string} package - Name of the package
 * @property {string} [version] - Installed package version
 * @property {string} [error] - Error message if installation failed
 * @property {number} installTime - Time taken to install in milliseconds
 */
/**
 * Available packages list
 * @typedef {Object} AvailablePackages
 * @property {boolean} success - Whether package list retrieval was successful
 * @property {Array<string>} packages - List of available package names
 * @property {number} count - Total number of available packages
 * @property {string} [error] - Error message if retrieval failed
 */
/**
 * Multer file upload object
 * @typedef {Object} UploadedFile
 * @property {string} fieldname - Field name specified in the form
 * @property {string} originalname - Name of the file on the user's computer
 * @property {string} encoding - Encoding type of the file
 * @property {string} mimetype - Mime type of the file
 * @property {Buffer} buffer - Raw file data (if using memory storage)
 * @property {number} size - Size of the file in bytes
 * @property {string} [destination] - Destination folder (if using disk storage)
 * @property {string} [filename] - Generated filename (if using disk storage)
 * @property {string} [path] - Full path to uploaded file (if using disk storage)
 */
module.exports = {}; // Export empty object for JSDoc purposes
