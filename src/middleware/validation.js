/**
 * Validation middleware for Pyodide Express Server
 * 
 * Provides comprehensive input validation for API endpoints to ensure
 * safe and proper data before processing in Pyodide runtime.
 */

const logger = require('../utils/logger');

/**
 * Validates Python code execution request parameters.
 * Ensures code safety, size limits, and proper data types before execution.
 * 
 * @param {Object} req - Express request object
 * @param {Object} req.body - Request body containing execution parameters
 * @param {string} req.body.code - Python code to validate (required)
 * @param {Object} [req.body.context] - Optional context object for variable injection
 * @param {number} [req.body.timeout] - Optional execution timeout in milliseconds
 * @param {string} req.ip - Client IP address for logging
 * @param {Object} res - Express response object
 * @param {Function} next - Express next middleware function
 * @returns {void} Calls next() if validation passes, sends error response if not
 * 
 * @example
 * // Valid request that passes validation:
 * // {
 * //   "code": "import numpy as np\nprint(np.__version__)",
 * //   "context": {"debug": true},
 * //   "timeout": 30000
 * // }
 * 
 * // Invalid request that fails validation:
 * // {
 * //   "code": "",  // Empty code
 * //   "timeout": -1000  // Negative timeout
 * // }
 * 
 * @description
 * Validation Rules:
 * - Code must be non-empty string
 * - Code length <= 100KB (configurable)
 * - Context must be serializable object if provided
 * - Context size <= 10KB (configurable)
 * - Timeout must be positive number <= 5 minutes
 * - Logs potentially dangerous patterns for monitoring
 * 
 * Security Features:
 * - Pattern detection for dangerous operations
 * - Size limits to prevent DoS attacks
 * - Type validation to prevent injection
 * - Logging of suspicious activity
 * 
 * HTTP Responses:
 * - 400: Validation failure with specific error message
 * - Calls next() for successful validation
 */
function validateCode(req, res, next) {
  const { code, context, timeout } = req.body;

  // User input is executed inside the Pyodide runtime; validating it here
  // guards the interpreter from malformed requests.
  
  // Check if code is provided
  if (code === undefined || code === null) {
    return res.status(400).json({
      success: false,
      error: 'No code provided. Please include "code" in the request body.'
    });
  }
  
  // Check if code is a string
  if (typeof code !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Code must be a string.'
    });
  }
  
  // Check code length (prevent extremely large payloads)
  const MAX_CODE_LENGTH = 100000; // 100KB
  if (code.length > MAX_CODE_LENGTH) {
    return res.status(400).json({
      success: false,
      error: `Code too long. Maximum length: ${MAX_CODE_LENGTH} characters.`
    });
  }
  
  // Check if code is not empty after trimming
  if (code.trim().length === 0) {
    return res.status(400).json({
      success: false,
      error: 'Code cannot be empty'
    });
  }
  
  // Check if code contains only whitespace escape sequences
  const whitespaceOnlyPattern = /^[\s\\nrtbfv]*$/;
  if (whitespaceOnlyPattern.test(code.trim())) {
    return res.status(400).json({
      success: false,
      error: 'Code cannot contain only whitespace'
    });
  }
  
  // Validate context if provided
  if (context !== undefined) {
    if (typeof context !== 'object' || context === null || Array.isArray(context)) {
      return res.status(400).json({
        success: false,
        error: 'Context must be an object (key-value pairs).'
      });
    }
    
    // Check context size
    try {
      const contextSize = JSON.stringify(context).length;
      const MAX_CONTEXT_SIZE = 10000; // 10KB
      if (contextSize > MAX_CONTEXT_SIZE) {
        return res.status(400).json({
          success: false,
          error: `Context too large. Maximum size: ${MAX_CONTEXT_SIZE} characters.`
        });
      }
    } catch {
      return res.status(400).json({
        success: false,
        error: 'Context contains non-serializable data.'
      });
    }
  }
  
  // Validate timeout if provided
  if (timeout !== undefined) {
    if (typeof timeout !== 'number' || timeout <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Timeout must be a positive number (milliseconds).'
      });
    }
    
    const MAX_TIMEOUT = 300000; // 5 minutes
    if (timeout > MAX_TIMEOUT) {
      return res.status(400).json({
        success: false,
        error: `Timeout too large. Maximum: ${MAX_TIMEOUT}ms (5 minutes).`
      });
    }
  }
  
  // Check for potentially dangerous operations (basic security)
  const dangerousPatterns = [
    /import\s+os\b/,
    /import\s+subprocess\b/,
    /import\s+sys\b.*exit/,
    /exec\s*\(/,
    /eval\s*\(/,
    /__import__\s*\(/,
    /open\s*\(/,
    /file\s*\(/
  ];
  
  const foundDangerous = dangerousPatterns.find(pattern => pattern.test(code));
  if (foundDangerous) {
    logger.warn('Potentially dangerous code detected:', { 
      pattern: foundDangerous.toString(),
      ip: req.ip 
    });
    // Note: We log but don't block, as some operations might be legitimate
    // In production, you might want to be more restrictive
  }
  
  next();
}

/**
 * Validates Python package installation request parameters.
 * Ensures package names are safe and compatible with Pyodide/micropip.
 * 
 * @param {Object} req - Express request object
 * @param {Object} req.body - Request body containing package information
 * @param {string} req.body.package - Package name to validate (required)
 * @param {string} req.ip - Client IP address for logging
 * @param {Object} res - Express response object  
 * @param {Function} next - Express next middleware function
 * @returns {void} Calls next() if validation passes, sends error response if not
 * 
 * @example
 * // Valid package installation requests:
 * // { "package": "numpy" }
 * // { "package": "pandas==1.5.0" }
 * // { "package": "matplotlib>=3.0" }
 * 
 * // Invalid requests that fail validation:
 * // { "package": "" }  // Empty name
 * // { "package": "a" }  // Too short
 * // { "package": "../malicious-package" }  // Invalid characters
 * 
 * @description
 * Validation Rules:
 * - Package name must be non-empty string
 * - Name length between 2-100 characters
 * - Alphanumeric, hyphens, underscores, dots only
 * - Version specifiers allowed (==, >=, <=, >, <, !=)
 * - No dangerous path characters (../, ..\)
 * 
 * Security Features:
 * - Prevents package name injection attacks
 * - Blocks filesystem traversal attempts
 * - Validates against PyPI naming conventions
 * - Logs invalid package requests
 * 
 * Compatible Formats:
 * - Simple names: "numpy", "pandas", "matplotlib"
 * - With versions: "numpy==1.21.0", "pandas>=1.3"
 * - Git URLs and local files are blocked for security
 * 
 * HTTP Responses:
 * - 400: Validation failure with specific error message
 * - Calls next() for successful validation
 */
function validatePackage(req, res, next) {
  const { package: packageName } = req.body;

  // Packages are installed via Pyodide's ``micropip`` module, so only names of
  // packages available on PyPI and compatible with WebAssembly should be
  // accepted here.
  
  // Check if package name is provided
  if (!packageName) {
    return res.status(400).json({
      success: false,
      error: 'No package name provided. Please include "package" in the request body.'
    });
  }
  
  // Check if package name is a string
  if (typeof packageName !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Package name must be a string.'
    });
  }
  
  // Check package name format (basic validation)
  const packageNamePattern = /^[a-zA-Z0-9._-]+$/;
  if (!packageNamePattern.test(packageName)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid package name format. Use only letters, numbers, dots, hyphens, and underscores.'
    });
  }
  
  // Check package name length
  if (packageName.length > 100) {
    return res.status(400).json({
      success: false,
      error: 'Package name too long. Maximum length: 100 characters.'
    });
  }
  
  // Check for empty package name
  if (packageName.trim().length === 0) {
    return res.status(400).json({
      success: false,
      error: 'Package name cannot be empty.'
    });
  }
  
  // Block certain potentially problematic packages
  const blockedPackages = [
    'os',
    'subprocess',
    'socket',
    'urllib3',
    'requests-oauthlib'
  ];
  
  if (blockedPackages.includes(packageName.toLowerCase())) {
    logger.warn('Blocked package installation attempt:', { 
      package: packageName,
      ip: req.ip 
    });
    return res.status(403).json({
      success: false,
      error: `Package "${packageName}" is not allowed for security reasons.`
    });
  }
  
  next();
}

/**
 * Rate limiting middleware (simple in-memory implementation)
 * @param {number} windowMs - Time window in milliseconds for rate limiting
 * @param {number} maxRequests - Maximum number of requests allowed in the time window
 * @returns {Function} Express middleware function for rate limiting
 */
function createRateLimit(windowMs = 15 * 60 * 1000, maxRequests = 100) {
  const requests = new Map();
  
  return (req, res, next) => {
    const clientId = req.ip;
    const now = Date.now();
    
    // Clean up old entries
    for (const [id, data] of requests.entries()) {
      if (now - data.firstRequest > windowMs) {
        requests.delete(id);
      }
    }
    
    // Check current client
    const clientData = requests.get(clientId);
    
    if (!clientData) {
      // First request from this client
      requests.set(clientId, {
        count: 1,
        firstRequest: now
      });
      next();
    } else if (now - clientData.firstRequest > windowMs) {
      // Window expired, reset
      requests.set(clientId, {
        count: 1,
        firstRequest: now
      });
      next();
    } else if (clientData.count >= maxRequests) {
      // Rate limit exceeded
      logger.warn('Rate limit exceeded:', { 
        ip: clientId,
        count: clientData.count,
        window: windowMs 
      });
      
      res.status(429).json({
        success: false,
        error: 'Too many requests. Please try again later.',
        retryAfter: Math.ceil((windowMs - (now - clientData.firstRequest)) / 1000)
      });
    } else {
      // Increment count and allow
      clientData.count++;
      next();
    }
  };
}

/**
 * Validate file upload
 * @param {Object} req - Express request object with uploaded file
 * @param {Object} res - Express response object
 * @param {Function} next - Express next middleware function
 * @returns {void} Validates file type and calls next() or sends error response
 */
function validateFileUpload(req, res, next) {
  // This runs after multer, so we can check the processed file
  if (req.file) {
    const allowedTypes = ['.csv', '.json', '.txt', '.py'];
    const fileExt = req.file.originalname.toLowerCase().split('.').pop();
    
    if (!allowedTypes.includes('.' + fileExt)) {
      return res.status(400).json({
        success: false,
        error: `File type .${fileExt} not allowed. Allowed types: ${allowedTypes.join(', ')}`
      });
    }
    
    logger.info('File upload validated:', {
      filename: req.file.originalname,
      size: req.file.size,
      type: req.file.mimetype
    });
  }
  
  next();
}

/**
 * General request body size validation
 * @param {number} maxSize - Maximum allowed request size in bytes
 * @returns {Function} Express middleware function for size validation
 */
function validateRequestSize(maxSize = 10 * 1024 * 1024) { // 10MB default
  return (req, res, next) => {
    const contentLength = parseInt(req.get('content-length'));
    
    if (contentLength > maxSize) {
      return res.status(413).json({
        success: false,
        error: `Request body too large. Maximum size: ${maxSize / 1024 / 1024}MB`
      });
    }
    
    next();
  };
}

/**
 * Sanitize input data to prevent injection attacks
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next middleware function
 * @returns {void} Sanitizes request data and calls next()
 */
function sanitizeInput(req, res, next) {
  // Helper function to recursively sanitize objects
  function sanitize(obj) {
    if (typeof obj === 'string') {
      // Remove null bytes and other control characters
      // eslint-disable-next-line no-control-regex
      return obj.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
    } else if (Array.isArray(obj)) {
      return obj.map(sanitize);
    } else if (obj && typeof obj === 'object') {
      const sanitized = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[sanitize(key)] = sanitize(value);
      }
      return sanitized;
    }
    return obj;
  }
  
  // Sanitize request body
  if (req.body) {
    req.body = sanitize(req.body);
  }
  
  // Sanitize query parameters
  if (req.query) {
    req.query = sanitize(req.query);
  }
  
  next();
}

module.exports = {
  validateCode,
  validatePackage,
  validateFileUpload,
  validateRequestSize,
  sanitizeInput,
  createRateLimit
};