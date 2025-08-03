/**
 * Validation middleware for Pyodide Express Server
 * 
 * Provides input validation for API endpoints to ensure
 * safe and proper data before processing.
 */

const logger = require('../utils/logger');

/**
 * Validate Python code execution request
 */
function validateCode(req, res, next) {
  const { code, context, timeout } = req.body;
  
  // Check if code is provided
  if (!code) {
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
      error: 'Code cannot be empty.'
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
    } catch (error) {
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
 * Validate package installation request
 */
function validatePackage(req, res, next) {
  const { package: packageName } = req.body;
  
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
 */
function sanitizeInput(req, res, next) {
  // Helper function to recursively sanitize objects
  function sanitize(obj) {
    if (typeof obj === 'string') {
      // Remove null bytes and other control characters
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