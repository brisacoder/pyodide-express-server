const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
const crypto = require('crypto');
/**
 * Executes raw Python code submitted as plain text in request body.
 * Designed for simple API integrations and command-line tools.
 *
 * @param {Object} req - Express request object
 * @param {string} req.body - Raw Python code as string (not JSON)
 * @param {string} req.ip - Client IP address (added by Express)
 * @param {Function} req.get - Function to get request headers
 * @param {Object} res - Express response object
 * @returns {Promise<void>} Resolves when response is sent
 *
 * @example
 * // POST /api/execute-raw
 * // Content-Type: text/plain
 * // Body: import numpy as np; print(np.__version__)
 *
 * // Success response:
 * // {
 * //   "success": true,
 * //   "data": {
 * //     "result": null,
 * //     "stdout": "1.24.3\n",
 * //     "stderr": "",
 * //     "executionTime": 150
 * //   },
 * //   "error": null,
 * //   "meta": {
 * //     "timestamp": "2025-08-20T10:30:00Z"
 * //   }
 * // }
 *
 * // Error response:
 * // {
 * //   "success": false,
 * //   "data": null,
 * //   "error": "SyntaxError: invalid syntax",
 * //   "meta": {
 * //     "timestamp": "2025-08-20T10:30:00Z"
 * //   }
 * // }
 *
 * @description
 * HTTP Status Codes:
 * - 200: Successful execution (regardless of Python errors)
 * - 400: Invalid request (empty code, wrong content type)
 * - 500: Server error (Pyodide initialization failure, etc.)
 *
 * Security Features:
 * - Generates SHA-256 hash of code for tracking
 * - Logs execution attempts with IP and User-Agent
 * - Measures and logs execution time
 * - Validates input format and content
 *
 * Rate Limiting:
 * - Subject to global rate limits
 * - Large code blocks may be rejected
 * - Execution timeout applies (default 30s)
 */
async function executeRaw(req, res) {
  const startTime = Date.now();
  try {
    const code = req.body;
    if (!code || typeof code !== 'string' || !code.trim()) {
      return res.status(400).json({
        success: false,
        data: null,
        error: 'No Python code provided in request body',
        meta: {
          timestamp: new Date().toISOString()
        }
      });
    }
    // Generate code hash for tracking and security
    const codeHash = crypto.createHash('sha256').update(code.trim()).digest('hex');
    logger.info('Executing raw Python code:', {
      codeLength: code.length,
      codeHash: codeHash.substring(0, 16), // Log first 16 chars of hash
      ip: req.ip,
      contentType: req.get('Content-Type'),
    });
    const result = await pyodideService.executeCode(code);
    const executionTime = Date.now() - startTime;
    // Security logging with complete execution data
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: result.success,
      executionTime,
      codeHash,
      codeLength: code.length,
      executionType: 'raw',
      error: result.success ? null : result.error,
      timestamp: new Date().toISOString(),
    });
    if (result.success) {
      logger.info('Raw code execution successful', {
        executionTime,
        codeHash: codeHash.substring(0, 16),
      });
    } else {
      logger.warn('Raw code execution failed:', {
        error: result.error,
        executionTime,
        codeHash: codeHash.substring(0, 16),
      });
    }
    // Format response according to API contract
    res.json({
      success: result.success,
      data: result.success ? {
        result: {
          stdout: result.stdout || result.result || '',
          stderr: result.stderr || '',
          executionTime: executionTime
        }
      } : null,
      error: result.success ? null : result.error,
      meta: {
        timestamp: result.timestamp || new Date().toISOString()
      }
    });
  } catch (error) {
    const executionTime = Date.now() - startTime;
    logger.error('Raw execution endpoint error:', {
      error: error.message,
      executionTime,
    });
    // Log security event for errors
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: false,
      executionTime,
      error: error.message,
    });
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
}
/**
 * Executes Python code with structured JSON input and optional context.
 * Main execution endpoint supporting advanced features and configuration.
 *
 * @param {Object} req - Express request object
 * @param {Object} req.body - JSON request body
 * @param {string} req.body.code - Python code to execute (required)
 * @param {Object} [req.body.context] - Additional context/variables to inject
 * @param {number} [req.body.timeout] - Execution timeout in milliseconds (default: 30000)
 * @param {string} req.ip - Client IP address
 * @param {Function} req.get - Function to get request headers
 * @param {Object} res - Express response object
 * @returns {Promise<void>} Resolves when response is sent
 *
 * @example
 * // POST /api/execute
 * // Content-Type: application/json
 * // {
 * //   "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.savefig('/plots/matplotlib/chart.png')",
 * //   "context": {"title": "My Chart"},
 * //   "timeout": 45000
 * // }
 *
 * // Success response:
 * // {
 * //   "success": true,
 * //   "data": {
 * //     "result": null,
 * //     "stdout": "",
 * //     "stderr": "",
 * //     "executionTime": 890
 * //   },
 * //   "error": null,
 * //   "meta": {
 * //     "timestamp": "2025-08-20T10:30:00Z"
 * //   }
 * // }
 *
 * // Complex execution with data analysis:
 * // {
 * //   "code": "df = pd.read_csv('/uploads/data.csv')\nresult = df.describe()\nprint(result)",
 * //   "context": {"debug": true}
 * // }
 *
 * @description
 * Advanced Features:
 * - Context injection for dynamic variables
 * - Custom timeout configuration
 * - Plot file detection and listing
 * - Comprehensive error reporting
 * - Execution time measurement
 *
 * Security Features:
 * - Code hashing for duplicate detection
 * - Complete audit trail logging
 * - Input validation and sanitization
 * - Resource usage monitoring
 *
 * Response Fields:
 * - success: Boolean indicating execution success
 * - output: Captured stdout/stderr from Python
 * - plots: Array of generated plot file paths
 * - executionTime: Duration in milliseconds
 * - error: Error message if execution failed
 * - timestamp: ISO timestamp of completion
 */
async function executeCode(req, res) {
  const startTime = Date.now();
  try {
    const { code, context, timeout } = req.body;
    // Generate code hash for tracking and security
    const codeHash = crypto.createHash('sha256').update(code.trim()).digest('hex');
    logger.info('Executing Python code:', {
      codeLength: code.length,
      codeHash: codeHash.substring(0, 16), // Log first 16 chars of hash
      hasContext: !!context,
      timeout: timeout || 'default',
      ip: req.ip,
    });
    const result = await pyodideService.executeCode(code, context, timeout);
    const executionTime = Date.now() - startTime;
    // Security logging with complete execution data
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: result.success,
      executionTime,
      codeHash,
      codeLength: code.length,
      executionType: 'structured',
      hasContext: !!context,
      timeout: timeout || 30000,
      error: result.success ? null : result.error,
      timestamp: new Date().toISOString(),
    });
    if (result.success) {
      logger.info('Code execution successful', {
        executionTime,
        codeHash: codeHash.substring(0, 16),
      });
    } else {
      logger.warn('Code execution failed:', {
        error: result.error,
        executionTime,
        codeHash: codeHash.substring(0, 16),
      });
    }
    // Format response according to API contract
    res.json({
      success: result.success,
      data: result.success ? {
        result: result.result,
        stdout: result.stdout,
        stderr: result.stderr,
        executionTime: executionTime
      } : null,
      error: result.success ? null : result.error,
      meta: {
        timestamp: result.timestamp || new Date().toISOString()
      }
    });
  } catch (error) {
    const executionTime = Date.now() - startTime;
    logger.error('Execution endpoint error:', {
      error: error.message,
      executionTime,
    });
    // Log security event for errors
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: false,
      executionTime,
      error: error.message,
    });
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
}
/**
 * Lists all files currently available in Pyodide's virtual filesystem.
 * Provides inventory of uploaded files, generated plots, and temporary files.
 *
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @returns {Promise<void>} Resolves when response is sent
 *
 * @example
 * // GET /api/files/pyodide
 *
 * // Response:
 * // {
 * //   "success": true,
 * //   "data": {
 * //     "files": {
 * //       "/uploads": ["data.csv", "users.xlsx"],
 * //       "/plots/matplotlib": ["chart1.png", "analysis.jpg"],
 * //       "/plots/seaborn": ["correlation.png"],
 * //       "/tmp": ["temp_data.json"]
 * //     },
 * //     "totalFiles": 6
 * //   },
 * //   "error": null,
 * //   "meta": {
 * //     "timestamp": "2025-08-20T10:30:00Z"
 * //   }
 * // }
 *
 * @description
 * Features:
 * - Recursive directory traversal
 * - Organized by directory structure
 * - File count and metadata
 * - Real-time filesystem state
 *
 * Use Cases:
 * - File management interfaces
 * - Debugging filesystem issues
 * - Cleanup and maintenance
 * - API exploration and testing
 *
 * Security Notes:
 * - Only shows Pyodide virtual filesystem
 * - No access to host filesystem
 * - Safe for external API consumption
 */
async function listPyodideFiles(req, res) {
  try {
    const result = await pyodideService.listPyodideFiles();
    // If the service already returns in API contract format, use it directly
    if (result.success !== undefined && result.data !== undefined && result.meta !== undefined) {
      res.json(result);
    } else {
      // Otherwise, wrap it in the API contract format
      res.json({
        success: true,
        data: result,
        error: null,
        meta: {
          timestamp: new Date().toISOString()
        }
      });
    }
  } catch (error) {
    logger.error('Pyodide file list endpoint error:', error);
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
}
/**
 * Deletes a specific file from Pyodide's virtual filesystem.
 * Removes uploaded files, generated plots, or temporary data.
 *
 * @param {Object} req - Express request object
 * @param {Object} req.params - URL parameters
 * @param {string} req.params.filename - Name of file to delete (including path)
 * @param {Object} res - Express response object
 * @returns {Promise<void>} Resolves when response is sent
 *
 * @example
 * // DELETE /api/files/pyodide/plots%2Fmatplotlib%2Fchart.png
 * // (URL encoded: plots/matplotlib/chart.png)
 *
 * // Success response:
 * // {
 * //   "success": true,
 * //   "message": "File deleted successfully",
 * //   "filename": "plots/matplotlib/chart.png",
 * //   "timestamp": "2025-08-20T10:30:00Z"
 * // }
 *
 * // File not found (404):
 * // {
 * //   "success": false,
 * //   "error": "File not found: plots/matplotlib/missing.png",
 * //   "timestamp": "2025-08-20T10:30:00Z"
 * // }
 *
 * // Delete uploaded CSV file:
 * // DELETE /api/files/pyodide/uploads%2Fdata.csv
 *
 * @description
 * HTTP Status Codes:
 * - 200: File successfully deleted
 * - 404: File not found
 * - 500: Server error during deletion
 *
 * Security Features:
 * - Path validation to prevent directory traversal
 * - Only operates within Pyodide virtual filesystem
 * - Logging of all deletion attempts
 *
 * Use Cases:
 * - Cleanup after data analysis
 * - File management in web interfaces
 * - Automated maintenance scripts
 * - Testing and development workflows
 *
 * Path Encoding:
 * - URL encode special characters in filenames
 * - Forward slashes must be encoded as %2F
 * - Spaces encoded as %20 or +
 */
async function deletePyodideFile(req, res) {
  try {
    const filename = req.params.filename;
    const result = await pyodideService.deletePyodideFile(filename);
    // Check if file was not found and return 404
    if (!result.success && result.error && result.error.includes('not found')) {
      return res.status(404).json({
        success: result.success,
        data: null,
        error: result.error,
        meta: {
          timestamp: result.timestamp || new Date().toISOString()
        }
      });
    }
    // Return appropriate status based on success
    const statusCode = result.success ? 200 : 500;
    res.status(statusCode).json({
      success: result.success,
      data: result.success ? {
        message: result.message,
        filename: result.filename
      } : null,
      error: result.success ? null : result.error,
      meta: {
        timestamp: result.timestamp || new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Pyodide file deletion endpoint error:', error);
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
  }
}
module.exports = { executeRaw, executeCode, listPyodideFiles, deletePyodideFile };
