const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
const crypto = require('crypto');

async function executeRaw(req, res) {
  const startTime = Date.now();
  try {
    const code = req.body;
    if (!code || typeof code !== 'string' || !code.trim()) {
      return res.status(400).json({
        success: false,
        error: 'No Python code provided in request body',
        timestamp: new Date().toISOString(),
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
      timestamp: new Date().toISOString()
    });

    if (result.success) {
      logger.info('Raw code execution successful', {
        executionTime,
        codeHash: codeHash.substring(0, 16)
      });
    } else {
      logger.warn('Raw code execution failed:', {
        error: result.error,
        executionTime,
        codeHash: codeHash.substring(0, 16)
      });
    }

    res.json(result);
  } catch (error) {
    const executionTime = Date.now() - startTime;
    logger.error('Raw execution endpoint error:', {
      error: error.message,
      executionTime
    });
    
    // Log security event for errors
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: false,
      executionTime,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

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
      timestamp: new Date().toISOString()
    });

    if (result.success) {
      logger.info('Code execution successful', {
        executionTime,
        codeHash: codeHash.substring(0, 16)
      });
    } else {
      logger.warn('Code execution failed:', {
        error: result.error,
        executionTime,
        codeHash: codeHash.substring(0, 16)
      });
    }

    res.json(result);
  } catch (error) {
    const executionTime = Date.now() - startTime;
    logger.error('Execution endpoint error:', {
      error: error.message,
      executionTime
    });
    
    // Log security event for errors
    logger.security('code_execution', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      success: false,
      executionTime,
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

async function listPyodideFiles(req, res) {
  try {
    const result = await pyodideService.listPyodideFiles();
    res.json(result);
  } catch (error) {
    logger.error('Pyodide file list endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

async function deletePyodideFile(req, res) {
  try {
    const filename = req.params.filename;
    const result = await pyodideService.deletePyodideFile(filename);
    
    // Check if file was not found and return 404
    if (!result.success && result.error && result.error.includes('not found')) {
      return res.status(404).json(result);
    }
    
    // Return appropriate status based on success
    const statusCode = result.success ? 200 : 500;
    res.status(statusCode).json(result);
  } catch (error) {
    logger.error('Pyodide file deletion endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

module.exports = { executeRaw, executeCode, listPyodideFiles, deletePyodideFile };
