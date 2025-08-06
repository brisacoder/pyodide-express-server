const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');

async function executeRaw(req, res) {
  try {
    const code = req.body;
    if (!code || typeof code !== 'string' || !code.trim()) {
      return res.status(400).json({
        success: false,
        error: 'No Python code provided in request body',
        timestamp: new Date().toISOString(),
      });
    }

    logger.info('Executing raw Python code:', {
      codeLength: code.length,
      ip: req.ip,
      contentType: req.get('Content-Type'),
    });

    const result = await pyodideService.executeCode(code);

    if (result.success) {
      logger.info('Raw code execution successful');
    } else {
      logger.warn('Raw code execution failed:', result.error);
    }

    res.json(result);
  } catch (error) {
    logger.error('Raw execution endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}

async function executeCode(req, res) {
  try {
    const { code, context, timeout } = req.body;
    logger.info('Executing Python code:', {
      codeLength: code.length,
      hasContext: !!context,
      timeout: timeout || 'default',
      ip: req.ip,
    });

    const result = await pyodideService.executeCode(code, context, timeout);

    if (result.success) {
      logger.info('Code execution successful');
    } else {
      logger.warn('Code execution failed:', result.error);
    }

    res.json(result);
  } catch (error) {
    logger.error('Execution endpoint error:', error);
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
    res.json(result);
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
