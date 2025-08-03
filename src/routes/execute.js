/**
 * Python Code Execution Routes
 * 
 * Handles all endpoints related to executing Python code using Pyodide
 */

const express = require('express');
const pyodideService = require('../services/pyodide-service');
const { validateCode, validatePackage } = require('../middleware/validation');
const logger = require('../utils/logger');

const router = express.Router();

/**
 * Execute Python code endpoint
 * POST /api/execute
 */
router.post('/execute', validateCode, async (req, res) => {
  try {
    const { code, context, timeout } = req.body;
    
    logger.info('Executing Python code:', { 
      codeLength: code.length,
      hasContext: !!context,
      timeout: timeout || 'default',
      ip: req.ip
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
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Install Python package endpoint
 * POST /api/install-package
 */
router.post('/install-package', validatePackage, async (req, res) => {
  try {
    const { package: packageName } = req.body;
    
    logger.info('Installing package:', packageName);
    
    const result = await pyodideService.installPackage(packageName);
    
    if (result.success) {
      logger.info('Package installed successfully:', packageName);
    } else {
      logger.warn('Package installation failed:', packageName, result.error);
    }
    
    res.json(result);
    
  } catch (error) {
    logger.error('Package installation endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Get installed packages endpoint
 * GET /api/packages
 */
router.get('/packages', async (req, res) => {
  try {
    const result = await pyodideService.getInstalledPackages();
    res.json(result);
  } catch (error) {
    logger.error('Package list endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Reset Pyodide environment endpoint
 * POST /api/reset
 */
router.post('/reset', async (req, res) => {
  try {
    await pyodideService.reset();
    
    logger.info('Pyodide environment reset successfully');
    
    res.json({
      success: true,
      message: 'Pyodide environment reset successfully',
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    logger.error('Reset endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Get Pyodide status endpoint
 * GET /api/status
 */
router.get('/status', (req, res) => {
  try {
    const status = pyodideService.getStatus();
    res.json(status);
  } catch (error) {
    logger.error('Status check failed:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Health check for Python execution specifically
 * GET /api/health
 */
router.get('/health', async (req, res) => {
  try {
    const status = pyodideService.getStatus();
    
    if (!status.isReady) {
      return res.status(503).json({
        success: false,
        status: 'not_ready',
        message: 'Pyodide is still initializing',
        pyodide: status
      });
    }
    
    // Test basic Python execution
    const testResult = await pyodideService.executeCode('2 + 2');
    
    if (testResult.success && testResult.result === 4) {
      res.json({
        success: true,
        status: 'healthy',
        message: 'Python execution is working correctly',
        pyodide: status,
        testResult: testResult
      });
    } else {
      res.status(500).json({
        success: false,
        status: 'unhealthy',
        message: 'Python execution test failed',
        pyodide: status,
        testResult: testResult
      });
    }
    
  } catch (error) {
    logger.error('Python health check failed:', error);
    res.status(500).json({
      success: false,
      status: 'error',
      error: error.message
    });
  }
});

/**
 * Execute Python code with streaming output (experimental)
 * POST /api/execute-stream
 */
router.post('/execute-stream', validateCode, async (req, res) => {
  try {
    const { code, context, timeout } = req.body;
    
    // Set up Server-Sent Events
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Cache-Control'
    });
    
    // Send initial status
    res.write(`data: ${JSON.stringify({ type: 'status', message: 'Starting execution...' })}\n\n`);
    
    try {
      const result = await pyodideService.executeCode(code, context, timeout);
      
      // Send output if available
      if (result.stdout) {
        res.write(`data: ${JSON.stringify({ type: 'stdout', content: result.stdout })}\n\n`);
      }
      
      if (result.stderr) {
        res.write(`data: ${JSON.stringify({ type: 'stderr', content: result.stderr })}\n\n`);
      }
      
      // Send final result
      res.write(`data: ${JSON.stringify({ type: 'result', ...result })}\n\n`);
      res.write(`data: ${JSON.stringify({ type: 'complete' })}\n\n`);
      
    } catch (executionError) {
      res.write(`data: ${JSON.stringify({ type: 'error', error: executionError.message })}\n\n`);
    }
    
    res.end();
    
  } catch (error) {
    logger.error('Streaming execution error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get execution statistics
 * GET /api/stats
 */
router.get('/stats', (req, res) => {
  try {
    const stats = {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      pyodide: pyodideService.getStatus(),
      timestamp: new Date().toISOString()
    };
    
    res.json(stats);
  } catch (error) {
    logger.error('Stats endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;