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
 * @swagger
 * /api/execute:
 *   post:
 *     summary: Execute Python code
 *     description: Execute Python code in the Pyodide environment with optional context variables and custom timeout
 *     tags: [Python Execution]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ExecuteRequest'
 *           examples:
 *             simple_calculation:
 *               summary: Simple calculation
 *               value:
 *                 code: |
 *                   result = 2 + 2
 *                   print(f"2 + 2 = {result}")
 *                   result
 *             pandas_example:
 *               summary: Pandas data analysis
 *               value:
 *                 code: |
 *                   import pandas as pd
 *                   import numpy as np
 *                   
 *                   # Create sample dataset
 *                   data = {
 *                       'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
 *                       'age': [25, 30, 35, 28],
 *                       'score': [85, 92, 78, 96]
 *                   }
 *                   df = pd.DataFrame(data)
 *                   
 *                   print("Dataset:")
 *                   print(df)
 *                   
 *                   # Calculate statistics
 *                   stats = {
 *                       'mean_age': df['age'].mean(),
 *                       'max_score': df['score'].max(),
 *                       'correlation': df[['age', 'score']].corr().iloc[0,1]
 *                   }
 *                   
 *                   print(f"\nStatistics: {stats}")
 *                   stats
 *             with_context:
 *               summary: Using context variables
 *               value:
 *                 code: |
 *                   print(f"Hello {name}!")
 *                   result = value * multiplier
 *                   print(f"{value} * {multiplier} = {result}")
 *                   result
 *                 context:
 *                   name: "Alice"
 *                   value: 42
 *                   multiplier: 2
 *     responses:
 *       200:
 *         description: Code execution completed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ExecuteResponse'
 *             examples:
 *               success:
 *                 summary: Successful execution
 *                 value:
 *                   success: true
 *                   result: 4
 *                   stdout: "2 + 2 = 4\n"
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               error:
 *                 summary: Execution error
 *                 value:
 *                   success: false
 *                   error: "NameError: name 'undefined_var' is not defined"
 *                   stdout: ""
 *                   stderr: "Traceback (most recent call last)..."
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
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
 * @swagger
 * /api/install-package:
 *   post:
 *     summary: Install Python package
 *     description: Install a Python package using micropip in the Pyodide environment
 *     tags: [Package Management]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/PackageRequest'
 *           examples:
 *             install_requests:
 *               summary: Install requests library
 *               value:
 *                 package: "requests"
 *             install_scikit_learn:
 *               summary: Install scikit-learn
 *               value:
 *                 package: "scikit-learn"
 *     responses:
 *       200:
 *         description: Package installation completed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ExecuteResponse'
 *             examples:
 *               success:
 *                 summary: Package installed successfully
 *                 value:
 *                   success: true
 *                   result: "Successfully installed requests"
 *                   stdout: "Installing requests...\nPackage installed successfully"
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               failure:
 *                 summary: Package installation failed
 *                 value:
 *                   success: false
 *                   error: "Installation failed: Package not found"
 *                   stdout: ""
 *                   stderr: "ERROR: Could not find package"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
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
 * @swagger
 * /api/packages:
 *   get:
 *     summary: Get installed packages
 *     description: Retrieve a list of all installed Python packages in the Pyodide environment
 *     tags: [Package Management]
 *     responses:
 *       200:
 *         description: Package list retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               allOf:
 *                 - $ref: '#/components/schemas/ExecuteResponse'
 *                 - type: object
 *                   properties:
 *                     result:
 *                       type: object
 *                       properties:
 *                         python_version:
 *                           type: string
 *                           example: "3.11.3 (main, Apr  8 2023, 17:45:25) [Clang 15.0.7 ]"
 *                         installed_packages:
 *                           type: array
 *                           items:
 *                             type: string
 *                           example: ["numpy", "pandas", "matplotlib", "requests", "micropip"]
 *                         total_packages:
 *                           type: integer
 *                           example: 125
 *             examples:
 *               success:
 *                 summary: Package list retrieved
 *                 value:
 *                   success: true
 *                   result:
 *                     python_version: "3.11.3 (main, Apr  8 2023, 17:45:25) [Clang 15.0.7 ]"
 *                     installed_packages: ["numpy", "pandas", "matplotlib", "requests", "micropip"]
 *                     total_packages: 125
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       500:
 *         $ref: '#/components/responses/InternalError'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
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
 * @swagger
 * /api/reset:
 *   post:
 *     summary: Reset Pyodide environment
 *     description: Reset the Python environment by clearing user-defined variables while keeping system packages
 *     tags: [Python Execution]
 *     responses:
 *       200:
 *         description: Environment reset successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: "Pyodide environment reset successfully"
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *             examples:
 *               success:
 *                 summary: Reset completed
 *                 value:
 *                   success: true
 *                   message: "Pyodide environment reset successfully"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       500:
 *         $ref: '#/components/responses/InternalError'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
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
 * @swagger
 * /api/status:
 *   get:
 *     summary: Get Pyodide status
 *     description: Check the current status and readiness of the Pyodide Python environment
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Status retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/StatusResponse'
 *             examples:
 *               ready:
 *                 summary: Pyodide ready
 *                 value:
 *                   isReady: true
 *                   initialized: true
 *                   executionTimeout: 30000
 *                   version: "0.28.0"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               not_ready:
 *                 summary: Pyodide initializing
 *                 value:
 *                   isReady: false
 *                   initialized: false
 *                   executionTimeout: 30000
 *                   version: "unknown"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       500:
 *         $ref: '#/components/responses/InternalError'
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
 * @swagger
 * /api/health:
 *   get:
 *     summary: Python execution health check
 *     description: Perform a comprehensive health check including a test Python execution
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Health check passed
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 status:
 *                   type: string
 *                   enum: [healthy, not_ready, unhealthy, error]
 *                 message:
 *                   type: string
 *                 pyodide:
 *                   $ref: '#/components/schemas/StatusResponse'
 *                 testResult:
 *                   $ref: '#/components/schemas/ExecuteResponse'
 *             examples:
 *               healthy:
 *                 summary: System healthy
 *                 value:
 *                   success: true
 *                   status: "healthy"
 *                   message: "Python execution is working correctly"
 *                   pyodide:
 *                     isReady: true
 *                     initialized: true
 *                     executionTimeout: 30000
 *                     version: "0.28.0"
 *                     timestamp: "2024-01-15T10:30:00.000Z"
 *                   testResult:
 *                     success: true
 *                     result: 4
 *                     stdout: ""
 *                     stderr: ""
 *                     timestamp: "2024-01-15T10:30:00.000Z"
 *       503:
 *         description: Service not ready
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 status:
 *                   type: string
 *                   example: "not_ready"
 *                 message:
 *                   type: string
 *                   example: "Pyodide is still initializing"
 *                 pyodide:
 *                   $ref: '#/components/schemas/StatusResponse'
 *       500:
 *         $ref: '#/components/responses/InternalError'
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
 * @swagger
 * /api/execute-stream:
 *   post:
 *     summary: Execute Python code with streaming output (experimental)
 *     description: Execute Python code and stream the output in real-time using Server-Sent Events
 *     tags: [Python Execution]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ExecuteRequest'
 *     responses:
 *       200:
 *         description: Streaming execution started
 *         content:
 *           text/event-stream:
 *             schema:
 *               type: string
 *               description: Server-Sent Events stream
 *               example: |
 *                 data: {"type":"status","message":"Starting execution..."}
 *                 
 *                 data: {"type":"stdout","content":"Hello World\n"}
 *                 
 *                 data: {"type":"result","success":true,"result":42,"stdout":"Hello World\n","stderr":"","timestamp":"2024-01-15T10:30:00.000Z"}
 *                 
 *                 data: {"type":"complete"}
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       500:
 *         $ref: '#/components/responses/InternalError'
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
 * @swagger
 * /api/stats:
 *   get:
 *     summary: Get execution statistics
 *     description: Retrieve server statistics including uptime, memory usage, and Pyodide status
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 uptime:
 *                   type: number
 *                   description: Server uptime in seconds
 *                   example: 3600.25
 *                 memory:
 *                   type: object
 *                   properties:
 *                     rss:
 *                       type: number
 *                       description: Resident Set Size in bytes
 *                     heapTotal:
 *                       type: number
 *                       description: Total heap size in bytes
 *                     heapUsed:
 *                       type: number
 *                       description: Used heap size in bytes
 *                     external:
 *                       type: number
 *                       description: External memory usage in bytes
 *                     arrayBuffers:
 *                       type: number
 *                       description: ArrayBuffer memory usage in bytes
 *                 pyodide:
 *                   $ref: '#/components/schemas/StatusResponse'
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *             examples:
 *               stats:
 *                 summary: Server statistics
 *                 value:
 *                   uptime: 3600.25
 *                   memory:
 *                     rss: 157286400
 *                     heapTotal: 134217728
 *                     heapUsed: 95367896
 *                     external: 12345678
 *                     arrayBuffers: 8192
 *                   pyodide:
 *                     isReady: true
 *                     initialized: true
 *                     executionTimeout: 30000
 *                     version: "0.28.0"
 *                     timestamp: "2024-01-15T10:30:00.000Z"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       500:
 *         $ref: '#/components/responses/InternalError'
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