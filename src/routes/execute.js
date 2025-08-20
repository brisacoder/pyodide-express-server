/**
 * Python Code Execution Routes
 * 
 * Handles all endpoints related to executing Python code using Pyodide
 */

const express = require('express');
const pyodideService = require('../services/pyodide-service');
const { validateCode, validatePackage } = require('../middleware/validation');
const logger = require('../utils/logger');
const { executeCode, listPyodideFiles, deletePyodideFile } = require('../controllers/executeController');

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
router.post('/execute', validateCode, executeCode);


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
    // Installation is performed via ``micropip`` inside the Pyodide runtime.
    // Only packages built for WebAssembly can be fetched successfully.
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
    // Query the Pyodide runtime for modules that are already loaded.
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
    // Reset interpreter state to a clean slate.
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
    // Query the in-memory service for its readiness and version information.
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
    // Start by checking whether the interpreter has completed initialization.
    const status = pyodideService.getStatus();
    
    if (!status.isReady) {
      return res.status(503).json({
        success: false,
        status: 'not_ready',
        message: 'Pyodide is still initializing',
        pyodide: status
      });
    }
    
    // Test basic Python execution to verify the runtime actually works.
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
      // Execute code and stream the resulting output chunks back to the client
      // using Server-Sent Events.
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
 *     summary: Get enhanced execution statistics (backward compatible)
 *     description: |
 *       Retrieve comprehensive server statistics including:
 *       - Legacy system info (uptime, memory, Pyodide status) for backward compatibility
 *       - Enhanced execution statistics (success rates, error tracking, IP monitoring)
 *       - Security logging data and performance metrics
 *       
 *       **Note**: This endpoint maintains backward compatibility while providing enhanced features.
 *       For dashboard-specific features, use `/api/dashboard/stats` instead.
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Enhanced statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 uptime:
 *                   type: number
 *                   description: Server uptime in seconds (legacy compatibility)
 *                   example: 3600.25
 *                 memory:
 *                   type: object
 *                   description: Memory usage statistics (legacy compatibility)
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
 *                   description: Pyodide interpreter status (legacy compatibility)
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                   description: Response timestamp (legacy compatibility)
 *                 executionStats:
 *                   type: object
 *                   description: Enhanced execution statistics and security monitoring
 *                   properties:
 *                     overview:
 *                       $ref: '#/components/schemas/StatisticsOverview'
 *                     recent:
 *                       $ref: '#/components/schemas/StatisticsRecent'
 *                     topIPs:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/IPStatistic'
 *                     topErrors:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/ErrorStatistic'
 *                     userAgents:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/UserAgentStatistic'
 *                     hourlyTrend:
 *                       type: array
 *                       items:
 *                         type: integer
 *                       description: Hourly execution counts for last 24 hours
 *             examples:
 *               enhanced_stats:
 *                 summary: Enhanced server statistics with security monitoring
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
 *                     version: "0.28.2"
 *                     timestamp: "2025-01-26T10:30:00.000Z"
 *                   timestamp: "2025-01-26T10:30:00.000Z"
 *                   executionStats:
 *                     overview:
 *                       totalExecutions: 150
 *                       successRate: "94.7"
 *                       averageExecutionTime: 1250
 *                       uptimeSeconds: 3600
 *                       uptimeHuman: "1h 0m 0s"
 *                     recent:
 *                       lastHourExecutions: 25
 *                       recentSuccessRate: "96.0"
 *                       packagesInstalled: 8
 *                       filesUploaded: 3
 *                     topIPs:
 *                       - ip: "127.0.0.1"
 *                         count: 45
 *                     topErrors:
 *                       - error: "SyntaxError"
 *                         count: 5
 *                     userAgents:
 *                       - agent: "Mozilla/5.0"
 *                         count: 120
 *                     hourlyTrend: [0, 0, 1, 3, 5, 8, 12, 15, 20, 25, 30, 28, 25, 22, 18, 15, 12, 8, 5, 3, 2, 1, 0, 0]
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.get('/stats', (req, res) => {
  try {
    // Get enhanced statistics from logger
    const enhancedStats = logger.getStats();
    
    // Maintain backward compatibility with original format
    const stats = {
      // Legacy top-level fields for existing tests (MUST be at top level)
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      pyodide: pyodideService.getStatus(),
      timestamp: new Date().toISOString(),
      
      // Enhanced statistics (new fields) - add without conflicting
      executionStats: {
        overview: enhancedStats.overview,
        recent: enhancedStats.recent,
        topIPs: enhancedStats.topIPs,
        topErrors: enhancedStats.topErrors,
        userAgents: enhancedStats.userAgents,
        hourlyTrend: enhancedStats.hourlyTrend
      }
    };

    logger.info('Enhanced statistics retrieved', {
      component: 'stats-endpoint',
      totalExecutions: enhancedStats.overview.totalExecutions,
      successRate: enhancedStats.overview.successRate
    });

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