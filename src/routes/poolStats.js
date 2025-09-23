const express = require('express');
const router = express.Router();
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');

/**
 * Pool & Memory Statistics Endpoint
 * 
 * Returns detailed memory and process pool statistics using the unified
 * API response contract:
 * {
 *   success: true|false,
 *   data: { ... } | null,
 *   error: string|null,
 *   meta: { timestamp: ISOString }
 * }
 * 
 * Query Parameters:
 *   verbose=1  -> include raw process.memoryUsage() and adaptive sizing details
 * 
 * This endpoint is designed to be inexpensive and safe to call after every
 * test. It does not mutate state; it only reads current metrics.
 */
router.get('/pool-stats', (req, res) => {
  const ts = new Date().toISOString();
  try {
    if (!pyodideService || !pyodideService.processPool) {
      return res.status(503).json({
        success: false,
        data: null,
        error: 'Pyodide service not initialized',
        meta: { timestamp: ts }
      });
    }

    const verbose = req.query.verbose === '1' || req.query.verbose === 'true';
    const poolStats = pyodideService.processPool.getStats();
    const memoryUsage = pyodideService.processPool.getMemoryUsage();

    const data = {
      pool: poolStats,
      memory: memoryUsage,
      uptimeSeconds: process.uptime(),
      adaptive: poolStats.memoryAdaptive,
    };

    if (verbose) {
      data.node = process.memoryUsage();
    }

    res.json({
      success: true,
      data,
      error: null,
      meta: { timestamp: ts }
    });
  } catch (error) {
    logger.error('Failed to retrieve pool stats', {
      component: 'pool-stats-endpoint',
      error: error.message
    });
    res.status(500).json({
      success: false,
      data: null,
      error: 'Failed to retrieve pool stats',
      meta: { timestamp: ts }
    });
  }
});

module.exports = router;
