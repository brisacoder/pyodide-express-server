/**
 * @swagger
 * components:
 *   schemas:
 *     CrashReport:
 *       type: object
 *       properties:
 *         crashId:
 *           type: string
 *           description: Unique identifier for the crash report
 *           example: "crash-1640995200000-abc123"
 *         timestamp:
 *           type: string
 *           format: date-time
 *           description: When the crash occurred
 *           example: "2024-01-01T12:00:00.000Z"
 *         type:
 *           type: string
 *           description: Type of crash
 *           enum: [uncaughtException, unhandledRejection, application_error, graceful_shutdown]
 *           example: "uncaughtException"
 *         error:
 *           type: object
 *           properties:
 *             name:
 *               type: string
 *               description: Error type name
 *               example: "TypeError"
 *             message:
 *               type: string
 *               description: Error message
 *               example: "Cannot read property 'test' of undefined"
 *             stack:
 *               type: string
 *               description: Full stack trace
 *         process:
 *           type: object
 *           description: Process information at time of crash
 *         system:
 *           type: object
 *           description: System information at time of crash
 *         memory:
 *           type: object
 *           description: Memory usage information
 *         pyodide:
 *           type: object
 *           description: Pyodide service state at time of crash
 *
 *     CrashSummary:
 *       type: object
 *       properties:
 *         id:
 *           type: string
 *           description: Crash report ID
 *           example: "crash-1640995200000-abc123"
 *         timestamp:
 *           type: string
 *           format: date-time
 *           description: When the crash occurred
 *         type:
 *           type: string
 *           description: Type of crash
 *         error:
 *           type: string
 *           description: Brief error description
 *           example: "TypeError: Cannot read property 'test' of undefined"
 *         file:
 *           type: string
 *           description: Crash report filename
 *         size:
 *           type: number
 *           description: File size in bytes
 *         mtime:
 *           type: string
 *           format: date-time
 *           description: File modification time
 *
 *     CrashStats:
 *       type: object
 *       properties:
 *         totalCrashes:
 *           type: number
 *           description: Total number of crash reports
 *           example: 15
 *         recentCrashes:
 *           type: number
 *           description: Number of crashes in last 24 hours
 *           example: 2
 *         crashTypes:
 *           type: object
 *           description: Breakdown of crash types
 *           additionalProperties:
 *             type: number
 *           example:
 *             uncaughtException: 10
 *             unhandledRejection: 5
 *         oldestCrash:
 *           type: string
 *           format: date-time
 *           description: Timestamp of oldest crash
 *           nullable: true
 *         newestCrash:
 *           type: string
 *           format: date-time
 *           description: Timestamp of newest crash
 *           nullable: true
 *
 * tags:
 *   - name: Crash Reports
 *     description: Debug and crash reporting endpoints for production monitoring
 */
const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const router = express.Router();
/**
 * @swagger
 * /api/debug/crash-reports:
 *   get:
 *     summary: List all crash reports
 *     description: Retrieve a list of all available crash reports, sorted by timestamp (newest first)
 *     tags: [Crash Reports]
 *     responses:
 *       200:
 *         description: Successfully retrieved crash reports list
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 crashes:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/CrashSummary'
 *                 count:
 *                   type: number
 *                   description: Total number of crash reports
 *                   example: 5
 *             examples:
 *               with_crashes:
 *                 summary: Response with crash reports
 *                 value:
 *                   success: true
 *                   crashes:
 *                     - id: "crash-1640995200000-abc123"
 *                       timestamp: "2024-01-01T12:00:00.000Z"
 *                       type: "uncaughtException"
 *                       error: "TypeError: Cannot read property 'test' of undefined"
 *                       file: "crash-1640995200000-abc123.json"
 *                       size: 2048
 *                   count: 1
 *               no_crashes:
 *                 summary: Response with no crashes
 *                 value:
 *                   success: true
 *                   crashes: []
 *                   message: "No crash reports directory found"
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Failed to list crash reports"
 *                 details:
 *                   type: string
 *                   example: "Permission denied accessing crash reports directory"
 */
// Get crash reports list
router.get('/crash-reports', async (req, res) => {
  try {
    const crashDir = path.join(process.cwd(), 'crash-reports');
    try {
      await fs.access(crashDir);
    } catch {
      return res.json({
        success: true,
        crashes: [],
        message: 'No crash reports directory found',
      });
    }
    const files = await fs.readdir(crashDir);
    const crashFiles = files.filter((f) => f.endsWith('.json') && f.startsWith('crash-'));
    const crashes = await Promise.all(
      crashFiles.map(async (file) => {
        try {
          const filePath = path.join(crashDir, file);
          const stat = await fs.stat(filePath);
          const content = await fs.readFile(filePath, 'utf8');
          const crash = JSON.parse(content);
          return {
            id: crash.crashId,
            timestamp: crash.timestamp,
            type: crash.type,
            error: crash.error.name + ': ' + crash.error.message,
            file: file,
            size: stat.size,
            mtime: stat.mtime,
          };
        } catch {
          return null;
        }
      })
    );
    const validCrashes = crashes.filter((c) => c !== null);
    validCrashes.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    res.json({
      success: true,
      crashes: validCrashes,
      count: validCrashes.length,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to list crash reports',
      details: error.message,
    });
  }
});
/**
 * @swagger
 * /api/debug/crash-reports/{crashId}:
 *   get:
 *     summary: Get detailed crash report
 *     description: Retrieve detailed information about a specific crash report including full stack trace, system state, and Pyodide context
 *     tags: [Crash Reports]
 *     parameters:
 *       - in: path
 *         name: crashId
 *         required: true
 *         description: Unique identifier of the crash report
 *         schema:
 *           type: string
 *           example: "crash-1640995200000-abc123"
 *     responses:
 *       200:
 *         description: Successfully retrieved crash report details
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 crash:
 *                   $ref: '#/components/schemas/CrashReport'
 *       404:
 *         description: Crash report not found
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Crash report not found"
 *                 crashId:
 *                   type: string
 *                   example: "crash-1640995200000-abc123"
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Failed to read crash report"
 */
// Get specific crash report
router.get('/crash-reports/:crashId', async (req, res) => {
  try {
    const { crashId } = req.params;
    const crashDir = path.join(process.cwd(), 'crash-reports');
    const crashFile = path.join(crashDir, `${crashId}.json`);
    try {
      const content = await fs.readFile(crashFile, 'utf8');
      const crash = JSON.parse(content);
      res.json({ success: true, crash });
    } catch {
      res.status(404).json({
        success: false,
        error: 'Crash report not found',
        crashId,
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to read crash report',
      details: error.message,
    });
  }
});
/**
 * @swagger
 * /api/debug/crash-reports/{crashId}/summary:
 *   get:
 *     summary: Get human-readable crash summary
 *     description: Retrieve a human-readable text summary of a crash report, formatted for easy reading and debugging
 *     tags: [Crash Reports]
 *     parameters:
 *       - in: path
 *         name: crashId
 *         required: true
 *         description: Unique identifier of the crash report
 *         schema:
 *           type: string
 *           example: "crash-1640995200000-abc123"
 *     responses:
 *       200:
 *         description: Successfully retrieved crash summary
 *         content:
 *           text/plain:
 *             schema:
 *               type: string
 *               example: |
 *                 CRASH REPORT SUMMARY
 *                 ====================
 *                 Crash ID: crash-1640995200000-abc123
 *                 Timestamp: 2024-01-01T12:00:00.000Z
 *                 Type: uncaughtException
 *
 *                 ERROR DETAILS
 *                 =============
 *                 Name: TypeError
 *                 Message: Cannot read property 'test' of undefined
 *
 *                 STACK TRACE
 *                 ===========
 *                 TypeError: Cannot read property 'test' of undefined
 *                     at /app/src/server.js:45:12
 *                     at ...
 *       404:
 *         description: Crash summary not found
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Crash summary not found"
 *                 crashId:
 *                   type: string
 *                   example: "crash-1640995200000-abc123"
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Failed to read crash summary"
 */
// Get crash report summary (human readable)
router.get('/crash-reports/:crashId/summary', async (req, res) => {
  try {
    const { crashId } = req.params;
    const crashDir = path.join(process.cwd(), 'crash-reports');
    const summaryFile = path.join(crashDir, `${crashId}-summary.txt`);
    try {
      const content = await fs.readFile(summaryFile, 'utf8');
      res.set('Content-Type', 'text/plain');
      res.send(content);
    } catch {
      res.status(404).json({
        success: false,
        error: 'Crash summary not found',
        crashId,
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to read crash summary',
      details: error.message,
    });
  }
});
/**
 * @swagger
 * /api/debug/crash-reports/{crashId}:
 *   delete:
 *     summary: Delete crash report
 *     description: Remove a specific crash report and its associated summary file from the system
 *     tags: [Crash Reports]
 *     parameters:
 *       - in: path
 *         name: crashId
 *         required: true
 *         description: Unique identifier of the crash report to delete
 *         schema:
 *           type: string
 *           example: "crash-1640995200000-abc123"
 *     responses:
 *       200:
 *         description: Successfully deleted crash report
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
 *                   example: "Crash report deleted"
 *                 crashId:
 *                   type: string
 *                   example: "crash-1640995200000-abc123"
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Failed to delete crash report"
 *                 details:
 *                   type: string
 *                   example: "Permission denied"
 */
// Delete crash report
router.delete('/crash-reports/:crashId', async (req, res) => {
  try {
    const { crashId } = req.params;
    const crashDir = path.join(process.cwd(), 'crash-reports');
    const crashFile = path.join(crashDir, `${crashId}.json`);
    const summaryFile = path.join(crashDir, `${crashId}-summary.txt`);
    try {
      await fs.unlink(crashFile);
    } catch {
      // File might not exist
    }
    try {
      await fs.unlink(summaryFile);
    } catch {
      // Summary might not exist
    }
    res.json({
      success: true,
      message: 'Crash report deleted',
      crashId,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to delete crash report',
      details: error.message,
    });
  }
});
/**
 * @swagger
 * /api/debug/crash-stats:
 *   get:
 *     summary: Get crash statistics
 *     description: Retrieve comprehensive statistics about crash reports including totals, recent activity, crash types breakdown, and timeline information
 *     tags: [Crash Reports]
 *     responses:
 *       200:
 *         description: Successfully retrieved crash statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 stats:
 *                   $ref: '#/components/schemas/CrashStats'
 *             examples:
 *               with_crashes:
 *                 summary: Statistics with crash data
 *                 value:
 *                   success: true
 *                   stats:
 *                     totalCrashes: 15
 *                     recentCrashes: 2
 *                     crashTypes:
 *                       uncaughtException: 10
 *                       unhandledRejection: 5
 *                     oldestCrash: "2024-01-01T12:00:00.000Z"
 *                     newestCrash: "2024-01-02T14:30:00.000Z"
 *               no_crashes:
 *                 summary: Statistics with no crashes
 *                 value:
 *                   success: true
 *                   stats:
 *                     totalCrashes: 0
 *                     recentCrashes: 0
 *                     crashTypes: {}
 *                     oldestCrash: null
 *                     newestCrash: null
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: false
 *                 error:
 *                   type: string
 *                   example: "Failed to get crash statistics"
 *                 details:
 *                   type: string
 *                   example: "File system error"
 */
// Crash statistics
router.get('/crash-stats', async (req, res) => {
  try {
    const crashDir = path.join(process.cwd(), 'crash-reports');
    try {
      await fs.access(crashDir);
    } catch {
      return res.json({
        success: true,
        stats: {
          totalCrashes: 0,
          recentCrashes: 0,
          crashTypes: {},
          oldestCrash: null,
          newestCrash: null,
        },
      });
    }
    const files = await fs.readdir(crashDir);
    const crashFiles = files.filter((f) => f.endsWith('.json') && f.startsWith('crash-'));
    if (crashFiles.length === 0) {
      return res.json({
        success: true,
        stats: {
          totalCrashes: 0,
          recentCrashes: 0,
          crashTypes: {},
          oldestCrash: null,
          newestCrash: null,
        },
      });
    }
    const crashes = await Promise.all(
      crashFiles.map(async (file) => {
        try {
          const content = await fs.readFile(path.join(crashDir, file), 'utf8');
          return JSON.parse(content);
        } catch {
          return null;
        }
      })
    );
    const validCrashes = crashes.filter((c) => c !== null);
    const now = new Date();
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const recentCrashes = validCrashes.filter((c) => new Date(c.timestamp) > oneDayAgo);
    const crashTypes = {};
    validCrashes.forEach((crash) => {
      const type = crash.type || 'unknown';
      crashTypes[type] = (crashTypes[type] || 0) + 1;
    });
    const timestamps = validCrashes.map((c) => new Date(c.timestamp));
    timestamps.sort();
    res.json({
      success: true,
      stats: {
        totalCrashes: validCrashes.length,
        recentCrashes: recentCrashes.length,
        crashTypes,
        oldestCrash: timestamps.length > 0 ? timestamps[0].toISOString() : null,
        newestCrash: timestamps.length > 0 ? timestamps[timestamps.length - 1].toISOString() : null,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to get crash statistics',
      details: error.message,
    });
  }
});
module.exports = router;
