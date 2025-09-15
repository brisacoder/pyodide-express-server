/**
 * Pyodide Internal Routes
 *
 * Handles all internal endpoints related to Pyodide virtual filesystem operations
 */
const express = require('express');
const fs = require('fs');
const path = require('path');
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
const config = require('../config');

const router = express.Router();

// Get upload directory from environment or default
const UPLOAD_DIR = path.resolve(path.join(config.pyodideDataDir, config.pyodideBases.uploads.urlBase));

/**
 * @swagger
 * /api/pyodide-files:
 *   get:
 *     summary: List files in Pyodide virtual filesystem
 *     description: Get a list of all files currently loaded in Pyodide's virtual filesystem (in-memory)
 *     tags: [Pyodide Internal]
 *     responses:
 *       200:
 *         description: Pyodide file list retrieved successfully
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
 *                         success:
 *                           type: boolean
 *                         files:
 *                           type: array
 *                           items:
 *                             type: object
 *                             properties:
 *                               name:
 *                                 type: string
 *                                 example: "data.csv"
 *                               size:
 *                                 type: number
 *                                 example: 1024
 *                               modified:
 *                                 type: number
 *                                 description: Unix timestamp
 *                                 example: 1691234567
 *                         count:
 *                           type: number
 *                           example: 2
 *             examples:
 *               success:
 *                 summary: Files in Pyodide filesystem
 *                 value:
 *                   success: true
 *                   result:
 *                     success: true
 *                     files: [
 *                       {
 *                         name: "data.csv",
 *                         size: 1024,
 *                         modified: 1691234567
 *                       },
 *                       {
 *                         name: "uploaded_file.csv",
 *                         size: 2048,
 *                         modified: 1691234600
 *                       }
 *                     ]
 *                     count: 2
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               empty:
 *                 summary: No files in Pyodide filesystem
 *                 value:
 *                   success: true
 *                   result:
 *                     success: true
 *                     files: []
 *                     count: 0
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.get('/pyodide-files', async (req, res) => {
  try {
    const result = await pyodideService.listPyodideFiles();
    // Parse the Python result if it's a string, otherwise use as-is
    let parsedResult = result;
    if (result.success && result.result) {
      if (typeof result.result === 'string') {
        try {
          // Convert Python dictionary string to JSON
          let jsonString = result.result
            .replace(/'/g, '"') // Single quotes to double quotes
            .replace(/True/g, 'true') // Python True to JSON true
            .replace(/False/g, 'false'); // Python False to JSON false
          parsedResult = {
            ...result,
            result: JSON.parse(jsonString),
          };
        } catch (parseError) {
          logger.warn('Could not parse Pyodide files result:', parseError.message);
        }
      } else {
        // Result is already a JavaScript object
        parsedResult = result;
      }
    }
    logger.info('Listed Pyodide files:', {
      success: parsedResult.success,
      fileCount: parsedResult.result?.count || 0,
    });
    res.json(parsedResult);
  } catch (error) {
    logger.error('Pyodide file list endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /api/pyodide-files/{filename}:
 *   delete:
 *     summary: Delete file from Pyodide filesystem
 *     description: Delete a specific file from Pyodide's virtual filesystem (in-memory)
 *     tags: [Pyodide Internal]
 *     parameters:
 *       - in: path
 *         name: filename
 *         required: true
 *         schema:
 *           type: string
 *         description: Name of the file to delete from Pyodide
 *         example: "data.csv"
 *     responses:
 *       200:
 *         description: File deletion attempted
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ExecuteResponse'
 *             examples:
 *               success:
 *                 summary: File deleted successfully
 *                 value:
 *                   success: true
 *                   result: "File data.csv deleted successfully"
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               not_found:
 *                 summary: File not found
 *                 value:
 *                   success: true
 *                   result: "File data.csv not found"
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *               error:
 *                 summary: Deletion error
 *                 value:
 *                   success: true
 *                   result: "Error deleting data.csv: Permission denied"
 *                   stdout: ""
 *                   stderr: ""
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.delete('/pyodide-files/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    // Security check - prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      logger.warn(`Attempted path traversal in pyodide file deletion: ${filename}`);
      return res.status(400).json({
        success: false,
        error: 'Invalid filename',
        timestamp: new Date().toISOString(),
      });
    }
    logger.info(`Attempting to delete Pyodide file: ${filename}`);
    const result = await pyodideService.deletePyodideFile(filename);
    logger.info('Pyodide file deletion result:', {
      filename: filename,
      success: result.success,
      result: result.result,
    });
    if (!result.success && result.error && result.error.includes('not found')) {
      return res.status(404).json(result);
    }
    res.json(result);
  } catch (error) {
    logger.error('Pyodide file deletion endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /api/extract-plots:
 *   post:
 *     summary: Extract virtual plot files to real filesystem
 *     description: Extract plot files from Pyodide's virtual filesystem to the real plots directory
 *     tags: [Pyodide Internal]
 *     responses:
 *       200:
 *         description: Plot files extracted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   description: Whether the extraction was successful
 *                 extracted_files:
 *                   type: array
 *                   items:
 *                     type: string
 *                   description: List of file paths that were extracted
 *                 count:
 *                   type: integer
 *                   description: Number of files extracted
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *       500:
 *         description: Server error during extraction
 */
router.post('/extract-plots', async (req, res) => {
  try {
    const extractedFiles = await pyodideService.extractAllPlotFiles();
    res.json({
      success: true,
      extracted_files: extractedFiles,
      count: extractedFiles.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    logger.error('Error extracting plot files:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

/**
 * @swagger
 * /api/clear-all-files:
 *   post:
 *     summary: Clear all uploaded files and reset Pyodide filesystem
 *     description: Removes all uploaded files from server storage and clears the Pyodide virtual filesystem
 *     tags: [Pyodide Internal]
 *     responses:
 *       200:
 *         description: All files cleared successfully
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
 *                   example: "All files cleared successfully"
 *                 cleared:
 *                   type: object
 *                   properties:
 *                     uploaded_files:
 *                       type: number
 *                       example: 5
 *                     pyodide_files:
 *                       type: number
 *                       example: 3
 *       500:
 *         description: Error clearing files
 */
router.post('/clear-all-files', async (req, res) => {
  try {
    let uploadedFilesCleared = 0;
    let pyodideFilesCleared = 0;
    // Clear uploaded files from server storage
    if (fs.existsSync(UPLOAD_DIR)) {
      const files = fs.readdirSync(UPLOAD_DIR);
      for (const file of files) {
        const filePath = path.join(UPLOAD_DIR, file);
        if (fs.statSync(filePath).isFile()) {
          fs.unlinkSync(filePath);
          uploadedFilesCleared++;
        }
      }
    }
    // Clear Pyodide virtual filesystem
    try {
      const result = await pyodideService.executeCode(`
import os
import json
from pathlib import Path

# Get list of files from /home/pyodide/uploads directory
files_before = []
uploads_dir = Path('/home/pyodide/uploads')
try:
    if uploads_dir.exists():
        files_before = [f.name for f in uploads_dir.iterdir() if f.is_file()]
        files_before = [f for f in files_before if not f.startswith('_') and f != 'tmp']
except:
    pass

# Clear user files from /uploads directory (keep system files)
cleared_count = 0
for filename in files_before:
    try:
        if not filename.startswith('_') and filename != 'tmp':
            file_path = uploads_dir / filename
            if file_path.exists():
                file_path.unlink()
                cleared_count += 1
    except:
        pass

result = {
    'files_before': files_before,
    'cleared': cleared_count
}
result
      `);
      if (result.success && result.result) {
        pyodideFilesCleared = result.result.cleared;
      }
    } catch (pyodideError) {
      logger.warn('Error clearing Pyodide files:', pyodideError.message);
      // Continue anyway, don't fail the whole operation
    }
    logger.info('Files cleared successfully', {
      uploadedFiles: uploadedFilesCleared,
      pyodideFiles: pyodideFilesCleared,
    });
    res.json({
      success: true,
      data: {
        message: 'All files cleared successfully',
        cleared: {
          uploaded_files: uploadedFilesCleared,
          pyodide_files: pyodideFilesCleared,
        }
      },
      error: null,
      meta: {
        timestamp: new Date().toISOString(),
      }
    });
  } catch (error) {
    logger.error('Error clearing files:', error);
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString(),
      }
    });
  }
});

module.exports = router;