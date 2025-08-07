/**
 * File Management Routes
 * 
 * Handles all endpoints related to file operations (uploads, listing, deletion)
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
const config = require('../config');

const router = express.Router();

// Get upload directory from environment or default
const UPLOAD_DIR = config.uploadDir;

/**
 * @swagger
 * /api/uploaded-files:
 *   get:
 *     summary: List uploaded files
 *     description: Get a list of all files currently in the upload directory on the server filesystem
 *     tags: [File Operations]
 *     responses:
 *       200:
 *         description: File list retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/FileListResponse'
 *             examples:
 *               with_files:
 *                 summary: Files found
 *                 value:
 *                   success: true
 *                   files: [
 *                     {
 *                       filename: "csvFile-1691234567890-123456789.csv",
 *                       size: 3820915,
 *                       uploaded: "2024-01-15T10:30:00.000Z",
 *                       modified: "2024-01-15T10:30:00.000Z"
 *                     },
 *                     {
 *                       filename: "data.csv",
 *                       size: 1024,
 *                       uploaded: "2024-01-15T10:25:00.000Z",
 *                       modified: "2024-01-15T10:25:00.000Z"
 *                     }
 *                   ]
 *                   count: 2
 *                   uploadDir: "uploads"
 *               no_files:
 *                 summary: No files found
 *                 value:
 *                   success: true
 *                   files: []
 *                   count: 0
 *                   uploadDir: "uploads"
 *                   message: "Upload directory does not exist"
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.get('/uploaded-files', (req, res) => {
  try {
    // If the upload directory hasn't been created yet, return an empty list.
    if (!fs.existsSync(UPLOAD_DIR)) {
      return res.json({
        success: true,
        files: [],
        count: 0,
        uploadDir: UPLOAD_DIR,
        message: 'Upload directory does not exist'
      });
    }

    // Gather metadata for each file stored on disk.
    const files = fs.readdirSync(UPLOAD_DIR)
      .filter(file => !file.startsWith('.'))  // Exclude hidden files
      .map(filename => {
        const filePath = path.join(UPLOAD_DIR, filename);
        const stats = fs.statSync(filePath);
        
        return {
          filename: filename,
          size: stats.size,
          uploaded: stats.birthtime,
          modified: stats.mtime
        };
      })
      .sort((a, b) => new Date(b.uploaded) - new Date(a.uploaded)); // Sort by upload time, newest first

    logger.info(`Listed ${files.length} uploaded files`);

    res.json({
      success: true,
      files: files,
      count: files.length,
      uploadDir: UPLOAD_DIR
    });

  } catch (error) {
    logger.error('Error listing uploaded files:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /api/uploaded-files/{filename}:
 *   delete:
 *     summary: Delete uploaded file
 *     description: Delete a specific file from the server's upload directory
 *     tags: [File Operations]
 *     parameters:
 *       - in: path
 *         name: filename
 *         required: true
 *         schema:
 *           type: string
 *         description: Name of the file to delete
 *         example: "csvFile-1691234567890-123456789.csv"
 *     responses:
 *       200:
 *         description: File deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/FileDeleteResponse'
 *             examples:
 *               success:
 *                 summary: File deleted successfully
 *                 value:
 *                   success: true
 *                   message: "File data.csv deleted successfully"
 *       404:
 *         description: File not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *             examples:
 *               not_found:
 *                 summary: File not found
 *                 value:
 *                   success: false
 *                   error: "File not found"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       400:
 *         description: Invalid file path
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *             examples:
 *               invalid_path:
 *                 summary: Invalid file path
 *                 value:
 *                   success: false
 *                   error: "Invalid file path"
 *                   timestamp: "2024-01-15T10:30:00.000Z"
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.delete('/uploaded-files/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    const filePath = path.join(UPLOAD_DIR, filename);

    // Resolve paths to guard against directory traversal attacks and
    // ensure we operate only within the configured upload directory.
    // Security check - ensure file is in upload directory
    const resolvedUploadDir = path.resolve(UPLOAD_DIR);
    const resolvedFilePath = path.resolve(filePath);

    if (!resolvedFilePath.startsWith(resolvedUploadDir)) {
      logger.warn(`Attempted to delete file outside upload directory: ${filename}`);
      return res.status(400).json({
        success: false,
        error: 'Invalid file path',
        timestamp: new Date().toISOString()
      });
    }

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({
        success: false,
        error: 'File not found',
        timestamp: new Date().toISOString()
      });
    }

    // Get file info before deletion for logging
    const stats = fs.statSync(filePath);
    
    fs.unlinkSync(filePath);
    logger.info(`Deleted uploaded file: ${filename} (${stats.size} bytes)`);

    res.json({
      success: true,
      message: `File ${filename} deleted successfully`,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Error deleting uploaded file:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /api/pyodide-files:
 *   get:
 *     summary: List files in Pyodide virtual filesystem
 *     description: Get a list of all files currently loaded in Pyodide's virtual filesystem (in-memory)
 *     tags: [File Operations]
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
    
    // Parse the Python result if it's a string
    let parsedResult = result;
    if (result.success && result.result && typeof result.result === 'string') {
      try {
        // Convert Python dictionary string to JSON
        let jsonString = result.result
          .replace(/'/g, '"')  // Single quotes to double quotes
          .replace(/True/g, 'true')  // Python True to JSON true
          .replace(/False/g, 'false');  // Python False to JSON false
        
        parsedResult = { 
          ...result, 
          result: JSON.parse(jsonString) 
        };
      } catch (parseError) {
        logger.warn(`Could not parse Pyodide files result:`, parseError.message);
      }
    }
    
    logger.info('Listed Pyodide files:', {
      success: parsedResult.success,
      fileCount: parsedResult.result?.count || 0
    });
    
    res.json(parsedResult);
  } catch (error) {
    logger.error('Pyodide file list endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /api/pyodide-files/{filename}:
 *   delete:
 *     summary: Delete file from Pyodide filesystem
 *     description: Delete a specific file from Pyodide's virtual filesystem (in-memory)
 *     tags: [File Operations]
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
    
    logger.info(`Attempting to delete Pyodide file: ${filename}`);
    
    const result = await pyodideService.deletePyodideFile(filename);
    
    logger.info('Pyodide file deletion result:', {
      filename: filename,
      success: result.success,
      result: result.result
    });
    
    res.json(result);
  } catch (error) {
    logger.error('Pyodide file deletion endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /api/file-info/{filename}:
 *   get:
 *     summary: Get detailed file information
 *     description: Get detailed information about a specific file in both upload directory and Pyodide filesystem
 *     tags: [File Operations]
 *     parameters:
 *       - in: path
 *         name: filename
 *         required: true
 *         schema:
 *           type: string
 *         description: Name of the file to get information about
 *         example: "data.csv"
 *     responses:
 *       200:
 *         description: File information retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 filename:
 *                   type: string
 *                 uploadedFile:
 *                   type: object
 *                   properties:
 *                     exists:
 *                       type: boolean
 *                     size:
 *                       type: number
 *                     uploaded:
 *                       type: string
 *                       format: date-time
 *                     modified:
 *                       type: string
 *                       format: date-time
 *                     path:
 *                       type: string
 *                 pyodideFile:
 *                   type: object
 *                   properties:
 *                     exists:
 *                       type: boolean
 *                     analysis:
 *                       type: object
 *             examples:
 *               found:
 *                 summary: File found in both locations
 *                 value:
 *                   success: true
 *                   filename: "data.csv"
 *                   uploadedFile:
 *                     exists: true
 *                     size: 1024
 *                     uploaded: "2024-01-15T10:30:00.000Z"
 *                     modified: "2024-01-15T10:30:00.000Z"
 *                     path: "uploads/data.csv"
 *                   pyodideFile:
 *                     exists: true
 *                     analysis:
 *                       shape: [100, 5]
 *                       columns: ["col1", "col2", "col3", "col4", "col5"]
 *       404:
 *         description: File not found in either location
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
router.get('/file-info/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    const result = {
      success: true,
      filename: filename,
      uploadedFile: { exists: false },
      pyodideFile: { exists: false }
    };

    // Check uploaded file on the Node.js filesystem
    const uploadPath = path.join(UPLOAD_DIR, filename);
    if (fs.existsSync(uploadPath)) {
      const stats = fs.statSync(uploadPath);
      result.uploadedFile = {
        exists: true,
        size: stats.size,
        uploaded: stats.birthtime,
        modified: stats.mtime,
        path: uploadPath
      };
    }

    // Check whether the file also exists inside Pyodide's in-memory FS
    try {
      const pyodideResult = await pyodideService.executeCode(`
import os
filename = '${filename}'
if os.path.exists(filename):
    stat_info = os.stat(filename)
    result = {
        'exists': True,
        'size': stat_info.st_size,
        'modified': stat_info.st_mtime
    }
else:
    result = {'exists': False}

result
      `);

      if (pyodideResult.success && pyodideResult.result) {
        try {
          // Convert Python dictionary string to JSON
          let jsonString = pyodideResult.result
            .replace(/'/g, '"')  // Single quotes to double quotes
            .replace(/True/g, 'true')  // Python True to JSON true
            .replace(/False/g, 'false');  // Python False to JSON false
          
          const pyodideFileInfo = JSON.parse(jsonString);
          result.pyodideFile = pyodideFileInfo;
        } catch (parseError) {
          logger.warn(`Could not parse Pyodide file result for ${filename}:`, parseError.message);
          result.pyodideFile = { exists: false };
        }
      }
    } catch (pyodideError) {
      logger.warn(`Could not check Pyodide file ${filename}:`, pyodideError.message);
    }

    // Return 404 if file doesn't exist anywhere
    if (!result.uploadedFile.exists && !result.pyodideFile.exists) {
      return res.status(404).json({
        success: false,
        error: `File ${filename} not found in uploaded files or Pyodide filesystem`,
        timestamp: new Date().toISOString()
      });
    }

    res.json(result);

  } catch (error) {
    logger.error('Error getting file info:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;