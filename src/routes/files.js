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
const { meta } = require('@eslint/js');
const router = express.Router();
// Get upload directory from environment or default
const UPLOAD_DIR = path.resolve(path.join(config.pyodideDataDir, config.pyodideBases.uploads.urlBase));
const PYODIDE_UPLOAD_DIR = config.pyodideBases.uploads.urlBase;
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
        data: {
          files: [],
          count: 0,
          uploadDir: UPLOAD_DIR,
          message: 'Upload directory does not exist',
        }
      });
    }
    // Gather metadata for each file stored on disk.
    const files = fs
      .readdirSync(UPLOAD_DIR)
      .filter((file) => !file.startsWith('.')) // Exclude hidden files
      .map((filename) => {
        const filePath = path.join(UPLOAD_DIR, filename);
        const stats = fs.statSync(filePath);
        return {
          filename: filename,
          size: stats.size,
          uploaded: stats.birthtime,
          modified: stats.mtime,
        };
      })
      .sort((a, b) => new Date(b.uploaded) - new Date(a.uploaded)); // Sort by upload time, newest first
    logger.info(`Listed ${files.length} uploaded files`);
    res.json({
      success: true,
      data: {
        files: files,
        count: files.length,
        uploadDir: PYODIDE_UPLOAD_DIR, // This is what external users see
      },
      error: null,
      meta: {
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error) {
    logger.error('Error listing uploaded files:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString(),
      },
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
router.delete('/uploaded-files/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    
    // Security check - prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      logger.warn(`Attempted path traversal in file deletion: ${filename}`);
      return res.status(400).json({
        success: false,
        error: 'Invalid filename',
        meta: {
          timestamp: new Date().toISOString(),
        },
      });
    }
    
    logger.info(`Attempting to delete file from Pyodide: ${filename}`);
    
    // Delete the file from within Pyodide's virtual filesystem
    const result = await pyodideService.deletePyodideFile(filename);
    
    if (result.success) {
      logger.info(`Deleted file from Pyodide: ${filename}`);
      res.json({
        success: true,
        data: {
          message: result.message || `File ${filename} deleted successfully`,
        },
        meta: {
          timestamp: new Date().toISOString(),
        },
      });
    } else {
      // Handle specific error cases
      if (result.error && result.error.includes('not found')) {
        return res.status(404).json({
          success: false,
          error: 'File not found',
          meta: {
            timestamp: new Date().toISOString(),
          },
        });
      }
      
      // Other errors
      return res.status(500).json({
        success: false,
        error: result.error || 'Failed to delete file',
        meta: {
          timestamp: new Date().toISOString(),
        },
      });
    }
  } catch (error) {
    logger.error('Error deleting uploaded file:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString(),
      },
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
        data: {
          exists: false,
          filename: filename,
          pyodideFile: null,        
          size: null,
          uploaded: null,
          modified: null,
          vfsPath: null,
        },
      meta: {
        timestamp: new Date().toISOString(),
      }     
    };
    // Check uploaded file on the Node.js filesystem
    const vfsPath  = path.posix.join(config.pyodideBases.uploads.urlBase, filename);
    const uploadPath = path.join(UPLOAD_DIR, filename);
    if (fs.existsSync(uploadPath)) {
      const stats = fs.statSync(uploadPath);
      result.data = {
        exists: true,
        filename: filename,
        size: stats.size,
        uploaded: stats.birthtime,
        modified: stats.mtime,
        vfsPath: vfsPath,
      };
    } 
    // Check whether the file also exists inside Pyodide's in-memory FS
    try {
      const pyodideResult = await pyodideService.executeCode(`
import os
filename = '${vfsPath}'
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
          // The result is already a JavaScript object from the execute endpoint
          if (typeof pyodideResult.result === 'object') {
            result.data.pyodideFile = pyodideResult.result;
          } else {
            // Fallback: if it's a string, try to parse it
            let jsonString = pyodideResult.result
              .toString()
              .replace(/'/g, '"') // Single quotes to double quotes
              .replace(/True/g, 'true') // Python True to JSON true
              .replace(/False/g, 'false'); // Python False to JSON false
            const pyodideFileInfo = JSON.parse(jsonString);
            result.data.pyodideFile = pyodideFileInfo;
          }
        } catch (parseError) {
          logger.warn(`Could not parse Pyodide file result for ${filename}:`, parseError.message);
          result.data.pyodideFile = { exists: false };
        }
      }
    } catch (pyodideError) {
      logger.warn(`Could not check Pyodide file ${filename}:`, pyodideError.message);
    }
    res.json(result);
  } catch (error) {
    logger.error('Error getting file info:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString(),
      }
    });
  }
});
module.exports = router;
