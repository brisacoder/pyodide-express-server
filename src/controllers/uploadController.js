/**
 * File Upload Controller for CSV and Data File Management
 *
 * Handles file uploads with security logging, validation, and integration
 * with Pyodide's virtual filesystem for data analysis workflows.
 */
const logger = require('../utils/logger');
const path = require('node:path');
const { URL } = require('url');
const config = require('../config');


/**
 * Build a POSIX URL path (always starts with /uploads).
 * Encode the filename for safety.
 *
 * @param {string} filename - The name of the file to encode and append to /uploads.
 * @returns {string} POSIX URL path for the uploaded file.
 */
function toUploadUrlPath(filename) {
  return path.posix.join('/uploads', encodeURIComponent(filename));
}

/**
 * Handles CSV file uploads and mounts them into Pyodide's virtual filesystem.
 * Supports data analysis workflows by making uploaded files accessible to Python code.
 *
 * @param {Object} req - Express request object with multipart/form-data
 * @param {Object} req.file - Multer file object from upload middleware
 * @param {string} req.file.originalname - Original filename from client
 * @param {string} req.file.path - Temporary file path on server
 * @param {number} req.file.size - File size in bytes
 * @param {string} req.file.mimetype - MIME type of uploaded file
 * @param {string} req.ip - Client IP address
 * @param {Function} req.get - Function to get request headers
 * @param {Object} res - Express response object
 * @returns {Promise<void>} Sends a JSON response indicating upload success or error.
 */
async function uploadFile(req, res) {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        data: null,
        error: 'No file uploaded',
        meta: {
          timestamp: new Date().toISOString()
        }
      });
    }

    // Public URL path and absolute URL
    const urlPath = toUploadUrlPath(req.file.filename);
    const absoluteUrl = new URL(urlPath, `${req.protocol}://${req.get('host')}`).toString();
    const vfsPath  = path.posix.join(config.pyodideBases.uploads.urlBase, req.file.filename);

    // Security logging for file uploads
    logger.security('file_upload', {
      ip: req.ip,
      originalName: req.file.originalname,
      sanitizedOriginal: req.file.safeOriginalName,
      storedFilename: req.file.filename,
      size: req.file.size,
      mimetype: req.file.mimetype,
      vfsPath : vfsPath, // Pyodide virtual filesystem path. External Python programs will us it
      urlPath,                // "/uploads/<file>"
      absoluteUrl,           // "http(s)://host/uploads/<file>"
      userAgent: req.get('User-Agent'),
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      timestamp: new Date().toISOString(),
    });
    logger.info('=== FILE UPLOAD DEBUG START ===');
    logger.info('File info:', {
      originalName: req.file.originalname,
      sanitizedOriginal: req.file.safeOriginalName,
      storedFilename: req.file.filename,
      size: req.file.size,
      mimetype: req.file.mimetype,
      vfsPath : vfsPath , // absolute server path
      urlPath,                // "/uploads/<file>"
      absoluteUrl,           // "http(s)://host/uploads/<file>"
      userAgent: req.get('User-Agent'),
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      timestamp: new Date().toISOString(),
    });

    res.json({
      success: true,
      data: {
        file: {
          originalName: req.file.originalname,
          sanitizedOriginal: req.file.safeOriginalName,
          storedFilename: req.file.filename,
          size: req.file.size,
          mimetype: req.file.mimetype,
          vfsPath : vfsPath , // absolute server path
          urlPath,                // "/uploads/<file>"
          absoluteUrl,           // "http(s)://host/uploads/<file>"
          userAgent: req.get('User-Agent'),
          fileSize: req.file.size,
          mimeType: req.file.mimetype,
          timestamp: new Date().toISOString(),
        }
      },
      error: null,
      meta: {
        timestamp: new Date().toISOString()
      }
    });
    logger.info('=== FILE UPLOAD DEBUG END ===');
  } catch (error) {
    logger.error('=== FILE UPLOAD DEBUG END - ERROR ===');
    logger.error('Upload error:', error);
    logger.error('Error stack:', error.stack);
    res.status(500).json({
      success: false,
      data: null,
      error: error.message,
      meta: {
        timestamp: new Date().toISOString(),
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      }
    });
  }
}
module.exports = { uploadFile };
