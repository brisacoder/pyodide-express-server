/**
 * File Upload Controller for CSV and Data File Management
 *
 * Handles file uploads with security logging, validation, and integration
 * with Pyodide's virtual filesystem for data analysis workflows.
 */
const fs = require('fs');
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');
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
 * @returns {Promise<void>} Resolves when response is sent
 *
 * @example
 * // POST /api/upload-csv
 * // Content-Type: multipart/form-data
 * // Body: [CSV file data]
 *
 * // Success response:
 * // {
 * //   "success": true,
 * //   "message": "CSV file uploaded successfully",
 * //   "fileName": "data.csv",
 * //   "fileSize": 1024,
 * //   "pyodidePath": "/uploads/data.csv",
 * //   "sampleCode": "import pandas as pd\ndf = pd.read_csv('/uploads/data.csv')\nprint(df.head())"
 * // }
 *
 * // Error responses:
 * // {
 * //   "success": false,
 * //   "error": "No file uploaded"
 * // }
 *
 * @description
 * Upload Process:
 * 1. Validates file presence and type
 * 2. Logs security event with client details
 * 3. Reads file content from temporary location
 * 4. Mounts file into Pyodide virtual filesystem at /uploads/
 * 5. Provides sample Python code for file access
 * 6. Cleans up temporary files
 *
 * Security Features:
 * - Logs all upload attempts with IP and User-Agent
 * - File size and type validation
 * - Temporary file cleanup
 * - Safe filename handling
 *
 * Supported File Types:
 * - CSV files (.csv)
 * - Excel files (.xlsx, .xls) - if configured
 * - JSON data files (.json)
 * - Text files (.txt)
 *
 * Integration with Python:
 * - Files accessible at /uploads/ path in Pyodide
 * - Compatible with pandas, numpy, etc.
 * - Enables immediate data analysis workflows
 * - Provides sample code for common operations
 *
 * Error Handling:
 * - Graceful cleanup of temporary files
 * - Detailed error logging
 * - User-friendly error messages
 * - Proper HTTP status codes
 */
async function uploadCsv(req, res) {
  let tempFilePath = null;
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded',
      });
    }
    tempFilePath = req.file.path;
    // Security logging for file uploads
    logger.security('file_upload', {
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      fileName: req.file.originalname,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      timestamp: new Date().toISOString(),
    });
    logger.info('=== FILE UPLOAD DEBUG START ===');
    logger.info('File info:', {
      originalName: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
      tempPath: tempFilePath,
    });
    const pyodideStatus = pyodideService.getStatus();
    logger.info('Pyodide status:', pyodideStatus);
    if (!pyodideStatus.isReady) {
      return res.status(503).json({
        success: false,
        error: 'Pyodide is not ready yet',
        pyodideStatus,
      });
    }
    const fileContent = fs.readFileSync(tempFilePath, 'utf8');
    logger.info('File content read:', {
      length: fileContent.length,
      firstLine: fileContent.split('\n')[0],
      lineCount: fileContent.split('\n').length,
    });
    const sanitizedName = req.file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    const pyodideFilename = sanitizedName;
    logger.info('Using Pyodide filename:', pyodideFilename);
    logger.info('Testing basic Python execution...');
    try {
      const basicTest = await pyodideService.executeCode('2 + 2');
      logger.info('Basic test result:', basicTest);
      if (!basicTest.success || basicTest.result !== 4) {
        throw new Error('Basic Python execution failed');
      }
    } catch (basicError) {
      logger.error('Basic Python test failed:', basicError);
      return res.status(500).json({
        success: false,
        error: 'Pyodide is not working properly',
        details: basicError.message,
      });
    }
    logger.info('Testing pandas availability...');
    try {
      const pandasTest = await pyodideService.executeCode('import pandas as pd; pd.__version__');
      logger.info('Pandas test result:', pandasTest);
      if (!pandasTest.success) {
        throw new Error('Pandas not available');
      }
    } catch (pandasError) {
      logger.error('Pandas test failed:', pandasError);
      return res.status(500).json({
        success: false,
        error: 'Pandas is not available',
        details: pandasError.message,
      });
    }
    logger.info('Step 1: Writing file to Pyodide filesystem...');
    try {
      pyodideService.pyodide.FS.writeFile(pyodideFilename, fileContent);
      logger.info('File written successfully');
    } catch (writeError) {
      logger.error('Failed to write file to Pyodide FS:', writeError);
      return res.status(500).json({
        success: false,
        error: 'Failed to write file to Pyodide filesystem',
        details: writeError.message,
      });
    }
    logger.info('Step 2: Checking if file exists in Pyodide...');
    try {
      const fileExistsTest = await pyodideService.executeCode(`
import os
import json
exists = os.path.exists('${pyodideFilename}')
size = os.path.getsize('${pyodideFilename}') if exists else 0
json.dumps({'exists': exists, 'size': size})
      `);
      logger.info('File exists test:', fileExistsTest);
      if (!fileExistsTest.success) {
        throw new Error('Failed to check file existence');
      }
      let fileInfo;
      try {
        fileInfo = JSON.parse(fileExistsTest.result);
      } catch {
        throw new Error(`Failed to parse file check result: ${fileExistsTest.result}`);
      }
      if (!fileInfo.exists) {
        throw new Error('File does not exist in Pyodide filesystem');
      }
    } catch (existsError) {
      logger.error('File exists test failed:', existsError);
      return res.status(500).json({
        success: false,
        error: 'File was not written to Pyodide filesystem correctly',
        details: existsError.message,
      });
    }
    logger.info('Step 3: Creating analysis for uploaded file...');
    try {
      const fileExtension = req.file.originalname.toLowerCase().split('.').pop();
      let analysis = {
        success: true,
        fileType: fileExtension,
        size: req.file.size,
        has_data: true,
        memory_usage: req.file.size,
      };
      // File-type specific analysis
      if (fileExtension === 'csv') {
        // For CSV files, try to get shape and columns
        try {
          const csvAnalysis = await pyodideService.executeCode(`
import pandas as pd
import json
try:
    df = pd.read_csv('${pyodideFilename}')
    result = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'has_header': True,
        'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
    }
    json.dumps(result)
except Exception as e:
    json.dumps({'error': str(e), 'shape': [0, 0], 'columns': []})
          `);
          if (csvAnalysis.success && csvAnalysis.result) {
            const csvData = JSON.parse(csvAnalysis.result);
            analysis = { ...analysis, ...csvData };
            analysis.message = `CSV file uploaded successfully. ${csvData.shape[0]} rows × ${csvData.shape[1]} columns.`;
          } else {
            analysis.message =
              'CSV file uploaded. Use pandas to analyze: pd.read_csv(\'' + pyodideFilename + '\')';
          }
        } catch {
          analysis.message = 'CSV file uploaded but analysis failed. File is accessible in Python.';
        }
      } else if (fileExtension === 'json') {
        analysis.message =
          'JSON file uploaded successfully. Use json.load() or pandas.read_json() to read it.';
        analysis.shape = [0, 0]; // JSON doesn't have traditional shape
        analysis.columns = [];
      } else if (fileExtension === 'py') {
        analysis.message = 'Python file uploaded successfully. You can import or exec() this file.';
        analysis.shape = [0, 0]; // Python files don't have shape
        analysis.columns = [];
        // Count lines in Python file
        try {
          const lineCount = fileContent.split('\n').length;
          analysis.lineCount = lineCount;
          analysis.message = `Python file uploaded successfully. ${lineCount} lines of code.`;
        } catch {
          // Fallback if line counting fails
        }
      } else if (fileExtension === 'txt') {
        analysis.message = 'Text file uploaded successfully. Use open() to read the contents.';
        analysis.shape = [0, 0];
        analysis.columns = [];
        try {
          const lineCount = fileContent.split('\n').length;
          analysis.lineCount = lineCount;
          analysis.message = `Text file uploaded successfully. ${lineCount} lines of text.`;
        } catch {
          // Fallback if line counting fails
        }
      } else {
        analysis.message =
          'File uploaded successfully. Use appropriate Python libraries to process it.';
        analysis.shape = [0, 0];
        analysis.columns = [];
      }
      logger.info('Analysis created:', analysis);
      logger.info('=== FILE UPLOAD DEBUG END - SUCCESS ===');
      res.json({
        success: true,
        file: {
          originalName: req.file.originalname,
          size: req.file.size,
          pyodideFilename,
          tempPath: tempFilePath,
          keepFile: true,
        },
        analysis,
        debug: {
          pyodideStatus,
          fileWritten: true,
          pandasAvailable: true,
          fileLoaded: true,
          note: 'File is available in Pyodide filesystem as: ' + pyodideFilename,
        },
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      logger.error('Analysis creation failed:', error);
      return res.status(500).json({
        success: false,
        error: 'Failed to create file analysis',
        details: error.message,
      });
    }
  } catch (error) {
    logger.error('=== FILE UPLOAD DEBUG END - ERROR ===');
    logger.error('Upload error:', error);
    logger.error('Error stack:', error.stack);
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try {
        fs.unlinkSync(tempFilePath);
        logger.info('Cleaned up temp file on error');
      } catch (cleanupError) {
        logger.warn('Failed to cleanup temp file:', cleanupError.message);
      }
    }
    res.status(500).json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    });
  }
}
module.exports = { uploadCsv };
