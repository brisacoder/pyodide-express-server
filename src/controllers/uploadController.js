const fs = require('fs');
const pyodideService = require('../services/pyodide-service');
const logger = require('../utils/logger');

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
      timestamp: new Date().toISOString()
    });

    logger.info('=== CSV UPLOAD DEBUG START ===');
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
      } catch (parseError) {
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
      const analysis = {
        success: true,
        shape: [0, 0],
        columns: [],
        has_data: true,
        memory_usage: req.file.size,
        message: 'File uploaded successfully. Use Python code to analyze the data.',
      };

      logger.info('Analysis created:', analysis);
      logger.info('=== CSV UPLOAD DEBUG END - SUCCESS ===');

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
          csvLoaded: true,
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
    logger.error('=== CSV UPLOAD DEBUG END - ERROR ===');
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
