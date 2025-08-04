/**
 * Main Express server for Pyodide Python execution
 * 
 * This server provides REST API endpoints for executing Python code,
 * uploading files, and managing Python packages using Pyodide.
 */

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Import services and utilities
const pyodideService = require('./services/pyodide-service');
const logger = require('./utils/logger');
const executeRoutes = require('./routes/execute');
const fileRoutes = require('./routes/files');  // ← NEW IMPORT

// Import middleware
const { validateCode, validatePackage } = require('./middleware/validation');

// Import Swagger configuration
const { swaggerSpec, swaggerUi, swaggerUiOptions } = require('./swagger-config');

// Configuration
const PORT = process.env.PORT || 3000;
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024; // 10MB
const UPLOAD_DIR = process.env.UPLOAD_DIR || 'uploads';

// Create Express app
const app = express();

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Configure multer for file uploads (v2.x syntax)
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function (req, file, cb) {
    // Generate unique filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: MAX_FILE_SIZE,
    files: 1
  },
  fileFilter: (req, file, cb) => {
    // Accept CSV, JSON, TXT, and Python files
    const allowedTypes = ['.csv', '.json', '.txt', '.py'];
    const ext = path.extname(file.originalname).toLowerCase();

    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`File type ${ext} not allowed. Allowed types: ${allowedTypes.join(', ')}`));
    }
  }
});

// Middleware
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Request logging middleware
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.path} - ${req.ip}`);
  next();
});

// Serve static files from public directory
app.use(express.static(path.join(__dirname, '../public')));

// Swagger Documentation
/**
 * @swagger
 * /:
 *   get:
 *     summary: Web interface for testing the API
 *     description: Serves an interactive HTML interface for testing Python code execution
 *     tags: [System]
 *     responses:
 *       200:
 *         description: HTML interface loaded successfully
 *         content:
 *           text/html:
 *             schema:
 *               type: string
 */

// API Documentation endpoint - Swagger UI
app.use('/docs', swaggerUi.serve);
app.get('/docs', swaggerUi.setup(swaggerSpec, swaggerUiOptions));

// API Documentation JSON endpoint
app.get('/docs.json', (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});

// Use route modules
app.use('/api', executeRoutes);  // Python execution routes
app.use('/api', fileRoutes);     // ← NEW: File management routes

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Main health check endpoint
 *     description: Comprehensive health check including server status, Pyodide status, and system information
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Health check completed successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/HealthResponse'
 *       500:
 *         description: Health check failed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
app.get('/health', (req, res) => {
  const status = pyodideService.getStatus();
  const logInfo = logger.getLogInfo();

  res.json({
    status: 'ok',
    server: 'running',
    pyodide: status,
    logging: logInfo,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage()
  });
});

// Add this endpoint to your server.js (before error handling middleware)

/**
 * @swagger
 * /api/execute-raw:
 *   post:
 *     summary: Execute Python code from raw text
 *     description: Execute Python code sent as plain text (bypasses JSON escaping issues)
 *     tags: [Python Execution]
 *     requestBody:
 *       required: true
 *       content:
 *         text/plain:
 *           schema:
 *             type: string
 *             example: |
 *               import pandas as pd
 *               import numpy as np
 *               
 *               def analyze_data():
 *                   """
 *                   Triple quoted docstring works fine!
 *                   """
 *                   name = "Alice"
 *                   result = f"Hello {name}!"  # f-strings work!
 *                   return result
 *               
 *               analyze_data()
 *         application/x-python:
 *           schema:
 *             type: string
 *     responses:
 *       200:
 *         description: Code execution completed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ExecuteResponse'
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
app.post('/api/execute-raw', express.text({ limit: '10mb' }), async (req, res) => {
  try {
    const code = req.body;

    if (!code || typeof code !== 'string' || !code.trim()) {
      return res.status(400).json({
        success: false,
        error: 'No Python code provided in request body',
        timestamp: new Date().toISOString()
      });
    }

    logger.info('Executing raw Python code:', {
      codeLength: code.length,
      ip: req.ip,
      contentType: req.get('Content-Type')
    });

    // Execute the code using the same service
    const result = await pyodideService.executeCode(code);

    if (result.success) {
      logger.info('Raw code execution successful');
    } else {
      logger.warn('Raw code execution failed:', result.error);
    }

    res.json(result);

  } catch (error) {
    logger.error('Raw execution endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Also add context support for raw text
app.post('/api/execute-raw-with-context', express.json({ limit: '10mb' }), async (req, res) => {
  try {
    const { code, context, timeout } = req.body;

    if (!code || typeof code !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'No Python code provided',
        timestamp: new Date().toISOString()
      });
    }

    logger.info('Executing raw Python code with context:', {
      codeLength: code.length,
      hasContext: !!context,
      timeout: timeout || 'default',
      ip: req.ip
    });

    const result = await pyodideService.executeCode(code, context, timeout);

    res.json(result);

  } catch (error) {
    logger.error('Raw execution with context error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /api/upload-csv:
 *   post:
 *     summary: Upload and process CSV file
 *     description: |
 *       Upload a CSV file, load it into Pyodide's virtual filesystem, and return detailed analysis.
 *       
 *       **File Handling:**
 *       - Files are temporarily stored in the uploads folder with unique names
 *       - Files are loaded into Pyodide's virtual filesystem with their original names (sanitized)
 *       - By default, files are kept in the uploads folder (can be configured to auto-delete)
 *       
 *       **Analysis Includes:**
 *       - File dimensions (rows × columns)
 *       - Column names and data types
 *       - Memory usage
 *       - Null value counts
 *       - Sample data (first 3 rows)
 *       - Statistical summary for numeric columns
 *     tags: [File Operations]
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               csvFile:
 *                 type: string
 *                 format: binary
 *                 description: CSV file to upload (max 10MB)
 *     responses:
 *       200:
 *         description: File uploaded and processed successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/UploadResponse'
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       500:
 *         $ref: '#/components/responses/InternalError'
 */
// Replace your upload endpoint in server.js with this debugging version

app.post('/api/upload-csv', upload.single('csvFile'), async (req, res) => {
  let tempFilePath = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      });
    }

    tempFilePath = req.file.path;
    
    logger.info('=== CSV UPLOAD DEBUG START ===');
    logger.info('File info:', {
      originalName: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
      tempPath: tempFilePath
    });

    // Check Pyodide status
    const pyodideStatus = pyodideService.getStatus();
    logger.info('Pyodide status:', pyodideStatus);
    
    if (!pyodideStatus.isReady) {
      return res.status(503).json({
        success: false,
        error: 'Pyodide is not ready yet',
        pyodideStatus: pyodideStatus
      });
    }

    // Read the uploaded file
    const fileContent = fs.readFileSync(tempFilePath, 'utf8');
    logger.info('File content read:', {
      length: fileContent.length,
      firstLine: fileContent.split('\n')[0],
      lineCount: fileContent.split('\n').length
    });

    // Use original filename (sanitized) in Pyodide
    const sanitizedName = req.file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    const pyodideFilename = sanitizedName;
    logger.info('Using Pyodide filename:', pyodideFilename);

    // Test if we can execute basic Python first
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
        details: basicError.message
      });
    }

    // Test pandas availability
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
        details: pandasError.message
      });
    }

    // Now try to load the CSV file step by step
    logger.info('Step 1: Writing file to Pyodide filesystem...');
    try {
      pyodideService.pyodide.FS.writeFile(pyodideFilename, fileContent);
      logger.info('File written successfully');
    } catch (writeError) {
      logger.error('Failed to write file to Pyodide FS:', writeError);
      return res.status(500).json({
        success: false,
        error: 'Failed to write file to Pyodide filesystem',
        details: writeError.message
      });
    }

    // Step 2: Test if file exists in Pyodide
    logger.info('Step 2: Checking if file exists in Pyodide...');
    try {
      const fileExistsTest = await pyodideService.executeCode(`
import os
exists = os.path.exists('${pyodideFilename}')
size = os.path.getsize('${pyodideFilename}') if exists else 0
{'exists': exists, 'size': size}
      `);
      logger.info('File exists test:', fileExistsTest);
      
      if (!fileExistsTest.success || !fileExistsTest.result.exists) {
        throw new Error('File does not exist in Pyodide filesystem');
      }
    } catch (existsError) {
      logger.error('File exists test failed:', existsError);
      return res.status(500).json({
        success: false,
        error: 'File was not written to Pyodide filesystem correctly',
        details: existsError.message
      });
    }

    // Step 3: Try to load with pandas
    logger.info('Step 3: Loading CSV with pandas...');
    try {
      const csvLoadTest = await pyodideService.executeCode(`
import pandas as pd
try:
    df = pd.read_csv('${pyodideFilename}')
    {
        'success': True,
        'shape': df.shape,
        'columns': list(df.columns)[:10],  # First 10 columns only
        'memory_usage': float(df.memory_usage(deep=True).sum()),
        'has_data': len(df) > 0
    }
except Exception as e:
    {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
      `);
      
      logger.info('CSV load test result:', csvLoadTest);
      
      if (!csvLoadTest.success) {
        throw new Error(`CSV load failed: ${csvLoadTest.result?.error || 'Unknown error'}`);
      }
      
      // If we get here, the CSV loaded successfully
      const analysis = csvLoadTest.result;
      
      logger.info('=== CSV UPLOAD DEBUG END - SUCCESS ===');
      
      res.json({
        success: true,
        file: {
          originalName: req.file.originalname,
          size: req.file.size,
          pyodideFilename: pyodideFilename,
          tempPath: tempFilePath,
          keepFile: true
        },
        analysis: analysis,
        debug: {
          pyodideStatus: pyodideStatus,
          fileWritten: true,
          pandasAvailable: true,
          csvLoaded: true
        },
        timestamp: new Date().toISOString()
      });
      
    } catch (csvError) {
      logger.error('CSV loading failed:', csvError);
      return res.status(500).json({
        success: false,
        error: 'Failed to load CSV with pandas',
        details: csvError.message
      });
    }

  } catch (error) {
    logger.error('=== CSV UPLOAD DEBUG END - ERROR ===');
    logger.error('Upload error:', error);
    logger.error('Error stack:', error.stack);

    // Clean up on error
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
      timestamp: new Date().toISOString()
    });
  }
});

// Serve simple web interface for testing
app.get('/', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html>
<head>
    <title>Pyodide Express Server</title>
    <style>
        body { 
            font-family: sans-serif; 
            margin: 2rem; 
            max-width: 1200px; 
            line-height: 1.6;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }
        .header h1 { margin: 0; }
        .docs-link {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 10px 20px;
            margin: 10px 5px;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s;
        }
        .docs-link:hover {
            background: rgba(255,255,255,0.3);
        }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .section { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; }
        textarea { 
            width: 100%; 
            font-family: 'Courier New', monospace; 
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        button { 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .result { 
            background: #ffffff; 
            border: 1px solid #dee2e6;
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px; 
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .error { background: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .success { background: #d4edda; color: #155724; border-color: #c3e6cb; }
        .status { 
            padding: 10px; 
            margin: 10px 0; 
            border-radius: 4px; 
            font-weight: bold;
        }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.ready { background: #d4edda; color: #155724; }
        h3 { color: #495057; margin-top: 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐍 Pyodide Express Server</h1>
        <p>Execute Python code in your browser with full data science capabilities</p>
        <a href="/docs" class="docs-link">📚 API Documentation</a>
        <a href="/docs.json" class="docs-link">📄 OpenAPI Spec</a>
        <a href="/health" class="docs-link">🔍 Health Check</a>
        <a href="/api/uploaded-files" class="docs-link">📁 Uploaded Files</a>
    </div>
    
    <div id="status" class="status loading">Checking Pyodide status...</div>
    
    <div class="container">
        <div class="section">
            <h3>Python Code Execution</h3>
                <textarea id="code" rows="12" placeholder="Enter Python code here...">
# Example: Advanced data analysis with complex Python syntax
import pandas as pd
import numpy as np

def analyze_battery_data():
    """
    Analyze battery performance with complex Python features.
    This now works with f-strings, docstrings, and any Python syntax!
    """
    # Create sample data with f-strings
    user_name = "Data Scientist"
    print(f"Analysis started by: {user_name}")
    
    data = {
        'battery_id': [f'BAT_{i:03d}' for i in range(1, 6)],
        'soh': [95.2, 87.8, 92.1, 89.5, 94.3],
        'temperature': [25.5, 32.1, 28.7, 30.2, 26.8],
        'cycles': [1250, 2100, 1680, 1950, 1320]
    }
    
    df = pd.DataFrame(data)
    print("Battery Data:")
    print(df)
    
    # Complex analysis with multiple quotes and f-strings
    avg_soh = df['soh'].mean()
    best_battery = df.loc[df['soh'].idxmax(), 'battery_id']
    
    summary = f"""
    📊 Battery Analysis Summary:
    - Average SOH: {avg_soh:.1f}%
    - Best performing battery: {best_battery}
    - Total batteries analyzed: {len(df)}
    """
    
    print(summary)
    
    # Return structured results
    return {
        'average_soh': round(avg_soh, 2),
        'best_battery': best_battery,
        'total_count': len(df),
        'data_preview': df.head(3).to_dict('records')
    }

# Execute the analysis
result = analyze_battery_data()
result
            </textarea><br>
            
            <button onclick="executeCode()" id="runBtn">Run Python Code</button>
            <button onclick="clearResult()">Clear Output</button>
            <button onclick="resetEnvironment()">Reset Environment</button>
            
            <h3>File Upload</h3>
            <input type="file" id="csvFile" accept=".csv,.json,.txt,.py">
            <button onclick="uploadFile()">Upload File</button>
            <button onclick="listFiles()">List Files</button>
            
            <h3>Package Management</h3>
            <input type="text" id="packageName" placeholder="Package name (e.g., requests)" style="width: 200px;">
            <button onclick="installPackage()">Install Package</button>
            <button onclick="listPackages()">List Packages</button>
        </div>
        
        <div class="section">
            <h3>Output</h3>
            <div id="result" class="result">Ready to execute Python code...</div>
        </div>
    </div>

<script>
        // Check Pyodide status on load
        window.addEventListener('load', checkStatus);
        
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                const statusDiv = document.getElementById('status');
                
                if (status.isReady) {
                    statusDiv.className = 'status ready';
                    statusDiv.textContent = '✅ Pyodide is ready! Complex Python syntax fully supported.';
                } else {
                    statusDiv.className = 'status loading';
                    statusDiv.textContent = '⏳ Pyodide is initializing... Please wait.';
                    // Check again in 2 seconds
                    setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                const statusDiv = document.getElementById('status');
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ Failed to connect to server: ' + error.message;
            }
        }
        
        // UPDATED: Use raw endpoint to handle complex Python syntax
        async function executeCode() {
            const code = document.getElementById('code').value;
            const resultDiv = document.getElementById('result');
            const runBtn = document.getElementById('runBtn');
            
            if (!code.trim()) {
                resultDiv.innerHTML = '<div class="error">Please enter some Python code</div>';
                return;
            }
            
            runBtn.disabled = true;
            runBtn.textContent = 'Running...';
            resultDiv.innerHTML = '<div>Executing Python code...</div>';
            
            try {
                // Use raw endpoint - handles f-strings, docstrings, any Python syntax!
                const response = await fetch('/api/execute-raw', {
                    method: 'POST',
                    headers: { 'Content-Type': 'text/plain' },
                    body: code  // Send as plain text, not JSON
                });
                
                const result = await response.json();
                
                if (result.success) {
                    let output = '';
                    if (result.stdout) {
                        output += 'Output:\\n' + result.stdout + '\\n';
                    }
                    if (result.result !== undefined && result.result !== null) {
                        output += '\\nReturn Value:\\n' + JSON.stringify(result.result, null, 2);
                    }
                    resultDiv.innerHTML = \`<div class="success">\${output || 'Code executed successfully (no output)'}</div>\`;
                } else {
                    resultDiv.innerHTML = \`<div class="error">Error:\\n\${result.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Request Error:\\n\${error.message}</div>\`;
            } finally {
                runBtn.disabled = false;
                runBtn.textContent = 'Run Python Code';
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            const resultDiv = document.getElementById('result');
            
            if (!file) {
                resultDiv.innerHTML = '<div class="error">Please select a file</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('csvFile', file);
            
            resultDiv.innerHTML = '<div>Uploading file...</div>';
            
            try {
                const response = await fetch('/api/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const analysis = result.analysis;
                    let output = \`File uploaded successfully!\\n\\n\`;
                    output += \`Original name: \${result.file.originalName}\\n\`;
                    output += \`Size: \${(result.file.size / 1024).toFixed(1)} KB\\n\`;
                    output += \`Available as: \${result.file.pyodideFilename}\\n\`;
                    output += \`Temp path: \${result.file.tempPath}\\n\\n\`;
                    
                    if (analysis && analysis.success) {
                        output += \`Shape: \${analysis.shape[0]} rows × \${analysis.shape[1]} columns\\n\`;
                        output += \`Columns: \${analysis.columns.join(', ')}\\n\`;
                        if (analysis.numeric_columns) {
                            output += \`Numeric columns: \${analysis.numeric_columns.length}\\n\`;
                        }
                    }
                    
                    resultDiv.innerHTML = \`<div class="success">\${output}</div>\`;
                } else {
                    resultDiv.innerHTML = \`<div class="error">Upload Error:\\n\${result.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Upload Error:\\n\${error.message}</div>\`;
            }
        }
        
        async function listFiles() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div>Listing files...</div>';
            
            try {
                const [uploadedResponse, pyodideResponse] = await Promise.all([
                    fetch('/api/uploaded-files'),
                    fetch('/api/pyodide-files')
                ]);
                
                const uploadedResult = await uploadedResponse.json();
                const pyodideResult = await pyodideResponse.json();
                
                let output = 'FILE LISTING\\n===========\\n\\n';
                
                // Uploaded files
                output += \`UPLOADED FILES (\${uploadedResult.count || 0}):\\n\`;
                if (uploadedResult.success && uploadedResult.files.length > 0) {
                    uploadedResult.files.forEach(file => {
                        output += \`  📁 \${file.filename} (\${(file.size / 1024).toFixed(1)} KB)\\n\`;
                    });
                } else {
                    output += '  (none)\\n';
                }
                
                output += '\\n';
                
                // Pyodide files
                if (pyodideResult.success && pyodideResult.result) {
                    const pyodideFiles = pyodideResult.result.files || [];
                    output += \`PYODIDE FILES (\${pyodideFiles.length}):\\n\`;
                    if (pyodideFiles.length > 0) {
                        pyodideFiles.forEach(file => {
                            output += \`  🐍 \${file.name} (\${(file.size / 1024).toFixed(1)} KB)\\n\`;
                        });
                    } else {
                        output += '  (none)\\n';
                    }
                } else {
                    output += 'PYODIDE FILES: (error getting list)\\n';
                }
                
                resultDiv.innerHTML = \`<div class="success">\${output}</div>\`;
                
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">File Listing Error:\\n\${error.message}</div>\`;
            }
        }
        
        async function installPackage() {
            const packageName = document.getElementById('packageName').value.trim();
            const resultDiv = document.getElementById('result');
            
            if (!packageName) {
                resultDiv.innerHTML = '<div class="error">Please enter a package name</div>';
                return;
            }
            
            resultDiv.innerHTML = \`<div>Installing package: \${packageName}...</div>\`;
            
            try {
                const response = await fetch('/api/install-package', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ package: packageName })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = \`<div class="success">Package installed successfully!\\n\${result.stdout}</div>\`;
                    document.getElementById('packageName').value = '';
                } else {
                    resultDiv.innerHTML = \`<div class="error">Installation failed:\\n\${result.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Installation Error:\\n\${error.message}</div>\`;
            }
        }
        
        async function listPackages() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div>Getting package list...</div>';
            
            try {
                const response = await fetch('/api/packages');
                const result = await response.json();
                
                if (result.success && result.result) {
                    const packages = result.result.installed_packages || [];
                    let output = \`Installed Packages (\${packages.length}):\\n\\n\`;
                    output += packages.join('\\n');
                    resultDiv.innerHTML = \`<div class="success">\${output}</div>\`;
                } else {
                    resultDiv.innerHTML = \`<div class="error">Failed to get package list:\\n\${result.error || 'Unknown error'}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Request Error:\\n\${error.message}</div>\`;
            }
        }
        
        async function resetEnvironment() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div>Resetting environment...</div>';
            
            try {
                const response = await fetch('/api/reset', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = '<div class="success">Environment reset successfully!</div>';
                } else {
                    resultDiv.innerHTML = \`<div class="error">Reset failed:\\n\${result.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Reset Error:\\n\${error.message}</div>\`;
            }
        }
        
        function clearResult() {
            document.getElementById('result').innerHTML = 'Ready to execute Python code...';
        }
        
        // Allow Ctrl+Enter to run code
        document.getElementById('code').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                executeCode();
            }
        });
    </script>
</body>
</html>
  `);
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);

  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: `File size too large. Maximum size: ${MAX_FILE_SIZE / 1024 / 1024}MB`
      });
    }
  }

  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    timestamp: new Date().toISOString()
  });
});

/**
 * Initialize and start the server
 */
async function startServer() {
  try {
    logger.info('Starting Pyodide Express Server...');

    // Initialize Pyodide
    logger.info('Initializing Pyodide...');
    await pyodideService.initialize();
    logger.info('Pyodide initialization completed!');

    // Start the Express server
    const server = app.listen(PORT, () => {
      const logInfo = logger.getLogInfo();

      logger.info(`🚀 Server running on port ${PORT}`);
      logger.info(`📖 Web interface: http://localhost:${PORT}`);
      logger.info(`📚 API Documentation: http://localhost:${PORT}/docs`);
      logger.info(`🔧 API base URL: http://localhost:${PORT}/api`);
      logger.info(`📊 Health check: http://localhost:${PORT}/health`);
      logger.info(`📁 File management: http://localhost:${PORT}/api/uploaded-files`);

      if (logInfo.isFileLoggingEnabled) {
        logger.info(`📝 Logs writing to: ${logInfo.logFile}`);
        logger.info(`📁 Log directory: ${logInfo.logDirectory}`);
      } else {
        logger.info(`📺 Console-only logging (no file output)`);
      }
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      logger.info('SIGINT received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    return server;

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server if this file is run directly
if (require.main === module) {
  startServer();
}

module.exports = { app, startServer };