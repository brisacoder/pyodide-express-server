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
const executeRoutes = require('./routes/execute')

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

// Use the execute routes
app.use('/api', executeRoutes);

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

/**
 * @swagger
 * /api/upload-csv:
 *   post:
 *     summary: Upload and process CSV file
 *     description: Upload a CSV file, load it into Pyodide's virtual filesystem, and return basic analysis
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
 *         description: No file provided or invalid file type
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 *       500:
 *         description: File processing failed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
app.post('/api/upload-csv', upload.single('csvFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      });
    }

    logger.info('Processing uploaded file:', {
      originalName: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype
    });

    // Read the uploaded file
    const fileContent = fs.readFileSync(req.file.path, 'utf8');

    // Load the file into Pyodide
    const filename = 'uploaded_file.csv';
    const loadResult = await pyodideService.loadCSVFile(filename, fileContent);

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    logger.info('File processed successfully:', loadResult.result);

    res.json({
      success: true,
      file: {
        originalName: req.file.originalname,
        size: req.file.size,
        pyodideFilename: filename
      },
      analysis: loadResult.result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('CSV upload error:', error);

    // Clean up on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      success: false,
      error: error.message,
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
    </div>
    
    <div id="status" class="status loading">Checking Pyodide status...</div>
    
    <div class="container">
        <div class="section">
            <h3>Python Code Execution</h3>
            <textarea id="code" rows="12" placeholder="Enter Python code here...">
# Example: Data analysis with pandas
import pandas as pd
import numpy as np

# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'score': [85, 92, 78, 96]
}

df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
print(f"\\nAverage age: {df['age'].mean():.1f}")
print(f"Top score: {df['score'].max()}")

# Return summary statistics
df.describe().to_dict()
            </textarea><br>
            
            <button onclick="executeCode()" id="runBtn">Run Python Code</button>
            <button onclick="clearResult()">Clear Output</button>
            <button onclick="resetEnvironment()">Reset Environment</button>
            
            <h3>File Upload</h3>
            <input type="file" id="csvFile" accept=".csv,.json,.txt,.py">
            <button onclick="uploadFile()">Upload File</button>
            
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
                    statusDiv.textContent = '✅ Pyodide is ready! You can now execute Python code.';
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
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code })
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
                    output += \`Available as: \${result.file.pyodideFilename}\\n\\n\`;
                    
                    if (analysis.success) {
                        output += \`Shape: \${analysis.shape[0]} rows × \${analysis.shape[1]} columns\\n\`;
                        output += \`Columns: \${analysis.columns.join(', ')}\\n\`;
                    }
                    
                    resultDiv.innerHTML = \`<div class="success">\${output}</div>\`;
                } else {
                    resultDiv.innerHTML = \`<div class="error">Upload Error:\\n\${result.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">Upload Error:\\n\${error.message}</div>\`;
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