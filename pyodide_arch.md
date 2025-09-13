# Pyodide Integration Documentation

## Overview

This documentation provides a detailed breakdown of how Pyodide (Python in WebAssembly) is initialized and how Python code execution works in the Express server. This covers the complete flow from server startup to Python code execution.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   server.js     │    │ pyodide-service  │    │   execute.js        │
│                 │    │                  │    │                     │
│ ┌─startServer()─┼────┼─initialize()─────┼────┼─POST /api/execute   │
│ │             │ │    │                  │    │                     │
│ └─app.listen()─┘ │    │ ┌─executeCode()──┼────┼─validateCode        │
│                 │    │ │                │    │                     │
│   Routes setup  │    │ └─Python exec    │    │   Response          │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 1. Server Initialization and Pyodide Setup

### 1.1 Server Startup Flow

**File: `src/server.js`**

The initialization begins when the server starts:

```javascript
// Lines 328-365: startServer() function
async function startServer() {
  try {
    logger.info('Starting Pyodide Express Server...');
    
    // Lines 331-334: Pyodide initialization
    logger.info('Initializing Pyodide...');
    await pyodideService.initialize();  // ← KEY CALL
    logger.info('Pyodide initialization completed!');
    
    // Lines 336-350: Express server startup
    const server = app.listen(PORT, () => {
      // Server ready logging
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}
```

**Call Path**: `server.js:startServer()` → `pyodide-service.js:initialize()`

### 1.2 Pyodide Service Initialization

**File: `src/services/pyodide-service.js`**

#### Class Structure
```javascript
// Lines 7-14: PyodideService class definition
class PyodideService {
  constructor() {
    this.pyodide = null;           // Pyodide instance
    this.isReady = false;          // Ready state flag
    this.initializationPromise = null;  // Singleton pattern
    this.executionTimeout = 30000; // 30 seconds default
  }
}
```

#### Initialize Method
```javascript
// Lines 20-27: initialize() - Entry point
async initialize() {
  if (this.initializationPromise) {
    return this.initializationPromise;  // Singleton pattern
  }

  this.initializationPromise = this._performInitialization();
  return this.initializationPromise;
}
```

**Call Path**: `initialize()` → `_performInitialization()`

### 1.3 Core Initialization Logic

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 33-165: _performInitialization() - The heart of setup
async _performInitialization() {
  try {
    logger.info('Starting Pyodide initialization...');

    // Lines 35-50: Import Pyodide module
    let loadPyodide;
    try {
      logger.info('Importing Pyodide module...');
      const pyodideModule = await import('pyodide');
      loadPyodide = pyodideModule.loadPyodide;
      logger.info('✅ Pyodide module imported successfully');
    } catch (importError) {
      throw new Error(`Could not import Pyodide: ${importError.message}`);
    }

    // Lines 55-60: Load Pyodide runtime
    logger.info('Loading Pyodide runtime (using default configuration)...');
    this.pyodide = await loadPyodide();  // ← CORE PYODIDE LOADING
    
    // Lines 64-82: Load essential packages
    const packagesToLoad = ['numpy', 'pandas', 'micropip', 'matplotlib', 'requests'];
    const loadedPackages = [];
    
    for (const packageName of packagesToLoad) {
      try {
        await this.pyodide.loadPackage([packageName]);  // ← PACKAGE LOADING
        loadedPackages.push(packageName);
      } catch (packageError) {
        logger.warn(`⚠️  Failed to load ${packageName}`);
      }
    }

    // Lines 85-100: Install additional packages via micropip
    const micropipPackages = ['seaborn', 'httpx'];
    for (const packageName of micropipPackages) {
      try {
        await this.pyodide.runPythonAsync(`
import micropip
await micropip.install("${packageName}")
        `);
      } catch (packageError) {
        logger.warn(`⚠️  Failed to install ${packageName}`);
      }
    }

    // Lines 105-160: Python environment setup
    await this.pyodide.runPythonAsync(`
import sys
import io
from io import StringIO

# Import available packages and make them globally accessible
try:
    import numpy as np
    globals()['np'] = np
except ImportError:
    pass

try:
    import pandas as pd
    globals()['pd'] = pd
except ImportError:
    pass

# ... more package imports ...

# Create output capture system
class OutputCapture:
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def start_capture(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
    
    def stop_capture(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return {
            'stdout': self.stdout.getvalue(),
            'stderr': self.stderr.getvalue()
        }

# Global output capturer
output_capture = OutputCapture()

# Helper function for JSON serialization
def make_json_safe(obj):
    # Convert Python objects to JSON-serializable format
    # ... conversion logic ...
    `);

    this.isReady = true;  // ← READY STATE SET
    logger.info('🎉 Pyodide initialization completed successfully!');
    return true;
  }
}
```

## 2. Python Code Execution Flow

### 2.1 HTTP Request Handling

**File: `src/routes/execute.js`**

```javascript
// Lines 66-96: POST /api/execute route handler
router.post('/execute', validateCode, async (req, res) => {
  try {
    const { code, context, timeout } = req.body;
    
    // Lines 68-73: Request logging
    logger.info('Executing Python code:', { 
      codeLength: code.length,
      hasContext: !!context,
      timeout: timeout || 'default',
      ip: req.ip
    });

    // Line 75: Core execution call
    const result = await pyodideService.executeCode(code, context, timeout);
    
    // Lines 77-83: Result handling
    if (result.success) {
      logger.info('Code execution successful');
    } else {
      logger.warn('Code execution failed:', result.error);
    }
    
    res.json(result);  // ← RESPONSE SENT
  } catch (error) {
    // Error handling
  }
});
```

**Call Path**: `execute.js:router.post('/execute')` → `pyodide-service.js:executeCode()`

### 2.2 Code Execution Implementation

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 176-250: executeCode() - The execution engine
async executeCode(code, context = {}, timeout = this.executionTimeout) {
  // Lines 177-182: Validation
  if (!this.isReady) {
    throw new Error('Pyodide is not ready. Please wait for initialization to complete.');
  }
  if (!code || typeof code !== 'string') {
    throw new Error('Code must be a non-empty string');
  }

  try {
    // Lines 185-188: Set up context variables
    for (const [key, value] of Object.entries(context)) {
      this.pyodide.globals.set(key, value);  // ← CONTEXT INJECTION
    }

    // Line 191: Start output capture
    await this.pyodide.runPythonAsync('output_capture.start_capture()');

    let result;
    let executionError = null;

    try {
      // Lines 196-202: Execute the code
      if (code.includes('await ') || code.includes('micropip.install')) {
        result = await this.pyodide.runPythonAsync(code);  // ← ASYNC EXECUTION
      } else {
        result = this.pyodide.runPython(code);  // ← SYNC EXECUTION
      }
    } catch (error) {
      executionError = error;
    }

    // Lines 205-208: Stop capture and get output
    const output = await this.pyodide.runPythonAsync('output_capture.stop_capture()');

    // Line 211: Clean up
    await this.pyodide.runPythonAsync('output_capture.reset()');

    // Lines 213-221: Error handling
    if (executionError) {
      return {
        success: false,
        error: executionError.message,
        stdout: output.get('stdout') || '',
        stderr: output.get('stderr') || '',
        timestamp: new Date().toISOString()
      };
    }

    // Lines 224-230: Result conversion
    let jsonSafeResult = null;
    if (result !== undefined && result !== null) {
      try {
        jsonSafeResult = await this.pyodide.runPythonAsync(`make_json_safe(${JSON.stringify(result)})`);
      } catch (conversionError) {
        jsonSafeResult = String(result);
      }
    }

    // Lines 232-239: Success response
    return {
      success: true,
      result: jsonSafeResult,
      stdout: output.get('stdout') || '',
      stderr: output.get('stderr') || '',
      timestamp: new Date().toISOString()
    };
  }
}
```

## 3. Key Components and Data Flow

### 3.1 Output Capture System

**Location**: `pyodide-service.js`, lines 105-160 (Python code within runPythonAsync)

```python
class OutputCapture:
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def start_capture(self):
        # Redirect stdout/stderr to StringIO objects
        self.stdout = StringIO()
        self.stderr = StringIO()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
    
    def stop_capture(self):
        # Restore original stdout/stderr and return captured content
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return {
            'stdout': self.stdout.getvalue(),
            'stderr': self.stderr.getvalue()
        }
```

**Usage Pattern**:
1. `start_capture()` - Redirect Python stdout/stderr (line 191)
2. Execute user code (lines 196-202)
3. `stop_capture()` - Get captured output (line 205)
4. `reset()` - Clean up (line 211)

### 3.2 Result Serialization

**Location**: `pyodide-service.js`, lines 224-230

The system converts Python objects to JSON-safe format using the `make_json_safe()` Python function:

```javascript
// JavaScript side
jsonSafeResult = await this.pyodide.runPythonAsync(`make_json_safe(${JSON.stringify(result)})`);
```

```python
# Python side (defined in initialization)
def make_json_safe(obj):
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif hasattr(obj, 'to_dict'):  # pandas objects
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return str(obj)
```

### 3.3 Context Variable Injection

**Location**: `pyodide-service.js`, lines 185-188

```javascript
// Set up context variables
for (const [key, value] of Object.entries(context)) {
  this.pyodide.globals.set(key, value);  // Makes variables available in Python
}
```

This allows JavaScript variables to be accessible in Python code:

```javascript
// JavaScript request
{
  "code": "print(f'Hello {name}! Your value is {value}')",
  "context": {
    "name": "Alice", 
    "value": 42
  }
}
```

## 4. File Upload and CSV Processing

### 4.1 CSV Upload Flow

**File: `src/server.js`**

```javascript
// Lines 124-180: CSV upload endpoint
app.post('/api/upload', upload.single('csvFile'), async (req, res) => {
  try {
    // Lines 128-132: File validation
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      });
    }

    // Lines 140-144: Read and process file
    const fileContent = fs.readFileSync(req.file.path, 'utf8');
    const filename = 'uploaded_file.csv';
    const loadResult = await pyodideService.loadCSVFile(filename, fileContent);

    // Line 147: Clean up uploaded file
    fs.unlinkSync(req.file.path);
  }
});
```

**Call Path**: `server.js:upload endpoint` → `pyodide-service.js:loadCSVFile()`

### 4.2 CSV File Loading

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 320-370: loadCSVFile() implementation
async loadCSVFile(filename, csvContent) {
  if (!this.isReady) {
    throw new Error('Pyodide is not ready');
  }

  try {
    // Line 326: Write file to Pyodide's virtual filesystem
    this.pyodide.FS.writeFile(filename, csvContent);

    // Lines 329-350: Verify the file was loaded correctly
    const verificationResult = await this.executeCode(`
# Check if pandas is available and load the file
try:
    import pandas as pd
    df = pd.read_csv('${filename}')
    {
        'success': True,
        'filename': '${filename}',
        'shape': df.shape,
        'columns': list(df.columns),
        'sample': df.head(3).to_dict('records') if len(df) > 0 else []
    }
except ImportError:
    {
        'success': False,
        'error': 'Pandas not available - cannot process CSV',
        'filename': '${filename}'
    }
except Exception as e:
    {
        'success': False,
        'error': str(e),
        'filename': '${filename}'
    }
    `);

    return verificationResult;
  }
}
```

## 5. Package Management

### 5.1 Package Installation

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 260-285: installPackage() implementation
async installPackage(packageName) {
  if (!this.isReady) {
    throw new Error('Pyodide is not ready');
  }

  try {
    logger.info(`Installing package: ${packageName}`);
    
    const result = await this.executeCode(`
# Check if micropip is available
try:
    import micropip
    await micropip.install("${packageName}")
    f"Successfully installed {packageName}"
except ImportError:
    "Micropip not available - cannot install packages"
except Exception as e:
    f"Installation failed: {str(e)}"
    `);

    return result;
  }
}
```

## 6. Error Handling and Recovery

### 6.1 Initialization Error Handling

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 157-165: Error handling in initialization
} catch (error) {
  logger.error('❌ Failed to initialize Pyodide:', error.message);
  logger.error('Error details:', error);
  this.isReady = false;
  throw new Error(`Pyodide initialization failed: ${error.message}`);
}
```

### 6.2 Execution Error Handling

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 241-250: Cleanup on execution error
} catch (error) {
  // Ensure cleanup
  try {
    await this.pyodide.runPythonAsync('output_capture.reset()');
  } catch (resetError) {
    // Ignore cleanup errors
  }

  throw new Error(`Execution error: ${error.message}`);
}
```

## 7. State Management

### 7.1 Service Status

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 375-384: getStatus() method
getStatus() {
  return {
    isReady: this.isReady,           // Boolean: ready for execution
    initialized: this.pyodide !== null,  // Boolean: Pyodide loaded
    executionTimeout: this.executionTimeout,  // Number: timeout in ms
    version: this.pyodide?.version || 'unknown',  // String: Pyodide version
    timestamp: new Date().toISOString()
  };
}
```

### 7.2 Environment Reset

**File: `src/services/pyodide-service.js`**

```javascript
// Lines 389-415: reset() method
async reset() {
  if (!this.isReady) {
    throw new Error('Pyodide is not ready');
  }

  try {
    await this.pyodide.runPythonAsync(`
# Clear user-defined variables but keep system ones
user_vars = [var for var in globals().keys() 
             if not var.startswith('_') 
             and var not in ['sys', 'io', 'StringIO', 'output_capture', 'make_json_safe']]
             
for var in user_vars:
    try:
        del globals()[var]
    except:
        pass

# Reset output capture
try:
    output_capture.reset()
except:
    pass
    `);
  }
}
```

## 8. Dependencies and Configuration

### 8.1 Package Dependencies

**File: `package.json`**

```json
{
  "dependencies": {
    "express": "^4.21.1",
    "multer": "^2.0.0", 
    "pyodide": "^0.28.0",  // ← Core Pyodide dependency
    "cors": "^2.8.5"
  }
}
```

### 8.2 Multer Configuration

**File: `src/server.js`**

```javascript
// Lines 30-60: Multer setup for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: MAX_FILE_SIZE,  // 10MB default
    files: 1
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.csv', '.json', '.txt', '.py'];
    const ext = path.extname(file.originalname).toLowerCase();
    
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`File type ${ext} not allowed`));
    }
  }
});
```

## 9. Enhanced Security Logging System

### 9.1 Security Architecture Overview

The enhanced security logging system provides comprehensive monitoring and audit trails for all Python code execution without disrupting the core Pyodide functionality.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   execute.js    │    │   logger.js      │    │   stats.js          │
│                 │    │                  │    │                     │
│ ┌─executeCode()─┼────┼─security()───────┼────┼─GET /dashboard/stats │
│ │               │    │                  │    │                     │
│ │ SHA-256 hash  │    │ ┌─updateStats()─ │    │   Chart.js render   │
│ │ IP tracking   │    │ │               │    │                     │
│ │ Timing        │    │ └─Dual logging  │    │   Interactive UI    │
│ └─Response      │    │                  │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### 9.2 Security Event Logging

**File: `src/utils/logger.js`**

#### Enhanced Logger Structure
```javascript
// Lines 45-95: security() method - Core security logging
security(event, details = {}) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    level: 'SECURITY',
    event,
    details,
    ...this.getRequestContext()
  };

  // Write to security.log
  this.writeToSecurityLog(logEntry);
  
  // Update real-time statistics
  this.updateExecutionStats(event, details);
}
```

#### Crypto Hash Generation
```javascript
// Lines 156-168: generateCodeHash() - SHA-256 hashing
generateCodeHash(code) {
  try {
    return crypto.createHash('sha256')
      .update(code, 'utf8')
      .digest('hex');
  } catch (error) {
    console.error('Error generating code hash:', error);
    return 'hash-error-' + Date.now();
  }
}
```

### 9.3 Code Execution Integration

**File: `src/routes/execute.js`**

The security logging is seamlessly integrated into the execution flow:

```javascript
// Lines 45-65: Enhanced execution with security logging
router.post('/execute', asyncHandler(async (req, res) => {
  try {
    const { code, timeout = 30000 } = req.body;
    
    // Generate security hash BEFORE execution
    const codeHash = logger.generateCodeHash(code);
    const startTime = Date.now();
    
    // Execute code with Pyodide
    const result = await pyodideService.executeCode(code, timeout);
    const executionTime = Date.now() - startTime;
    
    // Log successful execution with security details
    logger.security('code_execution', {
      success: true,
      codeHash,
      executionTime,
      resultLength: result?.length || 0,
      hasOutput: !!(result && result.trim())
    });
    
    // Return standard response (unchanged for compatibility)
    res.json({ success: true, result });
    
  } catch (error) {
    // Log security event for failed executions
    logger.security('code_execution_error', {
      success: false,
      error: error.message,
      codeHash: logger.generateCodeHash(code || ''),
      executionTime: Date.now() - startTime
    });
    
    res.status(500).json({ success: false, error: error.message });
  }
}));
```

### 9.4 Real-time Statistics Collection

**File: `src/utils/logger.js`**

#### Statistics Structure
```javascript
// Lines 170-185: Statistics data structure
this.executionStats = {
  totalExecutions: 0,
  successfulExecutions: 0,
  failedExecutions: 0,
  totalExecutionTime: 0,
  averageExecutionTime: 0,
  lastExecutionTime: null,
  startTime: Date.now(),
  errorCategories: {},
  topIPs: {},
  userAgents: {},
  hourlyTrend: new Array(24).fill(0)  // 24-hour trend
};
```

#### Real-time Updates
```javascript
// Lines 201-245: updateExecutionStats() - Live statistics
updateExecutionStats(event, details) {
  if (event === 'code_execution') {
    this.executionStats.totalExecutions++;
    
    if (details.success) {
      this.executionStats.successfulExecutions++;
      
      // Track execution time
      if (details.executionTime) {
        this.executionStats.totalExecutionTime += details.executionTime;
        this.executionStats.averageExecutionTime = 
          this.executionStats.totalExecutionTime / this.executionStats.successfulExecutions;
      }
    } else {
      this.executionStats.failedExecutions++;
      
      // Categorize errors
      const errorType = this.categorizeError(details.error);
      this.executionStats.errorCategories[errorType] = 
        (this.executionStats.errorCategories[errorType] || 0) + 1;
    }
    
    // Update hourly trend
    const currentHour = new Date().getHours();
    this.executionStats.hourlyTrend[currentHour]++;
    
    // Track IP addresses
    const clientIP = this.getClientIP();
    if (clientIP) {
      this.executionStats.topIPs[clientIP] = 
        (this.executionStats.topIPs[clientIP] || 0) + 1;
    }
    
    // Track User-Agent strings
    const userAgent = this.getUserAgent();
    if (userAgent) {
      this.executionStats.userAgents[userAgent] = 
        (this.executionStats.userAgents[userAgent] || 0) + 1;
    }
    
    this.executionStats.lastExecutionTime = new Date().toISOString();
  }
}
```

### 9.5 Interactive Dashboard System

**File: `src/routes/stats.js`**

#### Dashboard Endpoints
```javascript
// Lines 15-45: Dashboard statistics API
router.get('/dashboard/stats', (req, res) => {
  try {
    const stats = logger.getStats();
    const uptime = process.uptime();
    
    const response = {
      success: true,
      stats: {
        overview: {
          totalExecutions: stats.totalExecutions,
          successRate: ((stats.successfulExecutions / stats.totalExecutions) * 100).toFixed(1),
          averageExecutionTime: Math.round(stats.averageExecutionTime || 0),
          uptimeSeconds: Math.round(uptime),
          uptimeHuman: formatUptime(uptime)
        },
        recent: {
          lastHourExecutions: stats.hourlyTrend[new Date().getHours()],
          recentSuccessRate: calculateRecentSuccessRate(stats),
          packagesInstalled: stats.packagesInstalled || 0,
          filesUploaded: stats.filesUploaded || 0
        },
        topIPs: formatTopEntries(stats.topIPs),
        topErrors: formatTopEntries(stats.errorCategories),
        userAgents: formatTopEntries(stats.userAgents),
        hourlyTrend: stats.hourlyTrend
      },
      timestamp: new Date().toISOString()
    };
    
    res.json(response);
  } catch (error) {
    logger.error('Error getting dashboard stats:', error);
    res.status(500).json({ success: false, error: 'Failed to get statistics' });
  }
});
```

#### Chart.js Dashboard Generation
```javascript
// Lines 150-300: Interactive HTML dashboard
router.get('/dashboard/stats/dashboard', (req, res) => {
  try {
    const stats = logger.getStats();
    
    // Generate professional HTML with Chart.js
    const dashboardHTML = generateDashboardHTML(stats);
    
    res.setHeader('Content-Type', 'text/html');
    res.send(dashboardHTML);
  } catch (error) {
    logger.error('Error generating dashboard:', error);
    res.status(500).send('<h1>Error generating dashboard</h1>');
  }
});

function generateDashboardHTML(stats) {
  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pyodide Express Server - Statistics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            /* Professional CSS with gradients and responsive design */
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            /* ... extensive CSS for professional appearance ... */
        </style>
    </head>
    <body>
        <!-- Interactive charts and statistics display -->
        <script>
            // Chart.js configuration for real-time visualization
            const hourlyTrendChart = new Chart(ctx, {
                type: 'line',
                data: { /* real-time data */ },
                options: { /* responsive configuration */ }
            });
        </script>
    </body>
    </html>
  `;
}
```

### 9.6 Backward Compatibility

The security enhancements maintain 100% backward compatibility:

#### Legacy API Compatibility
```javascript
// Lines 25-45: /api/stats endpoint (unchanged format)
router.get('/api/stats', asyncHandler(async (req, res) => {
  try {
    // Get enhanced statistics
    const enhancedStats = logger.getStats();
    
    // Format in legacy structure for existing clients
    const legacyResponse = {
      uptime: process.uptime(),
      pyodideReady: pyodideService.isReady(),
      // ... other legacy fields preserved ...
      
      // NEW: Enhanced data nested under executionStats
      executionStats: {
        totalExecutions: enhancedStats.totalExecutions,
        successRate: enhancedStats.successfulExecutions / enhancedStats.totalExecutions,
        averageExecutionTime: enhancedStats.averageExecutionTime,
        errorCategories: enhancedStats.errorCategories,
        hourlyTrend: enhancedStats.hourlyTrend
      }
    };
    
    res.json(legacyResponse);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}));
```

### 9.7 Test Coverage

**File: `tests/test_security_logging.py`**

The security logging system includes comprehensive test coverage:

```python
class SecurityLoggingTestCase(unittest.TestCase):
    """Test the enhanced security logging system."""
    
    def test_01_security_logging_code_execution(self):
        """Test that code execution events are properly logged with security data."""
        # Execute Python code and verify security logging
        
    def test_02_security_logging_error_tracking(self):
        """Test that errors are properly categorized and tracked."""
        # Test error categorization and statistics
        
    def test_06_dashboard_endpoints_functionality(self):
        """Test all dashboard endpoints for proper functionality."""
        # Verify all dashboard endpoints work correctly
        
    def test_07_legacy_stats_endpoint_compatibility(self):
        """Test that legacy stats endpoint maintains backward compatibility."""
        # Ensure existing clients continue to work
```

## Summary

The Pyodide integration with enhanced security logging follows this comprehensive flow:

### Core Execution Flow
1. **Server Start** → `server.js:startServer()` 
2. **Initialize** → `pyodide-service.js:initialize()` → `_performInitialization()`
3. **Load Pyodide** → `await loadPyodide()` 
4. **Load Packages** → `loadPackage()` and `micropip.install()`
5. **Setup Environment** → Python code setup with output capture
6. **Ready State** → `this.isReady = true`

### Enhanced Security Flow (New!)
1. **Request Received** → `execute.js` routes with security context
2. **Code Hashing** → SHA-256 hash generation for audit trails
3. **Security Logging** → Real-time event logging with IP/User-Agent tracking
4. **Execute Python** → Standard Pyodide execution (unchanged)
5. **Statistics Update** → Real-time metrics collection and trend analysis
6. **Response** → Standard API response (backward compatible)
7. **Dashboard Update** → Live statistics available via Chart.js interface

### Security Architecture Benefits
- **Non-Disruptive**: 100% backward compatibility with existing APIs
- **Comprehensive Monitoring**: SHA-256 hashing, IP tracking, error categorization
- **Real-time Analytics**: Live dashboard with Chart.js visualizations
- **Audit Trail**: Dual logging streams (server.log + security.log)
- **Performance Metrics**: Execution timing, success rates, hourly trends
- **Professional UI**: Responsive dashboard with professional design

### System Capabilities
The enhanced system provides a robust Python execution environment in Node.js through WebAssembly, with:
- **Comprehensive error handling** and output capture
- **Package management** capabilities via micropip
- **Security monitoring** with audit trails and real-time statistics
- **Interactive dashboard** for monitoring and analytics
- **Extensive test coverage** including security-specific test suite
- **Professional documentation** and development tools

The architecture demonstrates how security enhancements can be seamlessly integrated into existing systems without disrupting functionality while providing enterprise-grade monitoring and audit capabilities.