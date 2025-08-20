# Copilot Instructions for Pyodide Express Server

## Project Overview

This is a **Node.js Express server** that provides a REST API for executing Python code using **Pyodide** (Python in WebAssembly). The project enables data science workflows with direct filesystem mounting, package management, and comprehensive testing.

**Key Technologies:**
- **Backend:** Node.js + Express.js
- **Python Runtime:** Pyodide (Python 3.11 in WebAssembly)
- **Data Science:** matplotlib, seaborn, pandas, scikit-learn, numpy
- **File Handling:** Multer for uploads, direct filesystem mounting
- **Documentation:** Swagger/OpenAPI
- **Testing:** Python unittest + Node.js testing
- **Environment:** Windows PowerShell 7, npm/node package management

## Critical Terminal Behavior

⚠️ **CRITICAL: POWERSHELL TERMINAL CREATION BEHAVIOR**
- **EVERY TIME YOU RUN `run_in_terminal()` A NEW POWERSHELL TERMINAL WINDOW IS CREATED**
- **THE NEW TERMINAL TAKES ~15 SECONDS TO INITIALIZE AND BE READY FOR COMMANDS**
- **IF YOU ISSUE A COMMAND IMMEDIATELY, IT GOES INTO NOTHING AND IS LOST**
- **YOU MUST WAIT 15+ SECONDS AFTER CREATING A TERMINAL BEFORE ISSUING ANY COMMANDS**
- **THIS AFFECTS ALL OPERATIONS: npm, python, curl, node, etc.**

⚠️ **CRITICAL: COMMANDS THAT DO NOT CREATE NEW TERMINALS**
- **`echo` commands run in EXISTING terminals and do NOT create new ones**
- **If `echo` responds immediately, that terminal is ALREADY INITIALIZED and ready to use**
- **Use `echo` to test if a terminal is ready before running other commands**
- **DO NOT assume that running `run_in_terminal()` uses the same terminal as previous commands**

⚠️ **CRITICAL: TERMINAL REUSE PATTERN**
- **Every `run_in_terminal()` call creates a NEW terminal unless specified otherwise**
- **To reuse a terminal: First run `echo "test"` to check if it responds immediately**
- **If echo responds, that terminal is ready for immediate use**
- **If echo doesn't respond or creates new terminal, wait 15+ seconds before next command**

```powershell
# ❌ WRONG - Command is lost because terminal isn't ready
run_in_terminal("npm start")           # Creates new terminal, command is lost
get_terminal_output(terminal_id)       # Will show empty prompt

# ❌ WRONG - Creates too many terminals
run_in_terminal("echo test")           # Creates terminal 1
run_in_terminal("node server.js")     # Creates terminal 2 (command lost)
run_in_terminal("curl localhost:3000") # Creates terminal 3 (command lost)

# ✅ CORRECT - Wait for terminal to initialize first
run_in_terminal("echo 'Terminal ready'")  # Create terminal with simple command
# WAIT 15-20 SECONDS for terminal initialization
get_terminal_output(terminal_id)           # Check if terminal is ready
run_in_terminal("npm start")               # Now issue the actual command

# ✅ CORRECT - Test terminal readiness first
run_in_terminal("echo 'Testing terminal readiness'")  # If this responds immediately, terminal is ready
run_in_terminal("node src/server.js")                 # This will work in the ready terminal

# ✅ ALTERNATIVE - Use existing ready terminals when possible
# Check context for existing terminals that are already initialized
```

## Critical Developer Workflows

### Environment Setup (Node.js + Python Testing)

```powershell
# Required: Node.js 18+, Python 3.11+ for testing
# First time setup
npm ci                                 # Install Node.js dependencies
mkdir -p uploads logs plots           # Create required directories

# Daily workflow - start the server
npm start                             # Start server on localhost:3000
npm run dev                           # Start with nodemon for development

# Testing workflows
python -m unittest tests.test_api -v              # Run specific test module
python run_comprehensive_tests.py                 # Run all tests with server management
python run_simple_tests.py                       # Quick smoke tests
```

### Server Management

```powershell
# Server operations
npm start                             # Production start
npm run dev                           # Development with auto-reload
npm run dev:inspect                   # Development with Node.js inspector

# Health and monitoring
curl http://localhost:3000/health     # Quick health check
curl http://localhost:3000/api/status # Detailed server status

# Cleanup operations
npm run clean:logs                    # Clear log files
npm run clean:uploads                 # Clear uploaded files
npm run clean                         # Full dependency reinstall
```

### Pyodide-Specific Operations

```javascript
// Executing Python code via API
POST /api/execute
{
  "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.savefig('/plots/matplotlib/test.png')",
  "timeout": 30000
}

// Installing Python packages
POST /api/install-package
{
  "package": "scikit-learn"
}

// File uploads for data analysis
POST /api/upload-csv
Content-Type: multipart/form-data
[CSV file]
```

## Development Patterns

### Automatic Task Completion
- **Always iterate and continue working** until tasks are fully complete
- **Never ask** "Continue to iterate?" or similar questions
- **Keep working through problems**, testing, debugging, and refining until the objective is achieved
- **Run tests after changes** to ensure functionality

### Communication Standards
- **Markdown Communication**: All explanations must use proper markdown formatting
- **Code sections** properly tagged with language identifiers (`javascript`, `python`, `powershell`, `json`, `yaml`)
- **File paths** wrapped in backticks: `src/server.js`, `tests/test_api.py`
- **API endpoints** clearly formatted: `POST /api/execute`

### Windows PowerShell 7 Environment
- **Always use PowerShell 7** command syntax
- **File operations** use PowerShell commands:
  ```powershell
  Remove-Item file1.js, file2.py               # Delete files
  Get-ChildItem -Path .\logs\ -Recurse         # List directory contents
  Test-Path .\uploads\data.csv                 # Check file existence
  ```

### Cross-Platform Path Handling
- **JavaScript/Node.js**: Use `path` module for all file operations
  ```javascript
  const path = require('path');
  const filePath = path.join(__dirname, 'uploads', filename);
  ```
- **Python code** (executed in Pyodide): ALWAYS use `pathlib.Path`
  ```python
  from pathlib import Path
  plots_dir = Path('/plots/matplotlib')
  plot_file = plots_dir / 'chart.png'
  ```

### Testing and Quality Assurance
- **Test-Driven Development**: Every new functionality must have corresponding tests
- **Run comprehensive tests** after significant changes
- **Always cleanup** temporary files, test artifacts, and uploaded files
- **Validate API responses** with proper status codes and structure

## Architecture and Code Organization

### Project Structure
```
src/
├── app.js                    # Express app configuration
├── server.js                 # Server startup and lifecycle
├── config/                   # Configuration management
├── controllers/              # Route handlers
├── middleware/               # Express middleware
├── routes/                   # API route definitions
├── services/                 # Business logic (Pyodide service)
└── utils/                    # Utilities (logging, metrics)

tests/                        # Python test suite
docs/                         # Documentation
examples/                     # Usage examples
uploads/                      # File upload storage
logs/                         # Server logs
plots/                        # Generated plot files
```

### Node.js/Express Best Practices

```javascript
// ✅ CORRECT - Express route pattern
const express = require('express');
const router = express.Router();
const { validateRequest } = require('../middleware/validation');

router.post('/api/execute', validateRequest, async (req, res) => {
  try {
    const { code, timeout = 30000 } = req.body;
    const result = await pyodideService.executeCode(code, timeout);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// ✅ CORRECT - Async error handling
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// ✅ CORRECT - Path handling
const path = require('path');
const uploadPath = path.join(__dirname, '..', 'uploads', filename);
```

### Pyodide Integration Patterns

```javascript
// ✅ CORRECT - Pyodide service pattern
class PyodideService {
  async initialize() {
    const { loadPyodide } = require('pyodide');
    this.pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.0/full/'
    });
    await this.setupFilesystem();
  }

  async executeCode(code, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timeout')), timeout);
      try {
        const result = this.pyodide.runPython(code);
        clearTimeout(timer);
        resolve(result);
      } catch (error) {
        clearTimeout(timer);
        reject(error);
      }
    });
  }
}
```

### Python Code Execution Standards

When writing Python code that will be executed in Pyodide:

```python
# ✅ CORRECT - Modern Python patterns
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# File operations using pathlib
data_path = Path('/uploads/data.csv')
if data_path.exists():
    df = pd.read_csv(data_path)

# Plot generation with proper paths
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / 'analysis.png', dpi=150, bbox_inches='tight')

# ❌ WRONG - Old-style file operations
import os
if os.path.exists('/uploads/data.csv'):
    df = pd.read_csv('/uploads/data.csv')
os.makedirs('/plots/matplotlib', exist_ok=True)
plt.savefig('/plots/matplotlib/analysis.png')
```

## Testing Strategy

### Test Categories
1. **API Tests** (`tests/test_api.py`) - Core API functionality
2. **Security Tests** (`tests/test_security.py`) - Input validation, safety
3. **Performance Tests** (`tests/test_performance.py`) - Timeout, resource limits
4. **Integration Tests** (`tests/test_integration.py`) - End-to-end workflows
5. **Data Science Tests** (`tests/test_matplotlib.py`, `tests/test_sklearn.py`) - Library functionality
6. **Filesystem Tests** (`tests/test_virtual_filesystem.py`) - File operations

### Test Execution Patterns

```powershell
# Individual test modules
python -m unittest tests.test_api -v
python -m unittest tests.test_performance -v

# Comprehensive testing with server management
python run_comprehensive_tests.py

# Quick smoke tests
python run_simple_tests.py

# Test specific functionality
python -c "
import requests
r = requests.post('http://localhost:3000/api/execute', 
                  json={'code': 'import pandas as pd; print(pd.__version__)'})
print(r.json())
"
```

### Test Data Management

```python
# ✅ CORRECT - Test cleanup pattern
class TestCase(unittest.TestCase):
    def setUp(self):
        self.uploaded_files = []
        self.temp_files = []
    
    def tearDown(self):
        # Clean up uploaded files
        for filename in self.uploaded_files:
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
```

## API Design Standards

### Request/Response Patterns

```javascript
// ✅ CORRECT - Consistent API response structure
{
  "success": true,
  "result": { /* actual data */ },
  "metadata": {
    "executionTime": 1234,
    "timestamp": "2025-08-19T10:30:00Z"
  }
}

// ✅ CORRECT - Error response structure
{
  "success": false,
  "error": "Descriptive error message",
  "code": "EXECUTION_TIMEOUT",
  "details": { /* additional context */ }
}
```

### Endpoint Documentation

```javascript
/**
 * @swagger
 * /api/execute:
 *   post:
 *     summary: Execute Python code
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               code:
 *                 type: string
 *                 description: Python code to execute
 *               timeout:
 *                 type: integer
 *                 description: Timeout in milliseconds
 *                 default: 30000
 */
```

## Security and Safety Guidelines

### Input Validation
- **Always validate** user input before Pyodide execution
- **Sanitize file uploads** - check extensions, file sizes, content types
- **Implement timeouts** for all Python code execution
- **Rate limiting** for API endpoints

### Code Execution Safety
```javascript
// ✅ CORRECT - Safe execution with limits
const executionLimits = {
  timeout: 30000,           // 30 second timeout
  maxMemory: 512 * 1024,    // 512MB memory limit
  maxFileSize: 10 * 1024,   // 10MB file upload limit
};

// ❌ DANGEROUS - Unrestricted execution
pyodide.runPython(userCode); // No timeout, no validation
```

## Performance Optimization

### Pyodide Performance
- **Minimize package installations** during runtime
- **Reuse Pyodide instances** when possible
- **Implement caching** for frequently executed code
- **Monitor memory usage** and implement cleanup

### File System Optimization
- **Regular cleanup** of temporary files
- **Implement file rotation** for logs
- **Monitor disk space** usage
- **Optimize plot generation** (appropriate DPI, formats)

## Debugging and Troubleshooting

### Terminal Initialization Issues

⚠️ **CRITICAL: Always account for 15-second terminal startup delay**
```powershell
# ❌ WRONG - Immediate command execution assumption
run_in_terminal("npm start")
# Immediately checking for output will show empty/failed

# ✅ CORRECT - Wait for terminal initialization
run_in_terminal("npm start")
# Wait 15-20 seconds before checking output or running next command
get_terminal_output(terminal_id)
```

### Common Issues and Solutions

```powershell
# Server won't start
npm run clean && npm install          # Clean dependencies
Test-Path .\uploads, .\logs           # Check required directories
Get-Process -Name node                # Check for hanging processes

# Pyodide execution fails
# Check browser console if testing web interface
# Verify Python syntax in executed code
# Check package installation status

# Tests failing
python -m unittest tests.test_api.ApiTestCase.test_basic_execution -v
# Run individual test methods for debugging
```

### Logging and Monitoring

```javascript
// ✅ CORRECT - Structured logging
const logger = require('./utils/logger');

logger.info('Starting Pyodide initialization', {
  component: 'pyodide-service',
  action: 'initialize'
});

logger.error('Code execution failed', {
  component: 'pyodide-service',
  action: 'executeCode',
  error: error.message,
  code: code.substring(0, 100) // Log first 100 chars
});
```

## Code Quality Standards

### JavaScript/Node.js
- **Use modern ES6+** syntax with proper async/await
- **Implement proper error handling** with try/catch blocks
- **Follow Express.js conventions** for routing and middleware
- **Use consistent naming** conventions (camelCase)
- **Add JSDoc comments** for complex functions

### Python (Pyodide Execution)
- **Follow PEP 8** style guidelines
- **Use type hints** where appropriate
- **Import pathlib Path** for all file operations
- **Handle exceptions** gracefully
- **Add docstrings** to functions and classes

### Testing Code
- **Descriptive test names** that explain what's being tested
- **Proper setup and teardown** for test isolation
- **Assert meaningful conditions** with helpful error messages
- **Test both happy path and error conditions**

## Documentation Standards

### Code Documentation
- **README.md** with clear setup instructions
- **API documentation** using Swagger/OpenAPI
- **Inline comments** for complex logic
- **Architecture documentation** in `docs/` folder

### Commit Messages
```bash
feat: add matplotlib plot extraction API endpoint
fix: resolve Pyodide initialization timeout issue
test: add comprehensive filesystem mounting tests
docs: update API documentation with new endpoints
refactor: improve error handling in pyodide service
```

## Communication Style

- **Professional but approachable** tone
- **Use relevant emojis** for visual clarity (🚀, 📊, 🔧, ⚠️)
- **Quick and clever humor** when appropriate 🐍
- **Clear explanations** with practical examples
- **Step-by-step guidance** for complex procedures
