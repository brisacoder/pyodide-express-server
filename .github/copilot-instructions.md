# GitHub Copilot Instructions for Pyodide Express Server

## 🚀 Project Overview

This is a **production-ready Node.js Express server** that provides a secure REST API for executing Python code using **Pyodide** (Python in WebAssembly). The project enables comprehensive data science workflows with advanced security, filesystem mounting, package management, and extensive testing infrastructure.

**Core Technologies:**
- **Backend:** Node.js 18+ + Express.js with Helmet security
- **Python Runtime:** Pyodide (Python 3.11 in WebAssembly) 
- **Data Science Stack:** matplotlib, seaborn, pandas, scikit-learn, numpy
- **File Operations:** Multer uploads + direct filesystem mounting
- **API Documentation:** Swagger/OpenAPI w## 🎯 Communication & Engagement Style

- **Professional yet approachable** - technical accuracy with clear explanations
- **Strategic emoji usage** for visual organization and emphasis 🚀📊🔧⚠️
- **Code-first solutions** - show working examples rather than just theory
- **Step-by-step guidance** for complex procedures with validation steps
- **Proactive problem solving** - anticipate issues and provide solutions
- **Performance awareness** - always consider scalability and optimization

## 🔄 Recent Project Enhancements & Current State

### Code Quality Improvements (2025-08)
- **✅ Python Import Optimization**: Removed 11+ unused imports across test suite
- **✅ PEP 8 Compliance**: Applied standard library → third-party → local import ordering
- **✅ Flake8 Clean**: Zero linting violations across entire Python codebase
- **✅ Modern Tooling**: Complete migration from pip to uv package management

### Documentation & Examples Modernization
- **✅ API Accuracy**: Corrected all endpoint documentation to match actual implementation
- **✅ Comprehensive Examples**: Created 4 client integration examples:
  - `basic-client.js` - Simple API usage patterns
  - `data-science-client.js` - Advanced workflows with execute-raw endpoint
  - `file-upload-client.js` - Complete file processing workflows
  - `execute-raw-client.js` - Raw endpoint usage for complex Python code
- **✅ Documentation Consolidation**: Merged redundant documentation into single README

### Security & Testing Infrastructure
- **✅ Penetration Testing**: Comprehensive security validation across all endpoints
- **✅ Test Categories**: 50+ tests organized by functionality (basic, security, matplotlib, etc.)
- **✅ Cleanup Automation**: Robust test artifact management and resource cleanup
- **✅ Performance Monitoring**: Execution time tracking and resource optimization

### Production-Ready Features
- **✅ Structured Logging**: Winston-based logging with request context tracking
- **✅ Error Handling**: Comprehensive error management with proper HTTP status codes
- **✅ Validation Middleware**: Input validation with express-validator
- **✅ Security Headers**: Helmet.js implementation for production security
- **✅ File System Mounting**: Direct filesystem access for Pyodide with proper isolation

### Current Development Priorities
1. **Maintain Code Quality** - Continue PEP 8 compliance and modern patterns
2. **Expand Test Coverage** - Add edge cases and performance benchmarks  
3. **Performance Optimization** - Package caching and execution efficiency
4. **Security Hardening** - Regular security audits and vulnerability assessments
5. **Documentation Excellence** - Keep examples and API docs synchronized

### Key Success Metrics (Current)
- **🟢 Code Quality**: 0 flake8 violations, clean linting across all files
- **🟢 Test Reliability**: 100% test pass rate with proper cleanup
- **🟢 API Accuracy**: All documented endpoints verified and working
- **🟢 Modern Tooling**: uv for Python, npm for Node.js, PowerShell 7 ready
- **🟢 Production Ready**: Comprehensive security, logging, and error handling

---

## 📚 Quick Reference Commands

### Essential Daily Workflows
```powershell
# 🚀 Server Management
npm start                                    # Production server
npm run dev                                  # Development with auto-reload
curl http://localhost:3000/health           # Health check

# 🧪 Testing & Quality
uv run python run_simple_tests.py          # Quick tests (~30 tests)
uv run python run_comprehensive_tests.py   # Full suite (50+ tests)  
uv run flake8 tests/ --statistics          # Python code quality
npm run lint                                # JavaScript code quality

# 🔧 Development Setup
npm ci && npm run setup                     # Node.js setup + directories
uv sync                                     # Python environment setup
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Install uv
```

### Example API Usage
```javascript
// Execute Python code (JSON response)
fetch('/api/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: 'import pandas as pd\nprint(pd.__version__)',
    timeout: 30000
  })
});

// Execute complex Python (raw text response)  
fetch('/api/execute-raw', {
  method: 'POST', 
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: `
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.savefig(Path('/home/pyodide/plots/matplotlib/demo.png'))
print("Plot saved successfully!")
    `
  })
});
```

**Remember**: This is a production-ready system with enterprise-grade security, testing, and monitoring. Always follow the established patterns and maintain the high code quality standards! 🚀ractive UI
- **Testing Framework:** Python unittest + comprehensive integration tests
- **Package Management:** Node.js (npm) + Python (uv with pyproject.toml)
- **Development Environment:** Windows PowerShell 7, cross-platform support
- **Security:** Comprehensive penetration testing, input validation, timeout controls

## ⚡ Critical Development Environment

### Terminal Management Protocol

⚠️ **CRITICAL: PowerShell Terminal Initialization**
- **Every `run_in_terminal()` creates a NEW PowerShell terminal**
- **New terminals require 15-20 seconds initialization time**
- **Commands sent to uninitialized terminals are LOST**
- **Always test terminal readiness with `echo` before complex commands**

```powershell
# ✅ CORRECT - Terminal readiness verification
run_in_terminal("echo 'Terminal initialization check'")
# Wait for response before proceeding with actual commands
run_in_terminal("npm start")  # Now safe to execute

# ❌ WRONG - Immediate command execution
run_in_terminal("npm start")  # Creates new terminal, command likely lost
```

### Modern Package Management

**Node.js Dependencies:**
```powershell
npm ci                    # Install Node.js dependencies (lockfile-based)
npm run setup            # Complete project setup with directories
```

**Python Dependencies (uv + pyproject.toml):**
```powershell
# Install uv (modern Python package manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Python environment setup
uv sync                  # Install dependencies from pyproject.toml + create venv
uv run python --version  # Execute Python in managed environment
```

## 🎯 Core Workflows

### Server Operations

```powershell
# Production server management
npm start                           # Production server (localhost:3000)
npm run dev                         # Development with nodemon auto-reload
npm run dev:inspect                 # Development + Node.js debugger

# Health monitoring & status
curl http://localhost:3000/health         # Basic health check
curl http://localhost:3000/api/status     # Detailed server status
curl http://localhost:3000/api/metrics    # Performance metrics
```

### Testing & Quality Assurance

```powershell
# Python test suite (using uv)
uv run python run_simple_tests.py          # Quick smoke tests (~30 tests)
uv run python run_comprehensive_tests.py   # Full test suite (50+ tests)

# Targeted test categories
uv run python run_comprehensive_tests.py --categories basic integration security
uv run python run_comprehensive_tests.py --categories matplotlib seaborn sklearn

# Individual test modules
uv run python -m unittest tests.test_api -v
uv run python -m unittest tests.test_security_penetration -v

# Code quality
npm run lint                        # ESLint JavaScript code analysis
uv run flake8 tests/               # Python PEP 8 compliance check
```

### API Endpoint Usage

```javascript
// Core Python execution (JSON response)
POST /api/execute
{
  "code": "import pandas as pd\ndf = pd.DataFrame({'x': [1,2,3]})\nprint(df.head())",
  "timeout": 30000
}

// Raw Python execution (text/plain response)
POST /api/execute-raw  
{
  "code": "print('Hello from Pyodide!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
}

// Package management
POST /api/install-package
{
  "package": "scikit-learn"
}

// File operations
POST /api/upload          # Upload CSV files
GET /api/uploaded-files        # List uploaded files
DELETE /api/uploaded-files/:filename
GET /api/plots/extract         # Extract generated plots
```

## 🏗️ Architecture & Project Structure

### Current Project Organization
```
pyodide-express-server/
├── 📁 src/                          # Core application source
│   ├── app.js                       # Express app configuration + middleware
│   ├── server.js                    # Server startup and lifecycle management
│   ├── swagger-config.js            # OpenAPI/Swagger documentation config
│   ├── 📁 config/                   # Environment & application configuration
│   ├── 📁 controllers/              # Route handlers and business logic
│   ├── 📁 middleware/               # Express middleware (validation, security)
│   ├── 📁 routes/                   # API endpoint definitions
│   │   ├── execute.js               # /api/execute (JSON responses)
│   │   ├── executeRaw.js            # /api/execute-raw (text responses)
│   │   ├── files.js                 # File management endpoints
│   │   ├── health.js                # Health check endpoints
│   │   ├── stats.js                 # Statistics and metrics
│   │   └── upload.js                # File upload handling
│   ├── 📁 services/                 # Core business logic
│   │   └── pyodide-service.js       # Pyodide WebAssembly integration
│   └── 📁 utils/                    # Shared utilities
│       ├── logger.js                # Structured logging with Winston
│       ├── metrics.js               # Performance metrics collection
│       └── requestContext.js        # Request context tracking
├── 📁 tests/                        # Comprehensive Python test suite
│   ├── test_api.py                  # Core API functionality tests
│   ├── test_security_*.py           # Security and penetration testing
│   ├── test_matplotlib_*.py         # Data visualization tests
│   ├── test_integration.py          # End-to-end workflow tests
│   └── test_performance.py          # Load and timeout testing
├── 📁 examples/                     # Client integration examples
│   ├── basic-client.js              # Simple API usage demonstration
│   ├── data-science-client.js       # Advanced data science workflow
│   ├── execute-raw-client.js        # Raw endpoint usage patterns
│   └── file-upload-client.js        # File upload and processing
├── 📁 docs/                         # Project documentation
├── 📁 uploads/                      # User file upload storage
├── 📁 logs/                         # Server logs and audit trails
├── 📁 plots/                        # Generated visualization outputs
│   ├── matplotlib/                  # Matplotlib plot files
│   ├── seaborn/                     # Seaborn visualization files
│   └── base64/                      # Base64-encoded plot data
├── pyproject.toml                   # Python dependencies (uv managed)
├── package.json                     # Node.js dependencies and scripts
└── README.md                        # Primary documentation
```

### Key Design Patterns

**Express.js Application Structure:**
```javascript
// ✅ MODERN - Modular routing with middleware
const router = express.Router();
const { validateRequest, securityHeaders } = require('../middleware');

router.post('/api/execute', 
  validateRequest,
  securityHeaders,
  asyncHandler(async (req, res) => {
    const { code, timeout = 30000 } = req.body;
    const result = await pyodideService.executeCode(code, timeout);
    res.json({ success: true, result, executionTime: Date.now() - start });
  })
);
```

**Pyodide Service Integration:**
```javascript
// ✅ PRODUCTION-READY - Error handling, timeouts, cleanup
class PyodideService {
  async executeCode(code, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.cleanup();
        reject(new Error('Execution timeout'));
      }, timeout);
      
      try {
        const result = this.pyodide.runPython(code);
        clearTimeout(timer);
        resolve(result);
      } catch (error) {
        clearTimeout(timer);
        this.cleanup();
        reject(new Error(`Python execution failed: ${error.message}`));
      }
    });
  }
}
```

## 🐍 Python Code Standards for Pyodide Execution

### Modern Python Patterns (Required)

```python
# ✅ REQUIRED - pathlib for all file operations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# File operations using pathlib (MANDATORY)
# Files are saved in /home/pyodide/uploads inside Pyodide
data_path = Path('/home/pyodide/uploads/data.csv')
if data_path.exists():
    df = pd.read_csv(data_path)

# Plot generation with proper directory handling
plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
plot_file = plots_dir / f'analysis_{int(time.time())}.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')

# ❌ DEPRECATED - Old-style file operations (DO NOT USE)
import os
if os.path.exists('/uploads/data.csv'):  # Deprecated - use /home/pyodide/uploads
    df = pd.read_csv('/uploads/data.csv')  # Deprecated
os.makedirs('/plots/matplotlib', exist_ok=True)  # Use pathlib instead
```

### Import Organization (PEP 8 Compliance)

```python
# ✅ CORRECT - Standard library imports first
import json
import time
from pathlib import Path

# Third-party imports second  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports last (if any)
# from my_module import my_function
```

## 🧪 Testing & Quality Assurance Strategy

### Test Suite Architecture

**Core Test Categories:**
1. **`test_api.py`** - Basic API functionality and response validation
2. **`test_security_*.py`** - Security validation, penetration testing, input sanitization
3. **`test_performance.py`** - Timeout handling, resource limits, load testing
4. **`test_integration.py`** - End-to-end workflows and complex scenarios
5. **`test_matplotlib_*.py`** - Data visualization and plotting functionality
6. **`test_seaborn_*.py`** - Advanced visualization libraries
7. **`test_virtual_filesystem.py`** - File system operations and mounting

### Testing Execution Patterns

```powershell
# Quick validation (30+ core tests)
uv run python run_simple_tests.py

# Comprehensive testing (50+ tests with categories)
uv run python run_comprehensive_tests.py

# Targeted category testing
uv run python run_comprehensive_tests.py --categories basic security
uv run python run_comprehensive_tests.py --categories matplotlib integration

# Individual test debugging
uv run python -m unittest tests.test_api.ApiTestCase.test_basic_execution -v
uv run python -m unittest tests.test_security_penetration -v

# Code quality validation
uv run flake8 tests/ --statistics
npm run lint
```

### Test Data Management & Cleanup

```python
# ✅ PRODUCTION PATTERN - Proper test isolation
class ApiTestCase(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with tracking"""
        self.uploaded_files = []
        self.temp_files = []
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up all test artifacts"""
        # Remove uploaded files via API
        for filename in self.uploaded_files:
            try:
                requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}")
            except:
                pass
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            if isinstance(temp_file, Path) and temp_file.exists():
                temp_file.unlink()
    
    def track_upload(self, filename):
        """Track uploaded files for cleanup"""
        self.uploaded_files.append(filename)
```

## 🔌 API Design & Integration Standards

### Complete API Endpoint Reference

| Method | Endpoint | Description | Request Body | Response Format |
|--------|----------|-------------|--------------|-----------------|
| **POST** | `/api/execute` | Execute Python code (JSON response) | `{"code": "...", "timeout": 30000}` | `{"success": true, "result": {...}}` |
| **POST** | `/api/execute-raw` | Execute Python code (text response) | `{"code": "...", "timeout": 30000}` | Plain text output |
| **POST** | `/api/install-package` | Install Python package | `{"package": "scikit-learn"}` | `{"success": true, "message": "..."}` |
| **POST** | `/api/upload` | Upload CSV file | `multipart/form-data` | `{"success": true, "filename": "..."}` |
| **GET** | `/api/uploaded-files` | List uploaded files | None | `{"files": [...]}` |
| **DELETE** | `/api/uploaded-files/:filename` | Delete uploaded file | None | `{"success": true}` |
| **GET** | `/api/plots/extract` | Extract generated plots | None | `{"plots": {...}}` |
| **GET** | `/health` | Basic health check | None | `{"status": "ok"}` |
| **GET** | `/api/status` | Detailed server status | None | `{"status": "...", "uptime": ...}` |
| **GET** | `/api/metrics` | Performance metrics | None | `{"requests": ..., "errors": ...}` |
| **POST** | `/api/stats/clear` | Clear server statistics | None | `{"success": true}` |

### Response Pattern Standards

```javascript
// ✅ STANDARD SUCCESS RESPONSE
{
  "success": true,
  "result": { /* actual data */ },
  "metadata": {
    "executionTime": 1234,
    "timestamp": "2025-08-20T10:30:00Z",
    "requestId": "uuid-string"
  }
}

// ✅ STANDARD ERROR RESPONSE  
{
  "success": false,
  "error": "Descriptive error message",
  "code": "EXECUTION_TIMEOUT|INVALID_INPUT|PACKAGE_INSTALL_FAILED",
  "details": {
    "originalError": "...",
    "suggestions": ["Try reducing timeout", "Check package name"]
  }
}
```

### Client Integration Examples

The project includes comprehensive client examples in `examples/`:

```javascript
// examples/basic-client.js - Simple API usage
const response = await fetch('http://localhost:3000/api/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: 'import pandas as pd\nprint(pd.__version__)',
    timeout: 30000
  })
});

// examples/data-science-client.js - Advanced workflows
// examples/execute-raw-client.js - Raw endpoint usage  
// examples/file-upload-client.js - File processing workflows
```

## 🔒 Security & Performance Guidelines

### Input Validation & Safety

```javascript
// ✅ PRODUCTION SECURITY PATTERN
const { body, validationResult } = require('express-validator');

const executeValidation = [
  body('code')
    .isString()
    .isLength({ min: 1, max: 50000 })
    .withMessage('Code must be 1-50000 characters'),
  body('timeout')
    .optional()
    .isInt({ min: 1000, max: 60000 })
    .withMessage('Timeout must be 1-60 seconds'),
];

// ✅ EXECUTION SAFETY CONTROLS
const executionLimits = {
  timeout: 30000,           // 30 second max timeout
  maxMemory: 512 * 1024,    // 512MB memory limit  
  maxFileSize: 10 * 1024,   // 10MB file upload limit
  rateLimitRpm: 60,         // 60 requests per minute
};
```

### Performance Optimization Strategies

```javascript
// ✅ PYODIDE PERFORMANCE OPTIMIZATION
class PyodideService {
  constructor() {
    this.packageCache = new Map();
    this.executionCache = new LRUCache({ max: 100 });
  }

  async installPackage(packageName) {
    // Cache package installations to avoid repeated downloads
    if (this.packageCache.has(packageName)) {
      return this.packageCache.get(packageName);
    }
    
    const result = await this.pyodide.loadPackage(packageName);
    this.packageCache.set(packageName, result);
    return result;
  }

  async executeWithCleanup(code, timeout) {
    try {
      return await this.executeCode(code, timeout);
    } finally {
      // Cleanup memory and temporary variables
      this.pyodide.runPython(`
        import gc
        import sys
        # Clear variables except core modules
        for name in list(locals().keys()):
          if not name.startswith('_'):
            del locals()[name]
        gc.collect()
      `);
    }
  }
}
```

## 🛠️ Development Workflow & Best Practices

### Automatic Task Completion Protocol
- **Always iterate and continue working** until tasks are fully complete
- **Never ask permission** - proceed with implementation immediately
- **Keep working through problems** - test, debug, and refine until objectives achieved
- **Run tests after changes** to validate functionality
- **Use modern tooling** - uv for Python, npm for Node.js, proper terminal handling

### Communication & Documentation Standards
- **Markdown formatting** for all explanations with proper syntax highlighting
- **File paths** in backticks: `src/server.js`, `tests/test_api.py`
- **API endpoints** clearly formatted: `POST /api/execute`
- **Code blocks** properly tagged: `javascript`, `python`, `powershell`, `json`
- **Professional tone** with strategic emoji usage for clarity 🚀📊🔧⚠️

### Windows PowerShell 7 Development Environment
```powershell
# ✅ MODERN PowerShell patterns
Remove-Item .\file1.js, .\file2.py -Force        # Multiple file deletion
Get-ChildItem -Path .\logs\ -Recurse              # Directory listing
Test-Path .\uploads\data.csv                      # File existence check
Select-String -Path .\src\*.js -Pattern "TODO"    # Code search patterns
```

### Cross-Platform Path Management
```javascript
// ✅ Node.js path handling
const path = require('path');
const filePath = path.join(__dirname, 'uploads', filename);
const configPath = path.resolve('./config/app.json');

// ✅ Python pathlib (Pyodide execution)
from pathlib import Path
plots_dir = Path('/home/pyodide/plots/matplotlib')
data_file = Path('/home/pyodide/uploads') / 'analysis.csv'
```

## 🐛 Debugging & Troubleshooting Guide

### Terminal Initialization & Command Execution

⚠️ **CRITICAL: Always test terminal readiness before complex operations**

```powershell
# ✅ SAFE TERMINAL PATTERN
# Step 1: Create terminal with simple command
run_in_terminal("echo 'Initializing terminal for operations'")

# Step 2: Wait and verify terminal responds (15-20 seconds)
get_terminal_output(terminal_id)  # Should show echo response

# Step 3: Execute actual commands once verified ready
run_in_terminal("npm start")      # Now safe to execute
run_in_terminal("uv run python run_simple_tests.py")

# ❌ DANGEROUS PATTERN - Commands lost
run_in_terminal("npm start")      # New terminal, command likely lost
run_in_terminal("curl localhost:3000/health")  # Another new terminal, lost
```

### Common Issue Resolution

```powershell
# 🔧 Server startup issues
npm run clean && npm ci           # Clean reinstall dependencies
Test-Path .\uploads, .\logs, .\plots  # Verify required directories exist
Get-Process -Name node | Stop-Process -Force  # Kill hanging Node processes

# 🔧 Python environment issues  
uv sync                          # Reinstall Python dependencies
uv run python --version         # Verify Python environment
uv run flake8 tests/ --statistics  # Check code quality

# 🔧 Pyodide execution debugging
# Check browser console for WebAssembly errors
# Verify Python syntax before sending to API
# Monitor server logs for execution failures

# 🔧 Test failures investigation
uv run python -m unittest tests.test_api.ApiTestCase.test_basic_execution -v
# Run specific failing tests in isolation
# Check test cleanup and artifact removal
```

### Structured Logging & Monitoring

```javascript
// ✅ PRODUCTION LOGGING PATTERN
const logger = require('./utils/logger');

// Structured logging with context
logger.info('Pyodide service initialization started', {
  component: 'pyodide-service',
  action: 'initialize',
  timestamp: new Date().toISOString(),
  environment: process.env.NODE_ENV
});

// Error logging with full context
logger.error('Python code execution failed', {
  component: 'pyodide-service', 
  action: 'executeCode',
  error: error.message,
  stack: error.stack,
  code: code.substring(0, 200), // First 200 chars for debugging
  timeout: timeout,
  requestId: req.id
});

// Performance monitoring
logger.debug('Execution completed', {
  component: 'pyodide-service',
  action: 'executeCode', 
  executionTime: Date.now() - startTime,
  codeLength: code.length,
  success: true
});
```

## 📋 Code Quality & Standards

### JavaScript/Node.js Excellence
```javascript
// ✅ MODERN EXPRESS.JS PATTERNS
const express = require('express');
const { body, validationResult } = require('express-validator');

// Async error handling wrapper
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Route with validation and error handling
router.post('/api/execute',
  [
    body('code').isString().isLength({ min: 1, max: 50000 }),
    body('timeout').optional().isInt({ min: 1000, max: 60000 })
  ],
  asyncHandler(async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        success: false, 
        error: 'Validation failed',
        details: errors.array() 
      });
    }

    const { code, timeout = 30000 } = req.body;
    const startTime = Date.now();
    
    try {
      const result = await pyodideService.executeCode(code, timeout);
      res.json({
        success: true,
        result,
        metadata: {
          executionTime: Date.now() - startTime,
          timestamp: new Date().toISOString()
        }
      });
    } catch (error) {
      logger.error('Execution failed', { error: error.message, code: code.substring(0, 100) });
      res.status(500).json({
        success: false,
        error: error.message,
        code: 'EXECUTION_FAILED'
      });
    }
  })
);
```

### Python Testing Excellence
```python
# ✅ COMPREHENSIVE TEST PATTERNS
import unittest
import requests
import tempfile
import time
from pathlib import Path

class ApiTestCase(unittest.TestCase):
    """Comprehensive API testing with proper cleanup"""
    
    @classmethod
    def setUpClass(cls):
        """One-time setup for test class"""
        cls.base_url = "http://localhost:3000"
        cls.session = requests.Session()
        cls.session.timeout = 30
    
    def setUp(self):
        """Per-test setup with tracking"""
        self.uploaded_files = []
        self.temp_files = []
        self.start_time = time.time()
        
    def tearDown(self):
        """Comprehensive cleanup after each test"""
        # Clean uploaded files via API
        for filename in self.uploaded_files:
            try:
                self.session.delete(f"{self.base_url}/api/uploaded-files/{filename}")
            except requests.RequestException:
                pass  # File might already be deleted
        
        # Clean temporary files
        for temp_file in self.temp_files:
            if isinstance(temp_file, Path) and temp_file.exists():
                temp_file.unlink()
                
        # Log test duration for performance monitoring  
        duration = time.time() - self.start_time
        if duration > 10:  # Log slow tests
            print(f"SLOW TEST: {self._testMethodName} took {duration:.2f}s")
    
    def test_comprehensive_workflow(self):
        """Test complete data science workflow"""
        # Test package installation
        response = self.session.post(f"{self.base_url}/api/install-package", 
                                   json={"package": "matplotlib"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        
        # Test code execution with visualization
        python_code = """
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Sine Wave Visualization')
plt.legend()
plt.grid(True, alpha=0.3)

# Save to mounted filesystem
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / 'test_sine_wave.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Plot saved successfully at {plots_dir / 'test_sine_wave.png'}")
        """
        
        response = self.session.post(f"{self.base_url}/api/execute",
                                   json={"code": python_code, "timeout": 30000})
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertTrue(result["success"])
        self.assertIn("Plot saved successfully", result["result"])
        
        # Verify plot was created
        plots_response = self.session.get(f"{self.base_url}/api/plots/extract")
        self.assertEqual(plots_response.status_code, 200)
        plots_data = plots_response.json()
        self.assertIn("matplotlib", plots_data.get("plots", {}))
```

### Documentation & Version Control
```bash
# ✅ SEMANTIC COMMIT MESSAGES
git commit -m "feat: add comprehensive matplotlib plot extraction API"
git commit -m "fix: resolve Pyodide initialization timeout in production"  
git commit -m "test: add security penetration testing for file uploads"
git commit -m "docs: update API documentation with new endpoint examples"
git commit -m "refactor: improve error handling in pyodide service layer"
git commit -m "perf: optimize package caching for faster execution"
git commit -m "security: implement rate limiting for code execution endpoints"
```

## 🎯 Communication & Engagement Style

- **Professional yet approachable** - technical accuracy with clear explanations
- **Strategic emoji usage** for visual organization and emphasis 🚀📊�⚠️
- **Code-first solutions** - show working examples rather than just theory
- **Step-by-step guidance** for complex procedures with validation steps
- **Proactive problem solving** - anticipate issues and provide solutions
- **Performance awareness** - always consider scalability and optimization
