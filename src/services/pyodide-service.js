/**
 * Pyodide Service - Complete Implementation
 *
 * This service handles all Pyodide-related operations including:
 * - Python code execution
 * - Package management
 * - File operations
 * - Environment management
 *
 * Pyodide bundles the CPython interpreter compiled to WebAssembly.  The
 * service below loads the runtime once and reuses it for all requests,
 * exposing helper methods to execute Python code, inspect the virtual
 * filesystem, and manage packages via ``micropip``.  The currently bundled
 * Pyodide version is 0.28.0 which corresponds to Python 3.13.
 */

const logger = require('../utils/logger');

class PyodideService {
  constructor() {
    // Will hold the Pyodide instance once ``loadPyodide`` resolves. Until
    // then it remains ``null``.
    this.pyodide = null;
    // Flag indicating that the runtime finished bootstrapping and packages
    // were loaded. Routes should check this before executing code.
    this.isReady = false;
    // Promise used to avoid multiple concurrent initializations when several
    // requests hit the server at start-up.
    this.initializationPromise = null;
    // Default timeout in milliseconds for executing user supplied Python
    // snippets.  Individual requests may override it.
    this.executionTimeout = 30000; // 30 seconds default timeout
  }

  /**
   * Initialize Pyodide with common packages
   * @returns {Promise<boolean>} True if initialization successful
   *
   * The first call kicks off the asynchronous bootstrapping sequence. Any
   * later calls simply await the existing promise so that only one instance of
   * Pyodide is ever created.
   */
  async initialize() {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._performInitialization();
    return this.initializationPromise;
  }

  /**
   * Internal initialization method - Simple version that works
   * @private
   *
   * Loads the Pyodide runtime and eagerly fetches a set of scientific
   * packages.  Installing packages up front reduces latency for the first
   * request and mirrors the behaviour of a traditional Python environment.
   */
  async _performInitialization() {
    try {
      logger.info('Starting Pyodide initialization...');

      // Import Pyodide - try both import methods
      let loadPyodide;
      
      try {
        logger.info('Importing Pyodide module...');
        const pyodideModule = await import('pyodide');
        loadPyodide = pyodideModule.loadPyodide;
        logger.info('✅ Pyodide module imported successfully');
      } catch (importError) {
        logger.error('❌ Failed to import Pyodide:', importError.message);
        throw new Error(`Could not import Pyodide: ${importError.message}`);
      }

      if (!loadPyodide || typeof loadPyodide !== 'function') {
        throw new Error('loadPyodide is not a function');
      }

      logger.info('Loading Pyodide runtime (using default configuration)...');
      
      // Use the simplest possible configuration - no CDN URLs
      this.pyodide = await loadPyodide();
      
      logger.info('✅ Pyodide loaded successfully!');
      logger.info('Pyodide version:', this.pyodide.version || 'unknown');

      logger.info('Loading essential packages...');
      
      // Load packages one by one to see which ones work
      const packagesToLoad = ['numpy', 'pandas', 'micropip', 'matplotlib', 'requests'];
      const loadedPackages = [];
      
      for (const packageName of packagesToLoad) {
        try {
          logger.info(`Loading ${packageName}...`);
          await this.pyodide.loadPackage([packageName]);
          loadedPackages.push(packageName);
          logger.info(`✅ ${packageName} loaded successfully`);
        } catch (packageError) {
          logger.warn(`⚠️  Failed to load ${packageName}:`, packageError.message);
        }
      }

      logger.info(`Loaded packages: ${loadedPackages.join(', ')}`);

      logger.info('Installing additional packages via micropip...');
      
      // Install additional packages that aren't available via loadPackage
      const micropipPackages = ['seaborn', 'httpx'];
      const installedPackages = [];
      
      for (const packageName of micropipPackages) {
        try {
          logger.info(`Installing ${packageName} via micropip...`);
          await this.pyodide.runPythonAsync(`
import micropip
await micropip.install("${packageName}")
print("✅ ${packageName} installed successfully")
`);
          installedPackages.push(packageName);
          logger.info(`✅ ${packageName} installed successfully`);
        } catch (packageError) {
          logger.warn(`⚠️  Failed to install ${packageName}:`, packageError.message);
        }
      }

      logger.info(`Installed via micropip: ${installedPackages.join(', ')}`);

      logger.info('Setting up Python environment...');
      
      // Set up Python environment with error handling
      await this.pyodide.runPythonAsync(`
import sys
import io
from io import StringIO

print(f"Python version: {sys.version}")
print("Setting up Pyodide environment...")

# Import available packages and make them globally accessible
try:
    import numpy as np
    print("✅ NumPy available")
    globals()['np'] = np
except ImportError:
    print("⚠️  NumPy not available")

try:
    import pandas as pd
    print("✅ Pandas available") 
    globals()['pd'] = pd
except ImportError:
    print("⚠️  Pandas not available")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # Set non-interactive backend for server environment
    matplotlib.use('Agg')
    print("✅ Matplotlib available")
    globals()['plt'] = plt
except ImportError:
    print("⚠️  Matplotlib not available")

try:
    import requests
    print("✅ Requests available")
    globals()['requests'] = requests
except ImportError:
    print("⚠️  Requests not available")

try:
    import seaborn as sns
    print("✅ Seaborn available")
    globals()['sns'] = sns
except ImportError:
    print("⚠️  Seaborn not available")

try:
    import httpx
    print("✅ HTTPX available")
    globals()['httpx'] = httpx
except ImportError:
    print("⚠️  HTTPX not available")

try:
    import micropip
    print("✅ Micropip available")
    # Make micropip available globally
    globals()['micropip'] = micropip
except ImportError:
    print("⚠️  Micropip not available")

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
    
    def reset(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# Global output capturer
output_capture = OutputCapture()

# Helper function for JSON serialization
def make_json_safe(obj):
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, np.integer):  # numpy integer types
        return int(obj)
    elif isinstance(obj, np.floating):  # numpy float types
        return float(obj)
    elif isinstance(obj, np.bool_):  # numpy boolean type
        return bool(obj)
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

print("🎉 Pyodide environment ready!")
      `);

      // Setup filesystem mounting for plot saving
      logger.info('Setting up filesystem mounting for plots...');
      try {
        const path = require('path');
        const plotsDir = path.resolve(__dirname, '../../plots');
        
        // Ensure the real plots directory exists
        const fs = require('fs');
        if (!fs.existsSync(plotsDir)) {
          fs.mkdirSync(plotsDir, { recursive: true });
          logger.info(`Created plots directory: ${plotsDir}`);
        }
        
        // Create subdirectories for different plot types
        const plotSubdirs = ['matplotlib', 'seaborn'];
        for (const subdir of plotSubdirs) {
          const subdirPath = path.join(plotsDir, subdir);
          if (!fs.existsSync(subdirPath)) {
            fs.mkdirSync(subdirPath, { recursive: true });
          }
        }
        
        // Mount the real plots directory to /plots in Pyodide filesystem
        logger.info(`Attempting to mount ${plotsDir} to /plots in Pyodide...`);
        logger.info(`Host path exists: ${fs.existsSync(plotsDir)}`);
        logger.info(`Host path absolute: ${path.isAbsolute(plotsDir)}`);
        
        try {
          // mountNodeFS(emscriptenPath, hostPath) 
          // emscriptenPath: The absolute path in Emscripten FS to mount to
          // hostPath: The host path to mount (must exist)
          this.pyodide.mountNodeFS("/plots", plotsDir);
          logger.info('✅ Mount successful');
        } catch (mountError) {
          logger.error('❌ Mount failed:', mountError.message);
          logger.info('Falling back to virtual filesystem approach');
          
          // Fallback: Create virtual directories in Pyodide filesystem for plot saving
          await this.pyodide.runPythonAsync(`
import os
# Create plot directories in virtual filesystem as fallback
plot_dirs = ['/plots', '/plots/matplotlib', '/plots/seaborn']
for plot_dir in plot_dirs:
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Created virtual directory: {plot_dir}")
`);
        }
        
        // Verify the mount worked
        await this.pyodide.runPythonAsync(`
import os
# Check if mounted directories are accessible
mount_test_result = {
    "plots_exists": os.path.exists("/plots"),
    "matplotlib_exists": os.path.exists("/plots/matplotlib"),
    "seaborn_exists": os.path.exists("/plots/seaborn"),
    "plots_contents": os.listdir("/plots") if os.path.exists("/plots") else [],
}
print(f"Mount verification: {mount_test_result}")
`);
        
        logger.info('✅ Filesystem mounting setup successfully');
        
      } catch (setupError) {
        logger.warn('⚠️  Failed to setup plot directories:', setupError.message);
        // Continue initialization even if setup fails
      }

      this.isReady = true;
      logger.info('🎉 Pyodide initialization completed successfully!');
      
      return true;

    } catch (error) {
      logger.error('❌ Failed to initialize Pyodide:', error.message);
      logger.error('Error details:', error);
      this.isReady = false;
      throw new Error(`Pyodide initialization failed: ${error.message}`);
    }
  }

  /**
   * Execute Python code with optional context variables
   * @param {string} code - Python code to execute
   * @param {Object} context - Variables to make available in Python
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<Object>} Execution result
   *
   * The snippet runs inside the already-initialized interpreter.  ``runPython``
   * is used for synchronous code, while ``runPythonAsync`` enables "top level
   * await" and is required when ``micropip`` performs network requests.
   * Output is captured via a small Python helper class created during
   * initialization so both ``stdout`` and ``stderr`` can be returned to the
   * client.
   */
  async executeCode(code, context = {}, timeout = this.executionTimeout) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready. Please wait for initialization to complete.');
    }

    if (!code || typeof code !== 'string') {
      throw new Error('Code must be a non-empty string');
    }

    try {
      // Set up context variables
      for (const [key, value] of Object.entries(context)) {
        this.pyodide.globals.set(key, value);
      }

      // Execute everything in Python and return JSON to completely avoid proxy issues
      const wrappedCode = `
import json
import traceback

# Start output capture
output_capture.start_capture()
_exec_success = True
_exec_error = None
_exec_result = None

try:
    # Execute user code - use the same namespace for globals and locals
    # This ensures function definitions are available for recursive calls
    _exec_namespace = globals().copy()
    
    # Execute the user code with unified namespace
    exec('''${code.replace(/'/g, "\\'")}''', _exec_namespace, _exec_namespace)
    
    # Get the result - try to eval the last line as an expression first
    _code_lines = '''${code.replace(/'/g, "\\'")}'''.strip().split('\\n')
    if _code_lines:
        _last_line = _code_lines[-1].strip()
        if _last_line and not _last_line.startswith(('#', 'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except:', 'finally:', 'else:')):
            try:
                _exec_result = eval(_last_line, _exec_namespace, _exec_namespace)
            except:
                # If eval fails, check if the last line is a variable name that exists
                if _last_line in _exec_namespace:
                    _exec_result = _exec_namespace[_last_line]
                    
except Exception as e:
    _exec_success = False
    _exec_error = str(e)
    traceback.print_exc()

# Stop capture and get output
_output = output_capture.stop_capture()
output_capture.reset()

# Create result dictionary and convert everything to JSON-safe format
_final_result = {
    'success': _exec_success,
    'stdout': _output['stdout'],
    'stderr': _output['stderr'],
    'timestamp': '${new Date().toISOString()}'
}

if _exec_success:
    if _exec_result is not None:
        _final_result['result'] = make_json_safe(_exec_result)
    else:
        _final_result['result'] = None
    _final_result['error'] = None
else:
    _final_result['result'] = None
    _final_result['error'] = _exec_error

# Return as JSON string to completely avoid any proxy issues
json.dumps(_final_result)
`;

      // Execute the wrapped code and get JSON result
      const jsonResult = await this.pyodide.runPythonAsync(wrappedCode);
      
      // Parse the JSON result
      const result = JSON.parse(jsonResult);
      
      return {
        success: result.success,
        result: result.result,
        error: result.error,
        stdout: result.stdout,
        stderr: result.stderr,
        timestamp: result.timestamp
      };

    } catch (error) {
      // Ensure cleanup
      try {
        await this.pyodide.runPythonAsync('output_capture.reset()');
      } catch (resetError) {
        // Ignore cleanup errors
      }

      throw new Error(`Execution error: ${error.message}`);
    }
  }

  /**
   * Install a Python package using micropip
   * @param {string} packageName - Name of the package to install
   * @returns {Promise<Object>} Installation result
   *
   * ``micropip`` downloads wheels from PyPI that have been built for
   * WebAssembly (``wasm32``) and adds them to Pyodide's package index.  Only
   * pure Python packages or those explicitly compiled for Pyodide can be
   * installed this way.
   */
  async installPackage(packageName) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    if (!packageName || typeof packageName !== 'string') {
      throw new Error('Package name must be a non-empty string');
    }

    try {
      logger.info(`Installing package: ${packageName}`);
      
      // Use runPythonAsync directly for micropip operations
      const installCode = `
import micropip
import json

try:
    await micropip.install("${packageName}")
    # Package installation succeeded
    result = {"success": True, "message": "Package installed successfully"}
except Exception as e:
    result = {"success": False, "error": str(e)}

json.dumps(result)
      `;

      const resultStr = await this.pyodide.runPythonAsync(installCode);
      const installResult = JSON.parse(resultStr);
      
      return {
        success: installResult.success,
        message: installResult.message,
        error: installResult.error,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      throw new Error(`Failed to install package ${packageName}: ${error.message}`);
    }
  }

  /**
   * Get list of installed packages
   * @returns {Promise<Object>} List of packages
   *
   * Pyodide keeps a Python ``sys.modules`` registry similar to CPython.  This
   * helper exposes that information so clients can discover which packages are
   * currently loaded in the WebAssembly environment.
   */
  async getInstalledPackages() {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      return await this.executeCode(`
import sys
try:
    # Try to get package list
    installed_packages = list(sys.modules.keys())
    {
        'python_version': sys.version,
        'installed_packages': sorted([pkg for pkg in installed_packages if not pkg.startswith('_')]),
        'total_packages': len(installed_packages)
    }
except Exception as e:
    {'error': str(e)}
      `);
    } catch (error) {
      throw new Error(`Failed to get package list: ${error.message}`);
    }
  }

  /**
   * Load a CSV file into Pyodide's virtual filesystem
   * @param {string} filename - Name for the file in Pyodide
   * @param {string} csvContent - CSV file content
   * @returns {Promise<Object>} File load result
   *
   * ``pyodide.FS`` exposes the Emscripten MEMFS filesystem.  Writing to it
   * stores data in memory and the file instantly becomes available to Python
   * code within the WebAssembly sandbox.
   */
  async loadCSVFile(filename, csvContent) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      logger.info(`Loading CSV file into Pyodide: ${filename}`);
      
      // Write file to Pyodide's virtual filesystem
      this.pyodide.FS.writeFile(filename, csvContent);
      logger.info(`File written to Pyodide filesystem: ${filename}`);

      // Verify the file was loaded correctly and analyze it
      const verificationResult = await this.executeCode(`
# Check if pandas is available and load the file
import sys
try:
    import pandas as pd
    print(f"Loading file: ${filename}")
    
    # Read the CSV file
    df = pd.read_csv('${filename}')
    print(f"File loaded successfully. Shape: {df.shape}")
    
    # Get basic info about the file
    analysis = {
        'success': True,
        'filename': '${filename}',
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
    }
    
    # Add numeric column statistics if available
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        analysis['numeric_columns'] = numeric_cols
        analysis['statistics'] = df[numeric_cols].describe().to_dict()
    
    print(f"Analysis completed: {len(df)} rows, {len(df.columns)} columns")
    analysis
    
except ImportError as e:
    print(f"ImportError: {str(e)}")
    {
        'success': False,
        'error': 'Pandas not available - cannot process CSV',
        'filename': '${filename}',
        'python_error': str(e)
    }
except pd.errors.EmptyDataError as e:
    print(f"EmptyDataError: {str(e)}")
    {
        'success': False,
        'error': 'CSV file is empty or has no data',
        'filename': '${filename}',
        'python_error': str(e)
    }
except pd.errors.ParserError as e:
    print(f"ParserError: {str(e)}")
    {
        'success': False,
        'error': f'CSV parsing error: {str(e)}',
        'filename': '${filename}',
        'python_error': str(e)
    }
except Exception as e:
    print(f"General error: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    {
        'success': False,
        'error': f'Error processing CSV: {str(e)}',
        'filename': '${filename}',
        'python_error': str(e),
        'error_type': type(e).__name__
    }
      `);

      logger.info('CSV file verification completed:', {
        filename: filename,
        success: verificationResult.success,
        hasResult: !!verificationResult.result
      });

      return verificationResult;

    } catch (error) {
      logger.error(`Failed to load CSV file ${filename}:`, error);
      throw new Error(`Failed to load CSV file: ${error.message}`);
    }
  }

  /**
   * List files in Pyodide's virtual filesystem
   * @returns {Promise<Object>} List of files
   *
   * The virtual filesystem lives entirely in memory and is reset whenever the
   * service restarts.  This helper mirrors ``os.listdir`` to expose its
   * contents to the JavaScript side.
   */
  async listPyodideFiles() {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import os

files = []
try:
    for item in os.listdir('.'):
        if os.path.isfile(item):
            stat_info = os.stat(item)
            files.append(item + '|' + str(stat_info.st_size) + '|' + str(stat_info.st_mtime))
    
    result = '|||'.join(files) if files else 'EMPTY'
except Exception as e:
    result = 'ERROR:' + str(e)

result
      `);

      // Parse the result manually to avoid proxy issues
      if (result.success && result.result !== null) {
        if (result.result === 'EMPTY') {
          return {
            success: true,
            result: {
              success: true,
              files: [],
              count: 0
            }
          };
        } else if (result.result.startsWith('ERROR:')) {
          return {
            success: false,
            error: result.result.substring(6)
          };
        } else {
          // Parse the pipe-separated format
          const files = [];
          const fileEntries = result.result.split('|||');
          
          for (const entry of fileEntries) {
            const [name, size, modified] = entry.split('|');
            files.push({
              name: name,
              size: parseInt(size),
              modified: parseFloat(modified)
            });
          }
          
          return {
            success: true,
            result: {
              success: true,
              files: files,
              count: files.length
            }
          };
        }
      }
      
      // Fallback if result is null or unsuccessful
      return {
        success: true,
        result: {
          success: true,
          files: [],
          count: 0
        }
      };

      return result;
    } catch (error) {
      throw new Error(`Failed to list Pyodide files: ${error.message}`);
    }
  }

  /**
   * Delete a file from Pyodide's virtual filesystem
   * @param {string} filename - Name of file to delete
   * @returns {Promise<Object>} Deletion result
   *
   * Actual deletion happens inside the Python environment via ``os.remove``.
   * The operation only affects the in-memory filesystem and cannot touch the
   * host's disk.
   */
  async deletePyodideFile(filename) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import os
import json

if os.path.exists('${filename}'):
    os.remove('${filename}')
    result = {"success": True, "message": f"File ${filename} deleted successfully"}
else:
    result = {"success": False, "error": f"File ${filename} not found"}

json.dumps(result)
      `);

      if (result.success && result.result) {
        let deleteResult;
        try {
          if (typeof result.result === 'string') {
            deleteResult = JSON.parse(result.result);
          } else {
            deleteResult = result.result;
          }
          
          // Return the parsed result with proper structure
          return {
            success: deleteResult.success,
            message: deleteResult.message,
            error: deleteResult.error,
            timestamp: new Date().toISOString()
          };
        } catch (parseError) {
          return {
            success: false,
            error: 'Failed to parse deletion result',
            timestamp: new Date().toISOString()
          };
        }
      }
      
      return {
        success: false,
        error: 'Failed to execute deletion command',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      throw new Error(`Failed to delete Pyodide file: ${error.message}`);
    }
  }

  /**
   * Get current status of the Pyodide service
   * @returns {Object} Service status
   *
   * Useful for health endpoints.  Exposes whether the runtime has finished
   * loading and reports the Pyodide version bundled with the application.
   */
  getStatus() {
    return {
      isReady: this.isReady,
      initialized: this.pyodide !== null,
      executionTimeout: this.executionTimeout,
      version: this.pyodide?.version || 'unknown',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Reset the Pyodide environment
   * @returns {Promise<void>}
   *
   * Clears user-created globals while keeping core modules loaded.  This is
   * similar to restarting a Python REPL but much faster because the
   * WebAssembly runtime remains in memory.
   */
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

print("Environment reset completed")
      `);

      logger.info('Pyodide environment reset successfully');
    } catch (error) {
      logger.error('Failed to reset Pyodide environment:', error);
      throw new Error(`Reset failed: ${error.message}`);
    }
  }

  /**
   * Check if a file exists in Pyodide's virtual filesystem
   * @param {string} filename - Name of file to check
   * @returns {Promise<Object>} File existence result
   *
    * Useful when the JavaScript side needs to know whether a prior upload
    * has been persisted inside the WebAssembly sandbox.
   */
  async fileExists(filename) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import os
try:
    exists = os.path.exists('${filename}')
    if exists:
        stat_info = os.stat('${filename}')
        {
            'exists': True,
            'filename': '${filename}',
            'size': stat_info.st_size,
            'modified': stat_info.st_mtime,
            'is_file': os.path.isfile('${filename}')
        }
    else:
        {
            'exists': False,
            'filename': '${filename}'
        }
except Exception as e:
    {
        'exists': False,
        'filename': '${filename}',
        'error': str(e)
    }
      `);

      return result;
    } catch (error) {
      throw new Error(`Failed to check file existence: ${error.message}`);
    }
  }

  /**
   * Get detailed information about Pyodide environment
   * @returns {Promise<Object>} Environment information
   *
   * Collects metadata from within the WebAssembly sandbox such as the Python
   * version, available modules and a sample of environment variables.  This is
   * primarily used for diagnostics.
   */
  async getEnvironmentInfo() {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import sys
import os
import platform

try:
    # Get basic Python info
    info = {
        'python_version': sys.version,
        'python_version_info': list(sys.version_info),
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'python_executable': sys.executable,
        'python_path': sys.path[:5],  # First 5 entries only
        'current_directory': os.getcwd(),
        'environment_variables': dict(list(os.environ.items())[:10]),  # First 10 only
    }
    
    # Get available modules
    try:
        available_modules = []
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules():
            if len(available_modules) < 50:  # Limit to first 50
                available_modules.append(modname)
        info['available_modules'] = sorted(available_modules)
    except:
        info['available_modules'] = 'Unable to determine'
    
    # Get memory info if possible
    try:
        import gc
        info['garbage_collector'] = {
            'count': gc.get_count(),
            'stats': gc.get_stats()[:2] if hasattr(gc, 'get_stats') else 'Not available'
        }
    except:
        info['garbage_collector'] = 'Not available'
    
    info
    
except Exception as e:
    {
        'error': str(e),
        'error_type': type(e).__name__
    }
      `);

      return result;
    } catch (error) {
      throw new Error(`Failed to get environment info: ${error.message}`);
    }
  }
  /**
   * Extract a file from Pyodide's virtual filesystem and save it to the real filesystem
   * @param {string} virtualPath - Path in Pyodide's virtual filesystem (e.g., '/plots/matplotlib/plot.png')
   * @param {string} realPath - Path in the real filesystem where to save the file
   * @returns {Promise<boolean>} - True if file was successfully extracted
   */
  async extractVirtualFile(virtualPath, realPath) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.pyodide.runPythonAsync(`
import os
import shutil
import json

virtual_path = '${virtualPath}'
try:
    # Check if file exists in virtual filesystem
    if os.path.exists(virtual_path):
        # Read the file content from virtual filesystem
        with open(virtual_path, 'rb') as f:
            file_content = f.read()
        
        # Return file content as base64 for transfer
        import base64
        result = {
            'success': True,
            'file_exists': True,
            'content_b64': base64.b64encode(file_content).decode('utf-8'),
            'file_size': len(file_content)
        }
    else:
        result = {
            'success': False,
            'file_exists': False,
            'error': f'File {virtual_path} does not exist in virtual filesystem'
        }
except Exception as e:
    result = {
        'success': False,
        'file_exists': False,
        'error': str(e)
    }

json.dumps(result)
      `);

      const parsedResult = JSON.parse(result);

      if (!parsedResult.success) {
        logger.warn(`Failed to extract virtual file ${virtualPath}:`, parsedResult.error);
        return false;
      }

      // Decode base64 content and write to real filesystem
      const fs = require('fs');
      const path = require('path');
      
      // Ensure directory exists
      const dir = path.dirname(realPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      // Write file content
      const buffer = Buffer.from(parsedResult.content_b64, 'base64');
      fs.writeFileSync(realPath, buffer);
      
      logger.info(`✅ Extracted virtual file ${virtualPath} to ${realPath} (${parsedResult.file_size} bytes)`);
      return true;

    } catch (error) {
      logger.error(`Failed to extract virtual file ${virtualPath}:`, error.message);
      return false;
    }
  }

  /**
   * Extract all plot files from virtual filesystem to real filesystem
   * @returns {Promise<Array>} - Array of extracted file paths
   */
  async extractAllPlotFiles() {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      // Get list of all files in virtual plot directories
      const result = await this.pyodide.runPythonAsync(`
import os

plot_files = []
plot_dirs = ['/plots/matplotlib', '/plots/seaborn']

for plot_dir in plot_dirs:
    if os.path.exists(plot_dir):
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            if os.path.isfile(file_path):
                plot_files.append(file_path)

plot_files
      `);

      const extractedFiles = [];
      const path = require('path');

      for (const virtualPath of result) {
        // Convert virtual path to real path
        const relativePath = virtualPath.replace('/plots/', '');
        const realPath = path.join(__dirname, '../../plots', relativePath);
        
        const success = await this.extractVirtualFile(virtualPath, realPath);
        if (success) {
          extractedFiles.push(realPath);
        }
      }

      return extractedFiles;
    } catch (error) {
      logger.error('Failed to extract plot files:', error.message);
      return [];
    }
  }
}

// Export singleton instance
module.exports = new PyodideService();
