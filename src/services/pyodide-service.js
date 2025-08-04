/**
 * Pyodide Service - Complete Implementation
 * 
 * This service handles all Pyodide-related operations including:
 * - Python code execution
 * - Package management
 * - File operations
 * - Environment management
 */

const logger = require('../utils/logger');

class PyodideService {
  constructor() {
    this.pyodide = null;
    this.isReady = false;
    this.initializationPromise = null;
    this.executionTimeout = 30000; // 30 seconds default timeout
  }

  /**
   * Initialize Pyodide with common packages
   * @returns {Promise<boolean>} True if initialization successful
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

print("🎉 Pyodide environment ready!")
      `);

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

      // Start output capture
      await this.pyodide.runPythonAsync('output_capture.start_capture()');

      let result;
      let executionError = null;

      try {
        // Execute the code
        if (code.includes('await ') || code.includes('micropip.install')) {
          result = await this.pyodide.runPythonAsync(code);
        } else {
          result = this.pyodide.runPython(code);
        }
      } catch (error) {
        executionError = error;
      }

      // Stop capture and get output
      const output = await this.pyodide.runPythonAsync('output_capture.stop_capture()');

      // Clean up
      await this.pyodide.runPythonAsync('output_capture.reset()');

      if (executionError) {
        return {
          success: false,
          error: executionError.message,
          stdout: output.get('stdout') || '',
          stderr: output.get('stderr') || '',
          timestamp: new Date().toISOString()
        };
      }

      // Convert result to JSON-safe format
      let jsonSafeResult = null;
      if (result !== undefined && result !== null) {
        try {
          jsonSafeResult = await this.pyodide.runPythonAsync(`make_json_safe(${JSON.stringify(result)})`);
        } catch (conversionError) {
          jsonSafeResult = String(result);
        }
      }

      return {
        success: true,
        result: jsonSafeResult,
        stdout: output.get('stdout') || '',
        stderr: output.get('stderr') || '',
        timestamp: new Date().toISOString()
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
      
      const result = await this.executeCode(`
# Check if micropip is available
try:
    import micropip
    await micropip.install("${packageName}")
    f"Successfully installed ${packageName}"
except ImportError:
    "Micropip not available - cannot install packages"
except Exception as e:
    f"Installation failed: {str(e)}"
      `);

      return result;

    } catch (error) {
      throw new Error(`Failed to install package ${packageName}: ${error.message}`);
    }
  }

  /**
   * Get list of installed packages
   * @returns {Promise<Object>} List of packages
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
   */
  async listPyodideFiles() {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import os
try:
    # List files in current directory
    files = []
    for item in os.listdir('.'):
        if os.path.isfile(item):
            stat_info = os.stat(item)
            files.append({
                'name': item,
                'size': stat_info.st_size,
                'modified': stat_info.st_mtime
            })
    
    {
        'success': True,
        'files': files,
        'count': len(files)
    }
except Exception as e:
    {
        'success': False,
        'error': str(e)
    }
      `);

      return result;
    } catch (error) {
      throw new Error(`Failed to list Pyodide files: ${error.message}`);
    }
  }

  /**
   * Delete a file from Pyodide's virtual filesystem
   * @param {string} filename - Name of file to delete
   * @returns {Promise<Object>} Deletion result
   */
  async deletePyodideFile(filename) {
    if (!this.isReady) {
      throw new Error('Pyodide is not ready');
    }

    try {
      const result = await this.executeCode(`
import os
try:
    if os.path.exists('${filename}'):
        os.remove('${filename}')
        f"File ${filename} deleted successfully"
    else:
        f"File ${filename} not found"
except Exception as e:
    f"Error deleting ${filename}: {str(e)}"
      `);

      return result;
    } catch (error) {
      throw new Error(`Failed to delete Pyodide file: ${error.message}`);
    }
  }

  /**
   * Get current status of the Pyodide service
   * @returns {Object} Service status
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
}

// Export singleton instance
module.exports = new PyodideService();