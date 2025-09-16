/**
 * Pyodide Service - Process Pool Implementation
 *
 * This service handles all Pyodide-related operations using a persistent
 * process pool that provides crash protection without the overhead of
 * initializing Pyodide for every execution.
 * 
 * Key Features:
 * - Persistent process pool for performance
 * - True process isolation prevents server crashes
 * - Automatic process recycling after errors
 * - Support for concurrent executions
 * - Comprehensive error handling and logging
 */

const PyodideProcessPool = require('./pyodide-process-pool');
const logger = require('../utils/logger');
const config = require('../config/index');

class PyodideService {
  constructor() {
    this.processPool = new PyodideProcessPool({
      poolSize: config.processPool.poolSize,
      maxExecutions: config.processPool.maxExecutions,
      processInitTimeout: config.processPool.processInitTimeout,
      executionTimeout: config.constants?.EXECUTION?.DEFAULT_TIMEOUT || 30000
    });
    this.defaultTimeout = config.constants?.EXECUTION?.DEFAULT_TIMEOUT || 30000;
    this.isReady = false;
    this.initializationPromise = null;
    
    // Initialize service
    this.initialize();
  }

  /**
   * Initialize the Pyodide service
   * @private
   * @returns {Promise<void>} Initialization promise
   */
  async initialize() {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._doInitialize();
    return this.initializationPromise;
  }

  /**
   * Perform actual initialization
   * @private
   */
  async _doInitialize() {
    try {
      logger.info('Initializing Pyodide service with process pool', {
        component: 'pyodide-service',
        action: 'initialize'
      });

      // Initialize the process pool
      await this.processPool.initialize();

      // Test that the pool is working
      const testResult = await this.processPool.executeCode(
        'print("Pyodide service initialized successfully")',
        {},
        20000 // 20 second timeout for initialization test
      );

      if (!testResult.success) {
        throw new Error(`Initialization test failed: ${testResult.error}`);
      }

      this.isReady = true;
      
      logger.info('Pyodide service initialized successfully', {
        component: 'pyodide-service',
        processPool: true,
        poolStats: this.processPool.getStats()
      });

    } catch (error) {
      logger.error('Failed to initialize Pyodide service', {
        component: 'pyodide-service',
        error: error.message
      });
      throw error;
    }
  }
  /**
   * Delete a file from the Pyodide virtual filesystem
   * @param {string} filename - Name of the file to delete
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<Object>} Deletion result
   */
  async deletePyodideFile(filename, timeout = this.defaultTimeout) {
    if (!this.isReady) {
      throw new Error('Pyodide service not ready');
    }

    if (!filename || typeof filename !== 'string') {
      throw new Error('Invalid filename provided');
    }

    logger.info('Deleting file from Pyodide filesystem', {
      component: 'pyodide-service',
      action: 'deletePyodideFile',
      filename,
      timeout
    });

    try {
      // Python code to delete file from virtual filesystem
      const deleteCode = `
from pathlib import Path
import os

filename = "${filename.replace(/"/g, '\\"')}"
upload_path = Path('/home/pyodide/uploads') / filename

success = False
message = ""

try:
    if upload_path.exists():
        upload_path.unlink()
        success = True
        message = f"File {filename} deleted successfully from virtual filesystem"
        print(f"DELETED: {filename}")
    else:
        success = False
        message = f"File {filename} not found in virtual filesystem"
        print(f"NOT_FOUND: {filename}")
except Exception as e:
    success = False
    message = f"Error deleting file {filename}: {str(e)}"
    print(f"ERROR: {str(e)}")

print(f"SUCCESS: {success}")
print(f"MESSAGE: {message}")
      `.trim();

      const result = await this.executeCode(deleteCode, {}, timeout);

      if (!result.success) {
        logger.error('Failed to delete file from Pyodide filesystem', {
          component: 'pyodide-service',
          filename,
          error: result.error
        });
        return {
          success: false,
          message: `Failed to delete file: ${result.error}`,
          error: result.error
        };
      }

      // Parse the output to determine success
      const output = result.stdout || '';
      const isDeleted = output.includes('DELETED:');
      const isNotFound = output.includes('NOT_FOUND:');
      const hasError = output.includes('ERROR:');

      if (isDeleted) {
        logger.info('File deleted from Pyodide filesystem', {
          component: 'pyodide-service',
          filename
        });
        return {
          success: true,
          message: `File ${filename} deleted successfully from virtual filesystem`
        };
      } else if (isNotFound) {
        logger.warn('File not found in Pyodide filesystem', {
          component: 'pyodide-service',
          filename
        });
        return {
          success: false,
          message: `File ${filename} not found in virtual filesystem`,
          notFound: true
        };
      } else if (hasError) {
        const errorMatch = output.match(/ERROR: (.+)/);
        const errorMsg = errorMatch ? errorMatch[1] : 'Unknown error';
        logger.error('Error deleting file from Pyodide filesystem', {
          component: 'pyodide-service',
          filename,
          error: errorMsg
        });
        return {
          success: false,
          message: `Error deleting file: ${errorMsg}`,
          error: errorMsg
        };
      } else {
        logger.warn('Unexpected output from file deletion', {
          component: 'pyodide-service',
          filename,
          output
        });
        return {
          success: false,
          message: 'Unexpected response from file deletion operation'
        };
      }

    } catch (error) {
      logger.error('Exception during file deletion', {
        component: 'pyodide-service',
        filename,
        error: error.message
      });
      return {
        success: false,
        message: `Exception during file deletion: ${error.message}`,
        error: error.message
      };
    }
  }

  /**
   * Execute Python code with full isolation and timeout protection
   * @param {string} code - Python code to execute
   * @param {Object} context - Variables to make available in Python scope
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<Object>} Execution result
   */
  async executeCode(code, context = {}, timeout = this.defaultTimeout) {
    // Ensure service is initialized
    await this.initialize();

    if (!this.isReady) {
      throw new Error('Pyodide service not ready');
    }

    // Validate inputs
    if (typeof code !== 'string') {
      throw new Error('Code must be a string');
    }

    if (code.trim().length === 0) {
      throw new Error('Code cannot be empty');
    }

    if (timeout < 1000 || timeout > 300000) { // 1 second to 5 minutes
      throw new Error('Timeout must be between 1000ms and 300000ms');
    }

    // Security check for dangerous patterns
    const dangerousPatterns = [
      /\bos\.system\b/i,
      /\bsubprocess\b/i,
      /\b__import__\s*\(\s*['"]os['"]\s*\)/i,
      /\beval\s*\(/i,
      /\bexec\s*\(/i,
      /\bopen\s*\(/i,
      /\bfile\s*\(/i
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(code)) {
        logger.warn('Potentially dangerous code detected', {
          component: 'pyodide-service',
          action: 'executeCode',
          pattern: pattern.toString(),
          codePreview: code.substring(0, 100)
        });
        // Note: We allow it to proceed since we have process isolation
      }
    }

    const startTime = Date.now();

    try {
      logger.debug('Executing Python code in isolated process', {
        component: 'pyodide-service',
        action: 'executeCode',
        codeLength: code.length,
        timeout,
        contextKeys: Object.keys(context)
      });

      const result = await this.processPool.executeCode(code, context, timeout);
      
      logger.info('Python code execution completed', {
        component: 'pyodide-service',
        action: 'executeCode',
        success: result.success,
        executionTime: result.executionTime,
        codeLength: code.length,
        hasOutput: !!(result.stdout || result.stderr)
      });

      return result;

    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      logger.error('Python code execution failed', {
        component: 'pyodide-service',
        action: 'executeCode',
        error: error.message,
        executionTime,
        codeLength: code.length
      });

      return {
        success: false,
        error: error.message,
        executionTime,
        metadata: {
          timestamp: new Date().toISOString(),
          processIsolated: true
        }
      };
    }
  }

  /**
   * Install Python package using micropip
   * @param {string} packageName - Name of package to install
   * @param {number} timeout - Installation timeout in milliseconds
   * @returns {Promise<Object>} Installation result
   */
  async installPackage(packageName, timeout = 60000) {
    // Ensure service is initialized
    await this.initialize();

    if (!this.isReady) {
      throw new Error('Pyodide service not ready');
    }

    // Validate package name
    if (typeof packageName !== 'string' || packageName.trim().length === 0) {
      throw new Error('Package name must be a non-empty string');
    }

    // Basic package name validation
    if (!/^[a-zA-Z0-9_\-.]+$/.test(packageName)) {
      throw new Error('Invalid package name format');
    }

    const startTime = Date.now();

    try {
      logger.info('Installing Python package in isolated process', {
        component: 'pyodide-service',
        action: 'installPackage',
        package: packageName,
        timeout
      });

      const result = await this.processPool.installPackage(packageName, timeout);

      logger.info('Package installation completed', {
        component: 'pyodide-service',
        action: 'installPackage',
        package: packageName,
        success: result.success,
        installTime: result.installTime
      });

      return result;

    } catch (error) {
      const installTime = Date.now() - startTime;

      logger.error('Package installation failed', {
        component: 'pyodide-service',
        action: 'installPackage',
        package: packageName,
        error: error.message,
        installTime
      });

      return {
        success: false,
        package: packageName,
        error: error.message,
        installTime,
        metadata: {
          timestamp: new Date().toISOString(),
          processIsolated: true
        }
      };
    }
  }

  /**
   * Get service status and statistics
   * @returns {Object} Service status information
   */
  getStatus() {
    const poolStats = this.processPool.getStats();
    
    return {
      ready: this.isReady,
      processPool: true,
      poolStats: poolStats,
      defaultTimeout: this.defaultTimeout,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Check if service is ready for use
   * @returns {boolean} Whether service is ready
   */
  isServiceReady() {
    return this.isReady;
  }

  /**
   * Reset the Pyodide environment by clearing user-defined variables
   * 
   * This method clears all user-defined variables from the Python environment
   * while preserving system modules and built-in functions. It uses garbage
   * collection to free up memory after variable deletion.
   * 
   * @returns {Promise<Object>} Reset result following API contract
   *   - success: boolean indicating if reset completed successfully
   *   - data: object containing reset details (stdout, stderr, result)
   *   - error: string with error message if failed, null if successful
   *   - meta: object with timestamp
   * 
   * @example
   * const result = await pyodideService.reset();
   * // Returns: {
   * //   success: true,
   * //   data: { 
   * //     result: "Environment reset complete",
   * //     stdout: "Cleared 5 user variables...",
   * //     stderr: null
   * //   },
   * //   error: null,
   * //   meta: { timestamp: "2025-09-16T00:15:00.000Z" }
   * // }
   */
  async reset() {
    await this.initialize();
    
    if (!this.isReady) {
      throw new Error('Pyodide service not ready');
    }

    try {
      logger.info('Resetting Pyodide environment', {
        component: 'pyodide-service',
        action: 'reset'
      });

      // Use robust Python code for resetting environment
      const resetCode = `import gc
import sys

# Initialize counters
cleared_count = 0
errors = []

try:
    # Get current globals before we start clearing
    current_globals = list(globals().keys())

    # Define essential variables to preserve (modules and built-ins)
    preserved_vars = {
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
        '__annotations__', '__builtins__',
        # Standard library modules
        'sys', 'os', 'json', 'io', 'gc', 'time', 'math', 'random',
        'pathlib', 'Path',
        # Core data science packages
        'numpy', 'np', 'pandas', 'pd', 'matplotlib', 'plt', 
        'seaborn', 'sns', 'sklearn', 'scipy',
        # Pyodide specific
        'pyodide', 'micropip', 'js',
        # Internal variables for this reset operation
        'cleared_count', 'errors', 'current_globals', 'preserved_vars'
    }

    # Clear user-defined variables with robust error handling
    for var_name in current_globals:
        if not var_name.startswith('_') and var_name not in preserved_vars:
            try:
                # First check if variable still exists
                if var_name in globals():
                    del globals()[var_name]
                    cleared_count += 1
            except (KeyError, AttributeError, NameError):
                # Variable already deleted or doesn't exist
                pass
            except Exception as e:
                # Log other errors but continue
                errors.append(f"Could not delete {var_name}: {str(e)}")

    # Force garbage collection
    gc.collect()

    # Report results
    result_message = f"Environment reset complete. Cleared {cleared_count} user variables."
    if errors:
        result_message += f" {len(errors)} variables could not be cleared."

    print(result_message)
    
except Exception as e:
    print(f"Reset operation failed: {str(e)}")
    # Even if there were issues, try garbage collection
    try:
        gc.collect()
        print("Garbage collection completed despite errors.")
    except:
        pass`;

      const result = await this.executeCode(resetCode, {}, 10000);
      
      if (!result.success) {
        logger.error('Failed to reset Pyodide environment', {
          component: 'pyodide-service',
          action: 'reset',
          error: result.error
        });
        
        return {
          success: false,
          data: null,
          error: `Reset failed: ${result.error}`,
          meta: { timestamp: new Date().toISOString() }
        };
      }

      logger.info('Pyodide environment reset successfully', {
        component: 'pyodide-service',
        action: 'reset',
        output: result.stdout
      });

        return {
          success: true,
          data: {
            result: 'Pyodide environment reset successfully',
            stdout: result.stdout || '',
            stderr: result.stderr || null
          },
          error: null,
          meta: { timestamp: new Date().toISOString() }
        };    } catch (error) {
      logger.error('Failed to reset Pyodide environment', {
        component: 'pyodide-service',
        action: 'reset',
        error: error.message
      });
      
      return {
        success: false,
        data: null,
        error: `Reset failed: ${error.message}`,
        meta: { timestamp: new Date().toISOString() }
      };
    }
  }

  /**
   * Shutdown the service and cleanup resources
   */
  async shutdown() {
    logger.info('Shutting down Pyodide service', {
      component: 'pyodide-service',
      action: 'shutdown'
    });

    try {
      await this.processPool.shutdown();
      this.isReady = false;
      
      logger.info('Pyodide service shutdown completed', {
        component: 'pyodide-service'
      });
    } catch (error) {
      logger.error('Error during Pyodide service shutdown', {
        component: 'pyodide-service',
        error: error.message
      });
    }
  }

  /**
   * Health check for the service
   * @returns {Promise<Object>} Health check result
   */
  async healthCheck() {
    try {
      if (!this.isReady) {
        return {
          healthy: false,
          error: 'Service not ready',
          timestamp: new Date().toISOString()
        };
      }

      // Quick execution test
      const startTime = Date.now();
      const result = await this.executeCode('print("health check")', {}, 5000);
      const responseTime = Date.now() - startTime;

      return {
        healthy: result.success,
        responseTime,
        poolStats: this.processPool.getStats(),
        error: result.success ? undefined : result.error,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return {
        healthy: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }
}

// Create singleton instance
const pyodideService = new PyodideService();

module.exports = pyodideService;