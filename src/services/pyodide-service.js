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
upload_path = Path('/uploads') / filename

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
   * Get installed packages information
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<Object>} Package information including installed packages and loaded modules
   */
  async getInstalledPackages(timeout = this.defaultTimeout) {
    const startTime = Date.now();
    
    try {
      logger.info('Getting installed packages information', {
        component: 'pyodide-service',
        action: 'getInstalledPackages',
        timeout
      });

      // Python code to get package information
      const packageInfoCode = `
import sys
import json
import micropip
from pathlib import Path

# Get Python version
python_version = sys.version

# Get all loaded modules
loaded_modules = list(sys.modules.keys())

# Get installed packages via micropip
try:
    # Get packages that micropip knows about
    installed_packages = []
    
    # Try to get the package list from micropip
    import importlib.metadata
    distributions = list(importlib.metadata.distributions())
    installed_packages = [dist.metadata['Name'] for dist in distributions]
    
    # Also include basic packages that might not be in distributions
    basic_packages = ['micropip', 'pyodide-js', 'pyodide']
    for pkg in basic_packages:
        if pkg not in installed_packages and pkg in sys.modules:
            installed_packages.append(pkg)
    
except Exception as e:
    # Fallback: use what we can determine from sys.modules
    installed_packages = [name for name in sys.modules.keys() 
                         if not name.startswith('_') and '.' not in name]

# Create result dictionary
result = {
    "python_version": python_version,
    "installed_packages": sorted(list(set(installed_packages))),
    "total_packages": len(set(installed_packages)),
    "loaded_modules": sorted([name for name in loaded_modules 
                            if not name.startswith('_')])
}

print(json.dumps(result, indent=2))
result
`;

      // Execute the package info code
      const response = await this.processPool.executeCode(packageInfoCode, {}, timeout);
      const executionTime = Date.now() - startTime;
      
      if (!response.success) {
        throw new Error(response.error || 'Failed to get package information');
      }

      // Parse the result - it should be a JSON string in stdout or the result itself
      let packageData;
      try {
        if (response.result && typeof response.result === 'object') {
          packageData = response.result;
        } else {
          // Try to parse from stdout
          const jsonMatch = response.stdout.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            packageData = JSON.parse(jsonMatch[0]);
          } else {
            throw new Error('Could not parse package information from response');
          }
        }
      } catch (parseError) {
        logger.warn('Failed to parse package info, using fallback', {
          parseError: parseError.message,
          stdout: response.stdout?.substring(0, 200)
        });
        
        // Fallback response structure
        packageData = {
          python_version: "Unknown",
          installed_packages: ["micropip", "pyodide-js"],
          total_packages: 2,
          loaded_modules: ["sys", "os", "json"]
        };
      }

      logger.info('Package information retrieved successfully', {
        component: 'pyodide-service',
        action: 'getInstalledPackages',
        packageCount: packageData.total_packages,
        executionTime
      });

      return packageData;

    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      logger.error('Failed to get package information', {
        component: 'pyodide-service',
        action: 'getInstalledPackages',
        error: error.message,
        executionTime
      });

      throw new Error(`Failed to get package information: ${error.message}`);
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