/**
 * Pyodide Process Pool Manager
 * 
 * Manages a pool of long-running Pyodide child processes to avoid
 * the overhead of initializing Pyodide for every execution.
 * 
 * Features:
 * - Persistent child processes that stay alive
 * - Process pool for handling concurrent requests
 * - Automatic process recycling after errors
 * - True isolation with killable processes for infinite loops
 */

const { spawn } = require('child_process');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

class PyodideProcessPool {
  constructor(options = {}) {
    this.poolSize = options.poolSize || 2; // Number of persistent processes
    this.maxExecutions = options.maxExecutions || 100; // Recycle after N executions
    this.processInitTimeout = options.processInitTimeout || 90000; // 90 seconds for Pyodide init
    this.executionTimeout = options.executionTimeout || 30000; // 30 seconds default execution timeout
    
    this.processes = []; // Array of process info objects
    this.availableProcesses = []; // Queue of available process IDs
    this.pendingExecutions = new Map(); // executionId -> Promise resolve/reject
    this.processStats = new Map(); // processId -> stats
    
    this.isInitialized = false;
    this.initializationPromise = null;
  }

  /**
   * Initialize the process pool
   * @returns {Promise<void>} Promise that resolves when pool is initialized
   */
  async initialize() {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this._initializePool();
    return this.initializationPromise;
  }

  /**
   * Initialize all processes in the pool
   * @private
   */
  async _initializePool() {
    logger.info('Initializing Pyodide process pool', {
      component: 'pyodide-process-pool',
      poolSize: this.poolSize,
      initTimeout: this.processInitTimeout
    });

    const initPromises = [];
    
    for (let i = 0; i < this.poolSize; i++) {
      initPromises.push(this._createProcess(i));
    }

    try {
      await Promise.all(initPromises);
      this.isInitialized = true;
      
      logger.info('Pyodide process pool initialized successfully', {
        component: 'pyodide-process-pool',
        availableProcesses: this.availableProcesses.length
      });
    } catch (error) {
      logger.error('Failed to initialize Pyodide process pool', {
        component: 'pyodide-process-pool',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Create a new persistent Pyodide process
   * @param {number} processId - The ID for the new process
   * @returns {Promise<Object>} Promise that resolves to process info
   * @private
   */
  async _createProcess(processId) {
    return new Promise((resolve, reject) => {
      const executorPath = path.join(__dirname, 'pyodide-executor.js');
      
      const childProcess = spawn('node', [executorPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: false,
        windowsHide: true
      });

      const processInfo = {
        id: processId,
        process: childProcess,
        pid: childProcess.pid,
        available: false,
        executionCount: 0,
        createdAt: Date.now(),
        lastUsed: Date.now(),
        outputBuffer: ''
      };

      this.processes[processId] = processInfo;
      this.processStats.set(processId, {
        totalExecutions: 0,
        successfulExecutions: 0,
        failedExecutions: 0,
        averageExecutionTime: 0
      });

      // Set up initialization timeout
      const initTimeout = setTimeout(() => {
        this._killProcess(processId, 'initialization timeout');
        reject(new Error(`Process ${processId} initialization timeout after ${this.processInitTimeout}ms`));
      }, this.processInitTimeout);

      // Handle process output
      childProcess.stdout.on('data', (data) => {
        processInfo.outputBuffer += data.toString();
        this._processMessages(processId);
      });

      childProcess.stderr.on('data', (data) => {
        logger.error('Process stderr output', {
          component: 'pyodide-process-pool',
          processId,
          stderr: data.toString().substring(0, 500)
        });
      });

      childProcess.on('error', (error) => {
        logger.error('Process error', {
          component: 'pyodide-process-pool',
          processId,
          error: error.message
        });
        clearTimeout(initTimeout);
        this._handleProcessDeath(processId);
        reject(error);
      });

      childProcess.on('exit', (code, signal) => {
        logger.warn('Process exited unexpectedly', {
          component: 'pyodide-process-pool',
          processId,
          code,
          signal,
          uptime: Date.now() - processInfo.createdAt
        });
        clearTimeout(initTimeout);
        this._handleProcessDeath(processId);
      });

      // Store the resolve function for when initialization completes
      processInfo.initResolve = () => {
        clearTimeout(initTimeout);
        processInfo.available = true;
        this.availableProcesses.push(processId);
        resolve(processInfo);
      };

      logger.debug('Created Pyodide process', {
        component: 'pyodide-process-pool',
        processId,
        pid: childProcess.pid
      });
    });
  }

  /**
   * Process messages from child processes
   * @param {number} processId - Process ID
   * @private
   */
  _processMessages(processId) {
    const processInfo = this.processes[processId];
    if (!processInfo) return;

    // Process complete JSON lines
    const lines = processInfo.outputBuffer.split('\n');
    processInfo.outputBuffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line.trim());
          this._handleMessage(processId, message);
        } catch (error) {
          logger.warn('Failed to parse process message', {
            component: 'pyodide-process-pool',
            processId,
            line: line.substring(0, 100),
            error: error.message
          });
        }
      }
    }
  }

  /**
   * Handle messages from child processes
   * @param {number} processId - The ID of the process that sent the message
   * @param {Object} message - The message received from the child process
   * @private
   */
  _handleMessage(processId, message) {
    const processInfo = this.processes[processId];
    if (!processInfo) return;

    switch (message.type) {
      case 'initialized':
        if (message.success) {
          logger.info('Process initialized successfully', {
            component: 'pyodide-process-pool',
            processId
          });
          if (processInfo.initResolve) {
            processInfo.initResolve();
            delete processInfo.initResolve;
          }
        } else {
          logger.error('Process initialization failed', {
            component: 'pyodide-process-pool',
            processId,
            error: message.error
          });
          this._killProcess(processId, 'initialization failed');
        }
        break;

      case 'result':
      case 'package_result': {
        const executionId = message.executionId;
        const pendingExecution = this.pendingExecutions.get(executionId);
        
        if (pendingExecution) {
          this.pendingExecutions.delete(executionId);
          
          // Update process stats
          const stats = this.processStats.get(processId);
          if (stats) {
            stats.totalExecutions++;
            if (message.success) {
              stats.successfulExecutions++;
            } else {
              stats.failedExecutions++;
            }
          }

          // Return process to available pool
          processInfo.available = true;
          processInfo.lastUsed = Date.now();
          processInfo.executionCount++;
          this.availableProcesses.push(processId);

          // Check if process needs recycling
          if (processInfo.executionCount >= this.maxExecutions) {
            this._recycleProcess(processId);
          }

          // Resolve the pending execution
          if (message.success) {
            pendingExecution.resolve(message);
          } else {
            pendingExecution.reject(new Error(message.error || 'Execution failed'));
          }
        }
        break;
      }

      case 'error': {
        const errorExecutionId = message.executionId;
        const errorPendingExecution = this.pendingExecutions.get(errorExecutionId);
        
        if (errorPendingExecution) {
          this.pendingExecutions.delete(errorExecutionId);
          processInfo.available = true;
          this.availableProcesses.push(processId);
          errorPendingExecution.reject(new Error(message.error || 'Process error'));
        }
        break;
      }

      default:
        logger.debug('Unknown message type from process', {
          component: 'pyodide-process-pool',
          processId,
          messageType: message.type
        });
    }
  }

  /**
   * Execute Python code using an available process
   * @param {string} code - Python code to execute
   * @param {Object} context - Execution context
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<Object>} Execution result
   */
  async executeCode(code, context = {}, timeout = this.executionTimeout) {
    if (!this.isInitialized) {
      throw new Error('Process pool not initialized');
    }

    // Wait for an available process
    const processId = await this._getAvailableProcess();
    const executionId = uuidv4();

    return new Promise((resolve, reject) => {
      // Store the pending execution
      this.pendingExecutions.set(executionId, { resolve, reject });

      // Set execution timeout
      const timeoutHandle = setTimeout(() => {
        this.pendingExecutions.delete(executionId);
        this._killProcess(processId, 'execution timeout');
        reject(new Error(`Execution timeout after ${timeout}ms`));
      }, timeout);

      // Send execution message to process
      const message = {
        type: 'execute',
        code,
        context,
        executionId
      };

      try {
        const processInfo = this.processes[processId];
        processInfo.process.stdin.write(JSON.stringify(message) + '\n');
        
        // Override resolve to clear timeout
        const originalResolve = resolve;
        const originalReject = reject;
        
        this.pendingExecutions.set(executionId, {
          resolve: (result) => {
            clearTimeout(timeoutHandle);
            originalResolve(result);
          },
          reject: (error) => {
            clearTimeout(timeoutHandle);
            originalReject(error);
          }
        });
        
      } catch (error) {
        clearTimeout(timeoutHandle);
        this.pendingExecutions.delete(executionId);
        // Return process to pool
        this.availableProcesses.push(processId);
        reject(new Error(`Failed to send execution message: ${error.message}`));
      }
    });
  }

  /**
   * Install a Python package using an available process
   * @param {string} packageName - Name of the Python package to install
   * @param {number} timeout - Timeout in milliseconds for the installation
   */
  async installPackage(packageName, timeout = 60000) {
    if (!this.isInitialized) {
      throw new Error('Process pool not initialized');
    }

    const processId = await this._getAvailableProcess();
    const executionId = uuidv4();

    return new Promise((resolve, reject) => {
      this.pendingExecutions.set(executionId, { resolve, reject });

      const timeoutHandle = setTimeout(() => {
        this.pendingExecutions.delete(executionId);
        this._killProcess(processId, 'package installation timeout');
        reject(new Error(`Package installation timeout after ${timeout}ms`));
      }, timeout);

      const message = {
        type: 'install_package',
        package: packageName,
        executionId
      };

      try {
        const processInfo = this.processes[processId];
        processInfo.process.stdin.write(JSON.stringify(message) + '\n');
        
        const originalResolve = resolve;
        const originalReject = reject;
        
        this.pendingExecutions.set(executionId, {
          resolve: (result) => {
            clearTimeout(timeoutHandle);
            originalResolve(result);
          },
          reject: (error) => {
            clearTimeout(timeoutHandle);
            originalReject(error);
          }
        });
        
      } catch (error) {
        clearTimeout(timeoutHandle);
        this.pendingExecutions.delete(executionId);
        this.availableProcesses.push(processId);
        reject(new Error(`Failed to send package installation message: ${error.message}`));
      }
    });
  }

  /**
   * Get an available process, waiting if necessary
   * @private
   * @returns {Promise<string>} Process ID of an available process
   */
  async _getAvailableProcess() {
    return new Promise((resolve) => {
      const checkAvailability = () => {
        if (this.availableProcesses.length > 0) {
          const processId = this.availableProcesses.shift();
          const processInfo = this.processes[processId];
          if (processInfo && processInfo.available) {
            processInfo.available = false;
            resolve(processId);
          } else {
            // Process was killed, try again
            setTimeout(checkAvailability, 0);
          }
        } else {
          // No processes available, check again in 100ms
          setTimeout(checkAvailability, 100);
        }
      };
      
      checkAvailability();
    });
  }

  /**
   * Kill a process and optionally replace it
   * @param {string} processId - ID of the process to kill
   * @param {string} reason - Reason for killing the process
   * @private
   */
  _killProcess(processId, reason) {
    const processInfo = this.processes[processId];
    if (!processInfo) return;

    logger.warn('Killing process', {
      component: 'pyodide-process-pool',
      processId,
      reason,
      uptime: Date.now() - processInfo.createdAt
    });

    try {
      processInfo.process.kill('SIGTERM');
      setTimeout(() => {
        if (!processInfo.process.killed) {
          processInfo.process.kill('SIGKILL');
        }
      }, 2000);
    } catch (error) {
      logger.error('Failed to kill process', {
        component: 'pyodide-process-pool',
        processId,
        error: error.message
      });
    }

    this._handleProcessDeath(processId);
  }

  /**
   * Handle process death and replacement
   * @param {string} processId - ID of the process that died
   * @private
   */
  _handleProcessDeath(processId) {
    // Remove from available processes
    const index = this.availableProcesses.indexOf(processId);
    if (index > -1) {
      this.availableProcesses.splice(index, 1);
    }

    // Clear the process info
    this.processes[processId] = null;

    // Create a replacement process
    if (this.isInitialized) {
      logger.info('Creating replacement process', {
        component: 'pyodide-process-pool',
        processId
      });
      
      this._createProcess(processId).catch(error => {
        logger.error('Failed to create replacement process', {
          component: 'pyodide-process-pool',
          processId,
          error: error.message
        });
      });
    }
  }

  /**
   * Recycle a process after it reaches max executions
   * @param {string} processId - ID of the process to recycle
   * @private
   */
  _recycleProcess(processId) {
    logger.info('Recycling process after max executions', {
      component: 'pyodide-process-pool',
      processId,
      executionCount: this.processes[processId].executionCount
    });
    
    this._killProcess(processId, 'recycling after max executions');
  }

  /**
   * Get pool statistics
   * @returns {Object} Pool statistics including active processes and execution counts
   */
  getStats() {
    const activeProcesses = this.processes.filter(p => p !== null).length;
    const totalExecutions = Array.from(this.processStats.values())
      .reduce((sum, stats) => sum + stats.totalExecutions, 0);

    return {
      poolSize: this.poolSize,
      activeProcesses,
      availableProcesses: this.availableProcesses.length,
      totalExecutions,
      pendingExecutions: this.pendingExecutions.size,
      processStats: Object.fromEntries(this.processStats)
    };
  }

  /**
   * Shutdown the entire pool
   */
  async shutdown() {
    logger.info('Shutting down Pyodide process pool', {
      component: 'pyodide-process-pool',
      activeProcesses: this.processes.filter(p => p !== null).length
    });

    this.isInitialized = false;

    for (let i = 0; i < this.processes.length; i++) {
      if (this.processes[i]) {
        this._killProcess(i, 'pool shutdown');
      }
    }

    // Wait for processes to exit
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    this.processes = [];
    this.availableProcesses = [];
    this.pendingExecutions.clear();
    this.processStats.clear();
  }
}

module.exports = PyodideProcessPool;