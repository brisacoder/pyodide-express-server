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
const os = require('os');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

class PyodideProcessPool {
  constructor(options = {}) {
    // Requested pool size (pre-adaptation)
    this.requestedPoolSize = options.poolSize || parseInt(process.env.PYODIDE_POOL_SIZE || '2', 10);

    // Establish adaptive memory constraints before finalizing poolSize
    const adaptive = this._computeAdaptivePoolSize(this.requestedPoolSize);
    this.poolSize = adaptive.finalPoolSize;
    this.adaptiveSizing = adaptive; // store details for stats/logging

    this.maxExecutions = options.maxExecutions || 20; // Recycle after N executions - reduced from 100 to prevent memory buildup
    this.processInitTimeout = options.processInitTimeout || 90000; // 90 seconds for Pyodide init
    this.executionTimeout = options.executionTimeout || 30000; // 30 seconds default execution timeout
    
    this.processes = []; // Array of process info objects
    this.availableProcesses = []; // Queue of available process IDs
    this.pendingExecutions = new Map(); // executionId -> Promise resolve/reject
    this.processStats = new Map(); // processId -> stats
    
    this.isInitialized = false;
    this.initializationPromise = null;
    this.memoryMonitorInterval = null;
    // Adaptive recycling & health tracking additions
    this.adaptiveEnabled = process.env.ADAPTIVE_CLEANUP_ENABLED === '1' || process.env.ADAPTIVE_CLEANUP_ENABLED === 'true';
    this.metricsHistoryLimit = 50; // retain last N metrics per process
    this.initFailureStreak = 0; // consecutive init failures (pool level)
    this.maxInitFailureStreak = parseInt(process.env.MAX_INIT_FAILURE_STREAK || '5', 10); // circuit breaker threshold
    this.circuitBreakerOpen = false;
    this.circuitBreakerResetMs = parseInt(process.env.CIRCUIT_BREAKER_RESET_MS || '60000', 10);
    this.lastCircuitOpenAt = 0;
    this.adaptiveRecycleReason = 'none';
    this.maxQueueSize = parseInt(process.env.MAX_PENDING_EXECUTIONS || '100', 10); // backpressure limit
    this.backpressureRejects = 0;
    // Performance protection additions
    this.minFreeMemFraction = Math.min(0.8, Math.max(0.01, parseFloat(process.env.MIN_FREE_MEM_FRACTION || '0.12')));
    this.minFreeMemFloorMB = Math.max(32, parseInt(process.env.MIN_FREE_MEM_MB_FLOOR || '256', 10));
    this.adaptiveQueueEnabled = process.env.ADAPTIVE_QUEUE_ENABLED !== '0';
    this.softTimeoutFactor = Math.min(0.95, Math.max(0.1, parseFloat(process.env.EXEC_SOFT_TIMEOUT_FACTOR || '0.7')));
    this.memGrowthRecycleMB = Math.max(10, parseInt(process.env.MEM_GROWTH_RECYCLE_MB || '120', 10));
    this.maxExecutions = parseInt(process.env.PYODIDE_MAX_EXECUTIONS_BEFORE_RECYCLE || `${this.maxExecutions}`, 10);
    this.lowMemRejects = 0;
    this.admissionRejects = 0;
    this.earlyTerminations = 0;
    this.recycledForGrowth = 0;
    this.recycledForRssLimit = 0;
    this.observedProcessRssMB = [];
    this.memoryObservationLimit = Math.max(1, parseInt(process.env.PYODIDE_MEMORY_OBSERVATION_LIMIT || '24', 10));
    this.maxWorkerRssMB = Math.max(128, parseInt(process.env.PYODIDE_MAX_WORKER_RSS_MB || '768', 10));
    this.memoryGuardEnabled = process.env.PYODIDE_MEMORY_GUARD !== '0';
    this.queueHighWaterMark = 0;
    const headroomCap = parseInt(process.env.PYODIDE_MIN_FREE_MEM_MB_CAP || '2048', 10);
    this.minFreeMemCapMB = Number.isFinite(headroomCap) && headroomCap > 0 ? headroomCap : Infinity;
    this.memoryGuardSoftBypass = process.env.PYODIDE_MEMORY_GUARD_SOFT_BYPASS !== '0';
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
      requestedPoolSize: this.requestedPoolSize,
      adaptiveReason: this.adaptiveSizing.reason,
      totalMemMB: this.adaptiveSizing.totalMemMB,
      perProcessMB: this.adaptiveSizing.perProcessMB,
      maxFraction: this.adaptiveSizing.maxPoolMemFraction,
      memCapCount: this.adaptiveSizing.memCapCount,
      cpuCount: this.adaptiveSizing.cpuCount,
      initTimeout: this.processInitTimeout
    });

    const createdProcesses = [];
    try {
      const targetPoolSize = this.poolSize;
      for (let i = 0; i < targetPoolSize; i++) {
        if (!this._canSpawnAnotherProcess(createdProcesses.length)) {
          this.adaptiveRecycleReason = 'memory_guard_startup';
          break;
        }
        const processInfo = await this._createProcess(i);
        createdProcesses.push(processInfo);
      }

      this.poolSize = createdProcesses.length;
      this.adaptiveSizing.finalPoolSize = createdProcesses.length;

      if (createdProcesses.length === 0) {
        throw new Error('No Pyodide workers could be started due to memory constraints');
      }

      this.isInitialized = true;
      this.startMemoryMonitoring();

      logger.info('Pyodide process pool initialized successfully', {
        component: 'pyodide-process-pool',
        availableProcesses: this.availableProcesses.length,
        poolSize: this.poolSize,
        observedProcessRssMB: this.observedProcessRssMB,
        maxWorkerRssMB: this.maxWorkerRssMB
      });
    } catch (error) {
      logger.error('Failed to initialize Pyodide process pool', {
        component: 'pyodide-process-pool',
        error: error.message
      });
      throw error;
    }
  }
  // ================= Performance Protection Helpers =================
  _memoryPressure() {
    const total = os.totalmem();
    const free = os.freemem();
    let available = free;

    if (process.platform === 'linux' && typeof fs.existsSync === 'function' && fs.existsSync('/proc/meminfo')) {
      try {
        const meminfo = fs.readFileSync('/proc/meminfo', 'utf8');
        const match = meminfo.match(/^MemAvailable:\s+(\d+)\s+kB/m);
        if (match) {
          available = parseInt(match[1], 10) * 1024;
        }
      } catch {
        // Ignore failures to read /proc/meminfo
      }
    }

    // On some platforms free memory can be zero while buffers/caches are reclaimable.
    // Fallback to free memory if the computed value is clearly wrong.
    if (!Number.isFinite(available) || available <= 0) {
      available = free;
    }

    return {
      total,
      free,
      available,
      totalMB: total / 1024 / 1024,
      freeMB: free / 1024 / 1024,
      availableMB: available / 1024 / 1024,
      freeFraction: available / total
    };
  }

  _recordProcessMemoryObservation(rssMB) {
    if (!Number.isFinite(rssMB) || rssMB <= 0) {
      return;
    }
    this.observedProcessRssMB.push(rssMB);
    if (this.observedProcessRssMB.length > this.memoryObservationLimit) {
      this.observedProcessRssMB.shift();
    }
  }

  _estimatePerProcessMB() {
    if (this.observedProcessRssMB.length > 0) {
      const sum = this.observedProcessRssMB.reduce((acc, value) => acc + value, 0);
      return Math.max(64, Math.round(sum / this.observedProcessRssMB.length));
    }
    return Math.max(128, parseInt(process.env.PYODIDE_EST_PROC_MB || '512', 10));
  }

  _canSpawnAnotherProcess(activeCount) {
    if (!this.memoryGuardEnabled) {
      return true;
    }

    const pressure = this._memoryPressure();
    const estimated = this._estimatePerProcessMB();
    const projectedAvailableMB = pressure.availableMB - estimated;
    let minFreeMB = Math.max(this.minFreeMemFloorMB, pressure.totalMB * this.minFreeMemFraction);
    if (Number.isFinite(this.minFreeMemCapMB)) {
      minFreeMB = Math.min(minFreeMB, this.minFreeMemCapMB);
    }

    if (projectedAvailableMB < minFreeMB) {
      const bypassFloor = Math.max(this.minFreeMemFloorMB, this.maxWorkerRssMB * 0.75);
      const canBypass = this.memoryGuardSoftBypass &&
        activeCount < Math.max(2, Math.ceil(this.poolSize / 2)) &&
        projectedAvailableMB >= bypassFloor;

      if (!canBypass) {
        logger.warn('Skipping Pyodide worker spawn due to memory guard', {
          component: 'pyodide-process-pool',
          activeCount,
          poolEmpty: activeCount === 0,
          estimatedPerProcessMB: estimated,
          availableMB: pressure.availableMB,
          projectedAvailableMB,
          minFreeMB,
          maxWorkerRssMB: this.maxWorkerRssMB
        });
        return false;
      }

      logger.warn('Memory guard soft bypass allowing worker spawn', {
        component: 'pyodide-process-pool',
        activeCount,
        poolEmpty: activeCount === 0,
        estimatedPerProcessMB: estimated,
        availableMB: pressure.availableMB,
        projectedAvailableMB,
        minFreeMB,
        bypassFloor,
        maxWorkerRssMB: this.maxWorkerRssMB
      });
    }
    return true;
  }
  _admissionAllowed() {
    const p = this._memoryPressure();
    if (p.freeFraction < this.minFreeMemFraction && p.availableMB < this.minFreeMemFloorMB) {
      return { allowed: false, reason: 'low-memory', pressure: p };
    }
    return { allowed: true, pressure: p };
  }

  _effectiveMaxQueueSize() {
    if (!this.adaptiveQueueEnabled) return this.maxQueueSize;
    const p = this._memoryPressure();
    const fraction = Math.max(0.05, Math.min(1, (p.freeFraction - this.minFreeMemFraction) / 0.5));
    const dynamicCap = Math.max(1, Math.floor(this.maxQueueSize * fraction));
    return Math.min(this.maxQueueSize, dynamicCap);
  }

  /**
   * Create a new persistent Pyodide process
   * @param {number} processId - The ID for the new process
   * @returns {Promise<Object>} Promise that resolves to process info
   * @private
   */
  async _createProcess(processId) {
    if (this.circuitBreakerOpen) {
      const now = Date.now();
      if (now - this.lastCircuitOpenAt > this.circuitBreakerResetMs) {
        // Reset circuit breaker after cooldown
        this.circuitBreakerOpen = false;
        this.initFailureStreak = 0;
        logger.warn('Circuit breaker reset, attempting process creation', { component: 'pyodide-process-pool' });
      } else {
        throw new Error('Circuit breaker open - refusing to create new Pyodide processes temporarily');
      }
    }
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
        outputBuffer: '',
        initialMemory: null,
        lastKnownRssMB: null,
        markedForRecycle: null
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
        this.initFailureStreak += 1;
        if (this.initFailureStreak >= this.maxInitFailureStreak && !this.circuitBreakerOpen) {
          this.circuitBreakerOpen = true;
          this.lastCircuitOpenAt = Date.now();
          logger.error('Circuit breaker opened due to repeated process init timeouts', {
            component: 'pyodide-process-pool',
            streak: this.initFailureStreak,
            maxInitFailureStreak: this.maxInitFailureStreak
          });
        }
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
      processInfo.initResolve = (memory) => {
        clearTimeout(initTimeout);
        processInfo.available = true;
        processInfo.initialMemory = memory || null;
        if (memory && typeof memory.rssMB === 'number') {
          processInfo.lastKnownRssMB = memory.rssMB;
          this._recordProcessMemoryObservation(memory.rssMB);
        }
        processInfo.markedForRecycle = null;
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
          // Reset init failure streak on success
          this.initFailureStreak = 0;
          if (processInfo.initResolve) {
            processInfo.initResolve(message.memory || null);
            delete processInfo.initResolve;
          }
        } else {
          logger.error('Process initialization failed', {
            component: 'pyodide-process-pool',
            processId,
            error: message.error
          });
          this.initFailureStreak += 1;
          if (this.initFailureStreak >= this.maxInitFailureStreak && !this.circuitBreakerOpen) {
            this.circuitBreakerOpen = true;
            this.lastCircuitOpenAt = Date.now();
            logger.error('Circuit breaker opened due to repeated init failures', {
              component: 'pyodide-process-pool',
              streak: this.initFailureStreak,
              maxInitFailureStreak: this.maxInitFailureStreak
            });
          }
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
          // Memory growth recycle heuristic using adaptive metrics if available
          const statsRecord = this.processStats.get(processId);
          let recycledForGrowth = false;
          if (statsRecord && statsRecord.metricsHistory && statsRecord.metricsHistory.length >= 2) {
            const lastTwo = statsRecord.metricsHistory.slice(-2);
            const prev = lastTwo[0];
            const curr = lastTwo[1];
            if (prev && curr && typeof prev.deltaMB === 'number' && typeof curr.deltaMB === 'number') {
              const growth = curr.deltaMB - prev.deltaMB;
              if (growth >= this.memGrowthRecycleMB) {
                this.recycledForGrowth += 1;
                logger.info('Recycling process due to memory growth spike', {
                  component: 'pyodide-process-pool',
                  processId,
                  growthMB: growth,
                  thresholdMB: this.memGrowthRecycleMB
                });
                this._recycleProcess(processId);
                recycledForGrowth = true;
              }
            }
          }

          const shouldRecycleForRss = processInfo.markedForRecycle === 'rss_limit';
          if (shouldRecycleForRss) {
            this.recycledForRssLimit += 1;
            this.adaptiveRecycleReason = 'rss_limit';
            processInfo.markedForRecycle = null;
            this._recycleProcess(processId);
          } else if (recycledForGrowth) {
            processInfo.markedForRecycle = null;
          } else if (processInfo.executionCount >= this.maxExecutions) {
            this._recycleProcess(processId);
          } else {
            processInfo.markedForRecycle = null;
            this.availableProcesses.push(processId);
          }

          // Resolve the pending execution
          if (message.success) {
            pendingExecution.resolve(message);
          } else {
            // Include more error details if available
            const error = new Error(message.error || 'Execution failed');
            if (message.errorName) error.name = message.errorName;
            if (message.errorStack) {
              logger.debug('Python execution error details', {
                component: 'pyodide-process-pool',
                processId,
                errorName: message.errorName,
                errorStack: message.errorStack
              });
            }
            pendingExecution.reject(error);
          }
        }
        break;
      }
      case 'post_exec_metrics': {
        if (message.processMemory && typeof message.processMemory.rssMB === 'number') {
          processInfo.lastKnownRssMB = message.processMemory.rssMB;
          this._recordProcessMemoryObservation(message.processMemory.rssMB);
          if (message.processMemory.rssMB >= this.maxWorkerRssMB && processInfo.markedForRecycle !== 'rss_limit') {
            processInfo.markedForRecycle = 'rss_limit';
            logger.warn('Pyodide worker flagged for RSS recycle', {
              component: 'pyodide-process-pool',
              processId,
              rssMB: message.processMemory.rssMB,
              maxWorkerRssMB: this.maxWorkerRssMB
            });
          }
        }
        // Adaptive metrics from child process after execution
        const stats = this.processStats.get(processId) || {};
        // Initialize adaptive fields if missing
        if (!stats.metricsHistory) stats.metricsHistory = [];
        if (typeof stats.tier3Executions !== 'number') stats.tier3Executions = 0;
        if (typeof stats.tier2Executions !== 'number') stats.tier2Executions = 0;
        if (typeof stats.recycleSuggestions !== 'number') stats.recycleSuggestions = 0;
        if (typeof stats.lastDeltaMB !== 'number') stats.lastDeltaMB = 0;
        if (typeof stats.aggressiveCleanups !== 'number') stats.aggressiveCleanups = 0;

        const tier = message.tier || 0;
        const deltaMB = message.stats && typeof message.stats.deltaMB === 'number' ? message.stats.deltaMB : 0;
        stats.lastDeltaMB = deltaMB;
        if (tier === 3) stats.tier3Executions += 1; else if (tier === 2) stats.tier2Executions += 1;
        if (message.recycleSuggested) stats.recycleSuggestions += 1;
        if (message.aggressive) stats.aggressiveCleanups += 1;
        stats.metricsHistory.push({ ts: Date.now(), tier, deltaMB, recycle: !!message.recycleSuggested });
        if (stats.metricsHistory.length > this.metricsHistoryLimit) {
          stats.metricsHistory.shift();
        }
        this.processStats.set(processId, stats);

        // Adaptive recycling decision (only if enabled)
        if (this.adaptiveEnabled) {
          let shouldRecycle = false;
          let reason = '';
          // Respect explicit suggestion
            if (message.recycleSuggested) {
              shouldRecycle = true;
              reason = 'child_suggested';
            }
          // Hard tier 3 more than once in short window
          const recentT3 = stats.metricsHistory.filter(m => Date.now() - m.ts < 60000 && m.tier === 3).length;
          if (!shouldRecycle && recentT3 >= 2) {
            shouldRecycle = true; reason = 'repeated_tier3_1min';
          }
          // Sustained growth: average positive delta over last 10 metrics
          const recent = stats.metricsHistory.slice(-10);
          if (!shouldRecycle && recent.length >= 5) {
            const avgDelta = recent.reduce((s, m) => s + m.deltaMB, 0) / recent.length;
            if (avgDelta > 50) { // 50MB average growth threshold
              shouldRecycle = true; reason = 'avg_growth_gt_50mb_last10';
            }
          }
          if (shouldRecycle && processInfo.available) {
            this.adaptiveRecycleReason = reason;
            this._recycleProcess(processId);
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
    // Admission control (memory pressure)
    const admission = this._admissionAllowed();
    if (!admission.allowed) {
      this.lowMemRejects += 1;
      logger.warn('Admission rejected due to low memory', {
        component: 'pyodide-process-pool',
        freeFraction: admission.pressure.freeFraction.toFixed(4),
        availableMB: Math.round(admission.pressure.availableMB),
        minFreeMemFraction: this.minFreeMemFraction,
        minFreeMemFloorMB: this.minFreeMemFloorMB,
        pendingExecutions: this.pendingExecutions.size
      });
      throw new Error('Server under memory pressure, try later');
    }

    // Adaptive queue sizing
    const effectiveMax = this._effectiveMaxQueueSize();
    if (this.pendingExecutions.size >= effectiveMax) {
      this.backpressureRejects += 1;
      logger.warn('Backpressure rejection', {
        component: 'pyodide-process-pool',
        pending: this.pendingExecutions.size,
        effectiveMaxQueueSize: effectiveMax,
        configuredMaxQueueSize: this.maxQueueSize
      });
      throw new Error('Server busy: too many pending executions');
    }
    if (this.pendingExecutions.size > this.queueHighWaterMark) {
      this.queueHighWaterMark = this.pendingExecutions.size;
    }

    // Wait for an available process
    const processId = await this._getAvailableProcess();
    const executionId = uuidv4();

    return new Promise((resolve, reject) => {
      // Store the pending execution
      this.pendingExecutions.set(executionId, { resolve, reject });

      // Hard timeout
      const timeoutHandle = setTimeout(() => {
        this.pendingExecutions.delete(executionId);
        this._killProcess(processId, 'execution timeout');
        reject(new Error(`Execution timeout after ${timeout}ms`));
      }, timeout);

      // Soft timeout watchdog for early termination under pressure
      const softMs = Math.max(100, Math.floor(timeout * this.softTimeoutFactor));
      const softHandle = setTimeout(() => {
        const pressure = this._memoryPressure();
        if (pressure.freeFraction < this.minFreeMemFraction * 0.75) {
          this.pendingExecutions.delete(executionId);
            this.earlyTerminations += 1;
            logger.warn('Early termination due to memory pressure (soft timeout)', {
              component: 'pyodide-process-pool',
              processId,
              softMs,
              timeout,
              freeFraction: pressure.freeFraction.toFixed(4),
              availableMB: Math.round(pressure.availableMB),
              minFreeMemFraction: this.minFreeMemFraction
            });
            this._killProcess(processId, 'soft-timeout-memory-pressure');
            reject(new Error(`Execution terminated early due to memory pressure at ~${softMs}ms`));
        }
      }, softMs);

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
            clearTimeout(softHandle);
            originalResolve(result);
          },
          reject: (error) => {
            clearTimeout(timeoutHandle);
            clearTimeout(softHandle);
            originalReject(error);
          }
        });
        
      } catch (error) {
        clearTimeout(timeoutHandle);
        clearTimeout(softHandle);
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
   
    if (!this.isInitialized) {
      return;
    }
   
    const attemptSpawn = () => {
      const activeCount = this.processes.filter((p) => p !== null).length;
      if (!this._canSpawnAnotherProcess(activeCount)) {
        return false;
      }
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
      return true;
    };
   
    if (attemptSpawn()) {
      return;
    }
   
    logger.warn('Deferred replacement process creation due to memory guard', {
      component: 'pyodide-process-pool',
      processId,
      maxWorkerRssMB: this.maxWorkerRssMB
    });
   
    setTimeout(() => {
      if (!this.isInitialized || this.processes[processId]) {
        return;
      }
      if (!attemptSpawn()) {
        logger.warn('Memory guard still preventing replacement process spawn', {
          component: 'pyodide-process-pool',
          processId,
          maxWorkerRssMB: this.maxWorkerRssMB
        });
      }
    }, Math.min(this.circuitBreakerResetMs, 30000));
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
      requestedPoolSize: this.requestedPoolSize,
      activeProcesses,
      availableProcesses: this.availableProcesses.length,
      totalExecutions,
      pendingExecutions: this.pendingExecutions.size,
      processStats: Object.fromEntries(this.processStats),
      observedProcessRssMB: this.observedProcessRssMB,

      memoryAdaptive: {
        totalMemMB: this.adaptiveSizing.totalMemMB,
        perProcessMB: this.adaptiveSizing.perProcessMB,
        maxFraction: this.adaptiveSizing.maxPoolMemFraction,
        memCapCount: this.adaptiveSizing.memCapCount,
        envMaxPool: this.adaptiveSizing.envMaxPool,
        cpuCount: this.adaptiveSizing.cpuCount,
        finalPoolSize: this.adaptiveSizing.finalPoolSize,
        reason: this.adaptiveSizing.reason,
        minFreeMemCapMB: Number.isFinite(this.minFreeMemCapMB) ? this.minFreeMemCapMB : null
      },
      adaptive: {
        enabled: this.adaptiveEnabled,
        circuitBreakerOpen: this.circuitBreakerOpen,
        initFailureStreak: this.initFailureStreak,
        maxInitFailureStreak: this.maxInitFailureStreak,
        adaptiveRecycleReason: this.adaptiveRecycleReason,
        backpressure: {
          maxQueueSize: this.maxQueueSize,
            currentPending: this.pendingExecutions.size,
            rejects: this.backpressureRejects,
            effectiveMaxQueueSize: this._effectiveMaxQueueSize(),
            highWaterMark: this.queueHighWaterMark
        },
        protection: {
          minFreeMemFraction: this.minFreeMemFraction,
          minFreeMemFloorMB: this.minFreeMemFloorMB,
          lowMemRejects: this.lowMemRejects,
          earlyTerminations: this.earlyTerminations,
          recycledForGrowth: this.recycledForGrowth,
          recycledForRssLimit: this.recycledForRssLimit,
          softTimeoutFactor: this.softTimeoutFactor,
          memGrowthRecycleMB: this.memGrowthRecycleMB,
          rssLimitMB: this.maxWorkerRssMB,
          headroomCapMB: Number.isFinite(this.minFreeMemCapMB) ? this.minFreeMemCapMB : null,
          softBypassEnabled: this.memoryGuardSoftBypass
        }
      }
    };
  }

  /**
   * Get memory usage for all processes
   * @returns {Object} Memory usage statistics
   */
  getMemoryUsage() {
    const memoryStats = {
      timestamp: new Date().toISOString(),
      nodeProcess: {
        rss: process.memoryUsage().rss,
        heapUsed: process.memoryUsage().heapUsed,
        heapTotal: process.memoryUsage().heapTotal,
        external: process.memoryUsage().external
      },
      childProcesses: []
    };

    this.processes.forEach((processInfo, index) => {
      if (processInfo && processInfo.process && !processInfo.process.killed) {
        try {
          // Note: We can't directly get child process memory from Node.js
          // but we can track the process state
          memoryStats.childProcesses.push({
            id: index,
            pid: processInfo.process.pid,
            executions: this.processStats.get(index)?.totalExecutions || 0,
            age: Date.now() - processInfo.startTime
          });
        } catch {
          // Process might have died
        }
      }
    });

    // Log memory usage if it's high
    const totalMemoryMB = memoryStats.nodeProcess.rss / 1024 / 1024;
    if (totalMemoryMB > 1024) { // More than 1GB
      logger.warn('High memory usage detected', {
        component: 'pyodide-process-pool',
        totalMemoryMB: totalMemoryMB.toFixed(2),
        activeProcesses: memoryStats.childProcesses.length,
        ...memoryStats
      });
    }

    return memoryStats;
  }

  /**
   * Force garbage collection and memory cleanup in all processes
   * Call this after running memory-intensive tests
   */
  async forceCleanup() {
    logger.info('Forcing memory cleanup in all processes', {
      component: 'pyodide-process-pool',
      activeProcesses: this.processes.filter(p => p !== null).length
    });

    // Send cleanup command to all active processes
    const cleanupPromises = this.processes
      .filter(p => p !== null)
      .map(processInfo => {
        return new Promise((resolve) => {
          try {
            processInfo.process.stdin.write(JSON.stringify({
              type: 'cleanup',
              executionId: `cleanup-${Date.now()}`
            }) + '\n');
            setTimeout(resolve, 500); // Give time for cleanup
          } catch (error) {
            logger.warn('Failed to send cleanup command to process', {
              component: 'pyodide-process-pool',
              processId: processInfo.id,
              error: error.message
            });
            resolve();
          }
        });
      });

    await Promise.all(cleanupPromises);
  }

  /**
   * Force recycle all processes - useful after memory-intensive operations
   */
  async forceRecycleAll() {
    logger.info('Force recycling all processes', {
      component: 'pyodide-process-pool',
      activeProcesses: this.processes.filter(p => p !== null).length
    });

    // Mark all processes for recycling
    for (let i = 0; i < this.processes.length; i++) {
      if (this.processes[i]) {
        this._killProcess(i, 'forced recycle for memory cleanup');
      }
    }

    // Wait for new processes to initialize
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  /**
   * Start memory monitoring
   * @private
   */
  startMemoryMonitoring() {
    // Monitor memory every 30 seconds
    this.memoryMonitorInterval = setInterval(() => {
      const stats = this.getMemoryUsage();
      const totalMemoryMB = stats.nodeProcess.rss / 1024 / 1024;
      
      // Log periodically if memory is growing
      if (totalMemoryMB > 500) { // More than 500MB
        logger.info('Memory usage report', {
          component: 'pyodide-process-pool',
          totalMemoryMB: totalMemoryMB.toFixed(2),
          heapUsedMB: (stats.nodeProcess.heapUsed / 1024 / 1024).toFixed(2),
          activeProcesses: stats.childProcesses.length,
          totalExecutions: stats.childProcesses.reduce((sum, p) => sum + p.executions, 0)
        });
      }
      
      // Force cleanup if memory is very high
      if (totalMemoryMB > 2048) { // More than 2GB
        logger.warn('Very high memory usage, forcing cleanup', {
          component: 'pyodide-process-pool',
          totalMemoryMB: totalMemoryMB.toFixed(2)
        });
        this.forceCleanup().catch(err => {
          logger.error('Failed to force cleanup', {
            component: 'pyodide-process-pool',
            error: err.message
          });
        });
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Stop memory monitoring
   * @private
   */
  stopMemoryMonitoring() {
    if (this.memoryMonitorInterval) {
      clearInterval(this.memoryMonitorInterval);
      this.memoryMonitorInterval = null;
    }
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
    this.stopMemoryMonitoring();

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

  /**
   * Compute an adaptive pool size based on host memory and env hints.
   * Keeps memory pressure low during large test runs to avoid OS-level crashes.
   * Order of constraints:
   *   1. Hard env max (PYODIDE_MAX_POOL_SIZE) if provided
   *   2. Memory cap derived from (totalMem * fraction) / perProcessEstimate
   *   3. Requested size
   * Returns an object with detailed reasoning.
   * @param {number} requested - originally requested pool size
   * @returns {{finalPoolSize:number, reason:string, totalMemMB:number, perProcessMB:number, maxPoolMemFraction:number, memCapCount:number, envMaxPool:number}} Object containing sizing decision details
   * @private
   */
  _computeAdaptivePoolSize(requested) {
    const totalMemMB = Math.max(1, Math.round(os.totalmem() / 1024 / 1024));
    const perProcessMB = Math.max(50, parseInt(process.env.PYODIDE_EST_PROC_MB || '250', 10));
    const maxPoolMemFraction = Math.min(0.9, Math.max(0.05, parseFloat(process.env.PYODIDE_MAX_POOL_MEM_FRACTION || '0.5')));
    const envMaxPool = parseInt(process.env.PYODIDE_MAX_POOL_SIZE || String(Number.MAX_SAFE_INTEGER), 10);
    let cpuCount = 1;
    try {
      const cpuInfo = typeof os.cpus === 'function' ? os.cpus() : [];
      cpuCount = Array.isArray(cpuInfo) && cpuInfo.length > 0 ? cpuInfo.length : 1;
    } catch {
      cpuCount = 1;
    }

    // memory-derived cap
    const memCapCount = Math.max(1, Math.floor((totalMemMB * maxPoolMemFraction) / perProcessMB));

    let finalPoolSize = requested;
    let reason = 'requested_ok';

    if (requested > envMaxPool) {
      finalPoolSize = envMaxPool;
      reason = 'capped_env_max';
    }
    if (finalPoolSize > memCapCount) {
      finalPoolSize = memCapCount;
      reason = reason === 'requested_ok' ? 'capped_memory' : reason + '+memory';
    }
    if (finalPoolSize < 1) {
      finalPoolSize = 1; // safety
      reason = 'safety_min_1';
    }

    if (finalPoolSize !== requested) {
      logger.warn('Adaptive pool size adjustment', {
        component: 'pyodide-process-pool',
        requested,
        finalPoolSize,
        reason,
        totalMemMB,
        perProcessMB,
        maxPoolMemFraction,
        memCapCount,
        envMaxPool,
        cpuCount
      });
    }

    return { finalPoolSize, reason, totalMemMB, perProcessMB, maxPoolMemFraction, memCapCount, envMaxPool, cpuCount };
  }
}

module.exports = PyodideProcessPool;
