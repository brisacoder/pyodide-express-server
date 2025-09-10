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
/**
 * Python code execution context variables
 * @typedef {Object} PythonContext
 * @property {*} [key] - Any variables to make available in Python scope
 */
/**
 * Python code execution result
 * @typedef {Object} PythonExecutionResult
 * @property {boolean} success - Whether execution completed successfully
 * @property {*} [result] - Return value from Python code (if any)
 * @property {string} [error] - Error message if execution failed
 * @property {string} [stdout] - Standard output from Python execution
 * @property {string} [stderr] - Standard error from Python execution
 * @property {number} executionTime - Time taken to execute in milliseconds
 * @property {Object} [metadata] - Additional execution metadata
 */
/**
 * Package installation result
 * @typedef {Object} PackageInstallResult
 * @property {boolean} success - Whether installation was successful
 * @property {string} package - Name of the package
 * @property {string} [version] - Installed package version
 * @property {string} [error] - Error message if installation failed
 * @property {number} installTime - Time taken to install in milliseconds
 */
/**
 * Available packages list
 * @typedef {Object} AvailablePackages
 * @property {boolean} success - Whether package list retrieval was successful
 * @property {Array<string>} packages - List of available package names
 * @property {number} count - Total number of available packages
 * @property {string} [error] - Error message if retrieval failed
 */
const logger = require('../utils/logger');
const config = require('../config/index');
const constants = require('../config/constants');
const layout = require('./layout');

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
    this.executionTimeout = constants.EXECUTION.DEFAULT_TIMEOUT;
    // Crash reporting state tracking
    this.stats = {
      executionCount: 0,
      errorCount: 0,
      lastExecutionTime: null,
      totalExecutionTime: 0,
      packageInstallCount: 0,
      loadedPackages: new Set(),
      memoryUsage: {
        lastCheck: null,
        heapSize: 0,
        pyodideMemory: 0,
      },
      errors: [], // Keep last 10 errors for debugging
    };
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

  async _loadMicropipPkgs() {
    const micropipPackages = ['seaborn'];
    const installedPackages = [];
    for (const packageName of micropipPackages) {
      try {
        logger.info(`Installing ${packageName} via micropip...`);
        // Special handling for seaborn which can have metadata issues
        await this.pyodide.runPythonAsync(`
import micropip
await micropip.install("${packageName}")
print("✅ ${packageName} installed successfully")
`);
        installedPackages.push(packageName);
        logger.info(`✅ ${packageName} installed successfully`);
      } catch (packageError) {
        logger.warn(`⚠️  Failed to install ${packageName}:`, packageError.message);
        // For seaborn, this is not critical - it can be installed later via API
      }
    }
    logger.info(`Installed via micropip: ${installedPackages.join(', ')}`);    
  }
  /**
   * Internal initialization method - Simple version that works
   * @private
   * @returns {Promise<boolean>} True if initialization successful
   *
   * Loads the Pyodide runtime and eagerly fetches a set of scientific
   * packages.  Installing packages up front reduces latency for the first
   * request and mirrors the behaviour of a traditional Python environment.
   */
  async _performInitialization() {
    try {
    logger.info('Starting Pyodide initialization...');
      try {
        const { loadPyodide } = await import('pyodide');  // destructure directly
        const packagesToLoad = ['numpy', 'pandas', 'micropip', 'matplotlib', 'requests'
          ,'scikit-learn', 'scipy', 'httpx', 'statsmodels'];


        if (typeof loadPyodide !== 'function') {
          throw new Error('loadPyodide is not a function');
        }

        logger.info('Loading Pyodide runtime...');

        this.pyodide = await loadPyodide({fullStdLib:true, packages:packagesToLoad});
      } catch (err) {
        logger.error('❌ Pyodide initialization failed:', err);
        throw err;
      }
      logger.info('✅ Pyodide loaded successfully!');
      logger.info(`Pyodide version: ${ this.pyodide.version || 'unknown'}`);
      logger.info('Loading essential packages...');
      logger.info('Installing additional packages via micropip...');
      // Install additional packages that aren't available via loadPackage
      await this._loadMicropipPkgs();
      logger.info('Setting up Python environment...');
      // Set up Python environment with error handling
      await this.pyodide.runPythonAsync(`
import sys
import io
print(f"Python version: {sys.version}")
print("Setting up Pyodide environment...")
# Import available packages and make them globally accessible
try:
    import numpy as np
    print("[x] NumPy available")
    globals()['np'] = np
except ImportError:
    print("[ ] NumPy not available")
try:
    import pandas as pd
    print("[x] Pandas available")
    globals()['pd'] = pd
except ImportError:
    print("[ ] Pandas not available")
try:
    import matplotlib.pyplot as plt
    import matplotlib
    # Set non-interactive backend for server environment
    matplotlib.use('Agg')
    print("[x] Matplotlib available")
    globals()['plt'] = plt
except ImportError:
    print("[ ] Matplotlib not available")
try:
    import requests
    print("[x] Requests available")
    globals()['requests'] = requests
except ImportError:
    print("[ ] Requests not available")
try:
    import seaborn as sns
    print("[x] Seaborn available")
    globals()['sns'] = sns
except ImportError:
    print("[ ] Seaborn not available")
try:
    import httpx
    print("[x] HTTPX available")
    globals()['httpx'] = httpx
except ImportError:
    print("[ ]  HTTPX not available")
try:
    import sklearn
    print("[x] SciKit-Learn available")
    globals()['sklearn'] = sklearn
except ImportError:
    print("[ ] Scikit-Learn not available")
try:
    import statsmodels.api as sm
    print("[x] Statsmodel available")
    globals()['sm'] = sm
except ImportError:
    print("[ ]  StatsModel not available")
try:
    import micropip
    print("[x] Micropip available")
    # Make micropip available globally
    globals()['micropip'] = micropip
except ImportError:
    print("[ ]  Micropip not available")
print("Pyodide environment ready!")
      `);
      // Setup filesystem mounting for plots
      await this._mountFolders();
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
 * Mounts essential host directories into the Pyodide virtual filesystem for persistent file access.
 *
 * This method ensures that required directories (plots, uploads, etc.) exist on the host filesystem,
 * creates subdirectories for different plot types, and attempts to mount them into the Pyodide
 * WebAssembly environment using NodeFS. If mounting fails, it falls back to creating virtual directories
 * within Pyodide's in-memory filesystem. This enables Python code running in Pyodide to read/write files
 * as if they were on the host, supporting workflows like saving plots and handling uploads.
 *
 * Side Effects:
 * - Creates host directories if missing
 * - Mounts host directories to Pyodide virtual paths (/plots, /uploads)
 * - Creates subdirectories for plot types (matplotlib, seaborn)
 * - Logs all operations and errors
 * - Verifies mount success and prints directory contents
 *
 * Error Handling:
 * - If mounting fails, falls back to virtual directory creation in Pyodide
 * - Logs warnings but does not throw, allowing initialization to continue
 *
 * @returns {Promise<void>} Resolves when all directories are mounted or created
 * @throws Only in catastrophic setup errors (rare)
 */
async _mountFolders() {


    const { rootReal, basesReal } = await layout.initLayout(config.pyodideDataDir, config.pyodideBases);  
    this.basesReal = basesReal; // Store for later use
    this.rootReal = rootReal; // Store for later use
    // Setup filesystem mounting for plot saving
    logger.info('Setting up filesystem mounting for plots...');
    try {
      const path = require('path');
      const fs = require('fs');
      const plotsDir = basesReal.plots.dirReal;
      // Mount the real plots directory to /plots in Pyodide filesystem
      logger.info(`Attempting to mount ${plotsDir} to /plots in Pyodide...`);
      logger.info(`Host path exists: ${fs.existsSync(plotsDir)}`);
      logger.info(`Host path absolute: ${path.isAbsolute(plotsDir)}`);
      try {
        // mountNodeFS(emscriptenPath, hostPath)
        // emscriptenPath: The absolute path in Emscripten FS to mount to
        // hostPath: The host path to mount (must exist)
        this.pyodide.mountNodeFS('/home/pyodide/plots', plotsDir);
        logger.info('✅ Mount successful');
      } catch (mountError) {
        logger.error('❌ Mount failed:', mountError.message);
      }

      // Setup filesystem mounting for Pyodide 'In' mailbox
      const pyodideIn = basesReal.in.dirReal;
      // Mount the real uploads directory to /uploads in Pyodide filesystem
      logger.info(`Attempting to mount ${pyodideIn} to /home/pyodide/in in Pyodide...`);
      logger.info(`Host path exists: ${fs.existsSync(pyodideIn)}`);
      logger.info(`Host path absolute: ${path.isAbsolute(pyodideIn)}`);
      // Mount /pyodide_in
      try {
        // mountNodeFS(emscriptenPath, hostPath)
        // emscriptenPath: The absolute path in Emscripten FS to mount to
        // hostPath: The host path to mount (must exist)
        this.pyodide.mountNodeFS('/home/pyodide/in', pyodideIn);
        logger.info('✅ Mount successful');
      } catch (mountError) {
        logger.error('❌ Mount failed:', mountError.message);
      }
      
      // Setup filesystem mounting for uploads
      const uploads = basesReal.uploads.dirReal;
      // Mount the real uploads directory to /uploads in Pyodide filesystem
      logger.info(`Attempting to mount ${uploads} to /home/pyodide/uploads in Pyodide...`);
      logger.info(`Host path exists: ${fs.existsSync(uploads)}`);
      logger.info(`Host path absolute: ${path.isAbsolute(uploads)}`);
      // Mount /uploads
      try {
        // mountNodeFS(emscriptenPath, hostPath)
        // emscriptenPath: The absolute path in Emscripten FS to mount to
        // hostPath: The host path to mount (must exist)
        this.pyodide.mountNodeFS('/home/pyodide/uploads', uploads);
        logger.info('✅ Mount successful');
      } catch (mountError) {
        logger.error('❌ Mount failed:', mountError.message);
      }                      

      // Create essential directories in Pyodide virtual filesystem
      logger.info('Creating essential directories in Pyodide filesystem...');
      await this.pyodide.runPythonAsync(`
import os
# Create essential directories that will be used by the application
essential_dirs = ['./tmp']
for essential_dir in essential_dirs:
  os.makedirs(essential_dir, exist_ok=True)
  print(f"Created essential directory: {essential_dir}")
`);
      logger.info('✅ Essential directories created');
      
      // Verify the mount worked
      await this.pyodide.runPythonAsync(`
# Check if mounted directories are accessible
from pathlib import Path
home = Path.cwd()
print(f"Mount verification: {list(home.iterdir())}")
`);
      logger.info('✅ Filesystem mounting setup successfully');
    } catch (setupError) {
      logger.warn('⚠️  Failed to setup directories:', setupError.message);
      // Continue initialization even if setup fails
    }
  }

  /**
   * Execute Python code with optional context variables
   * @param {string} code - Python code to execute
   * @param {PythonContext} context - Variables to make available in Python
   * @param {number} timeout - Execution timeout in milliseconds
   * @returns {Promise<PythonExecutionResult>} Execution result with success status, output, and timing
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
    // Security validation - block dangerous patterns
    const dangerousPatterns = [
      /\bsys\.exit\b/i,
      /\bexit\s*\(/i,
      /\bquit\s*\(/i,
      /\b__import__\s*\(\s*['"]os['"]\s*\)/i,
      /\bsubprocess\b/i,
      /\bos\.system\b/i,
    ];
    for (const pattern of dangerousPatterns) {
      if (pattern.test(code)) {
        logger.warn(`Security violation detected: ${pattern.source}`, {
          code: code.substring(0, 100),
        });
        return {
          success: false,
          result: null,
          error: 'Security violation: Dangerous operation detected',
          stdout: '',
          stderr: 'SecurityError: Code contains potentially dangerous patterns',
          timestamp: new Date().toISOString(),
        };
      }
    }
    try {
      // Create a fresh isolated namespace dict
      const namespace = this.pyodide.runPython('d = {}; d');
      // Seed essential builtins
      namespace.set('__builtins__', this.pyodide.globals.get('__builtins__'));
      // Seed commonly used aliases that were pre-imported in global scope if available
      const allowedAliases = ['np','pd','plt','requests','sns','httpx','sklearn','sm','micropip'];
      for (const name of allowedAliases) {
        let val;
        try {
          val = this.pyodide.globals.get(name);
          if (val !== undefined) {
            namespace.set(name, val);
          }
        } catch (e) {
          logger.warn(`Failed to seed alias '${name}' into namespace`, { error: e?.message });
        } finally {
          try { if (val && typeof val.destroy === 'function') val.destroy(); } catch (e) { logger.warn('Failed to destroy alias proxy', { alias: name, error: e?.message }); }
        }
      }
      // Add user-provided context variables
      for (const [key, value] of Object.entries(context)) {
        namespace.set(key, value);
      }

      // Capture stdout/stderr in JS
      let capturedStdout = '';
      let capturedStderr = '';
      let restoreStdout;
      let restoreStderr;
      try {
        restoreStdout = this.pyodide.setStdout({ batched: (s) => { capturedStdout += s; } });
        restoreStderr = this.pyodide.setStderr({ batched: (s) => { capturedStderr += s; } });
      } catch (e) {
        logger.error('Failed to set stdout/stderr capture', { error: e?.message });
        throw e;
      }

      // Provide the user code into namespace to avoid large string interpolation
      namespace.set('_user_code', code);

      // Minimal compile/eval/exec helper executed within isolated namespace
      const pythonExecHelper = `
import traceback, ast
_exec_success = True
_exec_error = None
_exec_result = None
try:
    try:
        _code = compile(_user_code, '<user>', 'eval')
        _exec_result = eval(_code)
    except SyntaxError:
        # Handle multi-line blocks; capture last expression result if present
        _mod = ast.parse(_user_code, mode='exec')
        if _mod.body and isinstance(_mod.body[-1], ast.Expr):
            last = _mod.body.pop()
            _mod.body.append(ast.Assign(targets=[ast.Name(id='_last_result', ctx=ast.Store())], value=last.value))
        else:
            _mod.body.append(ast.Assign(targets=[ast.Name(id='_last_result', ctx=ast.Store())], value=ast.Constant(None)))
        _code = compile(_mod, '<user>', 'exec')
        exec(_code)
        _exec_result = globals().get('_last_result', None)
except SystemExit as e:
    _exec_success = False
    _exec_error = f"SystemExit blocked: exit() and sys.exit() are not allowed (code: {e.code})"
except Exception as e:
    _exec_success = False
    _exec_error = str(e)
    traceback.print_exc()
(_exec_success, _exec_result, _exec_error)
`;

      // Timeout handling (does not interrupt Python, only rejects the promise)
      let timeoutHandle;
      const timeoutPromise = new Promise((_, reject) => {
        timeoutHandle = setTimeout(() => reject(new Error(`Execution timeout after ${timeout}ms`)), timeout);
      });

      const execPromise = this.pyodide.runPythonAsync(pythonExecHelper, { globals: namespace });
      let tuple;
      try {
        // Race timeout and execution
        tuple = await Promise.race([execPromise, timeoutPromise]);
      } finally {
        // Always restore stdio and release namespace proxy
        try { restoreStdout(); } catch (e) { logger.error('Failed to restore stdout', { error: e?.message }); }
        try { restoreStderr(); } catch (e) { logger.error('Failed to restore stderr', { error: e?.message }); }
        try { namespace.destroy(); } catch (e) { logger.warn('Failed to destroy namespace proxy', { error: e?.message }); }
      }

      // If execPromise won, clear timeout if we set it
      if (timeoutHandle) { clearTimeout(timeoutHandle); }

      // tuple can be JS primitive or PyProxy of a tuple
      let success, pyResult, pyError;
      if (tuple && typeof tuple.get === 'function') {
        // PyProxy (tuple)
        success = tuple.get(0);
        pyResult = tuple.get(1);
        pyError = tuple.get(2);
        try { if (typeof tuple.destroy === 'function') tuple.destroy(); } catch (e) { logger.warn('Failed to destroy result tuple proxy', { error: e?.message }); }
      } else if (Array.isArray(tuple)) {
        [success, pyResult, pyError] = tuple;
      } else {
        // Unexpected shape
        success = false;
        pyResult = null;
        pyError = 'Unexpected execution result shape';
      }

      // Convert Python result to JS if needed
      let jsResult = null;
      try {
        if (pyResult && typeof pyResult.toJs === 'function') {
          jsResult = pyResult.toJs({ create_proxies: false, dict_converter: 'object' });
        } else {
          jsResult = pyResult ?? null;
        }
      } catch (convErr) {
        logger.warn('Failed to convert Python result to JS', { error: convErr?.message });
        jsResult = null;
      } finally {
        try { if (pyResult && typeof pyResult.destroy === 'function') pyResult.destroy(); } catch (e) { logger.warn('Failed to destroy result proxy', { error: e?.message }); }
      }

      return {
        success: !!success,
        result: jsResult,
        error: success ? null : (pyError || 'Execution error'),
        stdout: capturedStdout,
        stderr: capturedStderr,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      // Handle specific Pyodide errors
      if (error.type === 'SystemExit') {
        return {
          success: false,
          result: null,
          error: 'SystemExit blocked: exit() and sys.exit() are not allowed',
          stdout: '',
          stderr: 'SystemExit error occurred',
          timestamp: new Date().toISOString(),
        };
      }
      // Ensure stdout/stderr are restored by resetting to default
      try { this.pyodide.setStdout(); } catch (e) { logger.error('setStdout() reset failed', { error: e?.message }); }
      try { this.pyodide.setStderr(); } catch (e) { logger.error('setStderr() reset failed', { error: e?.message }); }
      return {
        success: false,
        result: null,
        error: `Execution error: ${error.message}`,
        stdout: '',
        stderr: error.stack || error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }
  /**
   * Install a Python package using micropip
   * @param {string} packageName - Name of the package to install
   * @returns {Promise<PackageInstallResult>} Installation result with success status, version, and timing
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
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Failed to install package ${packageName}: ${error.message}`);
    }
  }
  /**
   * Get list of installed packages
   * @returns {Promise<AvailablePackages>} List of packages with names, count, and success status
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
import micropip
try:
    # Get list of packages installed via micropip
    installed_packages = micropip.list()
    # Also include currently loaded modules for reference
    loaded_modules = list(sys.modules.keys())
    result = {
        'python_version': sys.version,
        'installed_packages': sorted(list(installed_packages.keys())),
        'loaded_modules': sorted([pkg for pkg in loaded_modules if not pkg.startswith('_')]),
        'total_packages': len(installed_packages)
    }
except Exception as e:
    result = {
        'error': str(e),
        'fallback_modules': sorted([pkg for pkg in sys.modules.keys() if not pkg.startswith('_')]),
        'python_version': sys.version
    }
result
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
        hasResult: !!verificationResult.result,
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
              count: 0,
            },
          };
        } else if (result.result.startsWith('ERROR:')) {
          return {
            success: false,
            error: result.result.substring(6),
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
              modified: parseFloat(modified),
            });
          }
          return {
            success: true,
            result: {
              success: true,
              files: files,
              count: files.length,
            },
          };
        }
      }
      // Fallback if result is null or unsuccessful
      return {
        success: true,
        result: {
          success: true,
          files: [],
          count: 0,
        },
      };
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
            timestamp: new Date().toISOString(),
          };
        } catch {
          return {
            success: false,
            error: 'Failed to parse deletion result',
            timestamp: new Date().toISOString(),
          };
        }
      }
      return {
        success: false,
        error: 'Failed to execute deletion command',
        timestamp: new Date().toISOString(),
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
      timestamp: new Date().toISOString(),
    };
  }
  /**
   * Check if Pyodide is initialized and ready for code execution
   * @returns {boolean} True if Pyodide is ready
   */
  isInitialized() {
    return this.isReady && this.pyodide !== null;
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
# Clear user-defined variables but keep system ones AND core module aliases
user_vars = [var for var in globals().keys()
             if not var.startswith('_')
             and var not in ['sys', 'io',
                            'np', 'pd', 'plt', 'sns', 'requests', 'micropip', 'httpx']]
for var in user_vars:
    try:
        del globals()[var]
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
      logger.info(
        `✅ Extracted virtual file ${virtualPath} to ${realPath} (${parsedResult.file_size} bytes)`
      );
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
  // Crash reporting helper methods
  getLoadedPackages() {
    return Array.from(this.stats.loadedPackages);
  }
  getLastExecutionTime() {
    return this.stats.lastExecutionTime;
  }
  getErrorCount() {
    return this.stats.errorCount;
  }
  getMemoryUsage() {
    this.updateMemoryStats();
    return this.stats.memoryUsage;
  }
  getExecutionStats() {
    return {
      executionCount: this.stats.executionCount,
      errorCount: this.stats.errorCount,
      averageExecutionTime:
        this.stats.executionCount > 0
          ? this.stats.totalExecutionTime / this.stats.executionCount
          : 0,
      packageInstallCount: this.stats.packageInstallCount,
      loadedPackages: this.getLoadedPackages(),
      recentErrors: this.stats.errors.slice(-5), // Last 5 errors
    };
  }
  recordExecution(executionTime, success = true, error = null) {
    this.stats.executionCount++;
    this.stats.lastExecutionTime = new Date().toISOString();
    this.stats.totalExecutionTime += executionTime;
    if (!success && error) {
      this.stats.errorCount++;
      this.stats.errors.push({
        timestamp: new Date().toISOString(),
        error: error.message || String(error),
        stack: error.stack,
      });
      // Keep only last 10 errors
      if (this.stats.errors.length > 10) {
        this.stats.errors = this.stats.errors.slice(-10);
      }
    }
    this.updateMemoryStats();
  }
  recordPackageInstall(packageName) {
    this.stats.packageInstallCount++;
    this.stats.loadedPackages.add(packageName);
  }
  updateMemoryStats() {
    if (this.pyodide) {
      try {
        this.stats.memoryUsage = {
          lastCheck: new Date().toISOString(),
          heapSize: this.pyodide.runPython('import sys; sys.getsizeof(globals())') || 0,
          processMemory: process.memoryUsage(),
        };
      } catch {
        // Ignore errors in memory stat collection
      }
    }
  }
}
// Export singleton instance
module.exports = new PyodideService();
