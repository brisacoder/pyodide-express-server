#!/usr/bin/env node
/**
 * Pyodide Child Process Executor
 * 
 * This separate Node.js process runs Pyodide in isolation,
 * allowing the main server to kill it if execution hangs.
 * 
 * Communication via stdin/stdout with JSON messages.
 */

const { loadPyodide } = require('pyodide');
const path = require('path');

let pyodide = null;
let initialized = false;
let contextVariableNames = new Set(); // Track context variables for cleanup

/**
 * Initialize Pyodide with required packag/**
 * Send message to parent process
 * @param {Object} data - Data to send to parent
 * @returns {Promise} Promise that resolves when message is fully sent
 */
function sendMessage(data) {
  return new Promise((resolve, reject) => {
    try {
      const jsonData = JSON.stringify(data) + '\n';
      const success = process.stdout.write(jsonData);
      
      if (success) {
        // Data was written immediately (no buffering needed)
        resolve();
      } else {
        // Data was buffered, wait for drain event
        process.stdout.once('drain', resolve);
        process.stdout.once('error', reject);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      reject(error);
    }
  });
}

/*
 * Pyodide initialization and system setup
 */
async function initializePyodide() {
  try {
    // Load Pyodide with locally available packages
    pyodide = await loadPyodide({
      packages: ['micropip', 'numpy', 'pandas', 'matplotlib', 'scikit-learn', 'statsmodels']
    });

    // Setup filesystem directories
    // Create /home/pyodide directories for user data
    // Use a helper function to create directories safely
    const mkdirSafe = (path) => {
      try {
        pyodide.FS.mkdir(path);
      } catch {
        // Directory might already exist, which is fine
      }
    };

    // Mount host filesystem directories from pyodide_data
    // Note: /home and /home/pyodide are automatically created by VFS
    const projectRoot = path.resolve(__dirname, '../..');
    const pyodideDataDir = path.join(projectRoot, 'pyodide_data');
    const uploadsPath = path.join(pyodideDataDir, 'uploads');
    const plotsPath = path.join(pyodideDataDir, 'plots');
    
    try {
      pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: uploadsPath }, '/home/pyodide/uploads');
      pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: plotsPath }, '/home/pyodide/plots');
      
      // Create subdirectories after mounting (these will be created on the host filesystem)
      mkdirSafe('/home/pyodide/plots/matplotlib');
      mkdirSafe('/home/pyodide/plots/seaborn');
      mkdirSafe('/home/pyodide/plots/base64');
    } catch (err) {
      console.warn('Failed to mount filesystem directories:', err.message);
      // Fallback to virtual filesystem if mounting fails
      mkdirSafe('/home/pyodide/uploads');
      mkdirSafe('/home/pyodide/plots');
      mkdirSafe('/home/pyodide/plots/matplotlib');
      mkdirSafe('/home/pyodide/plots/seaborn');
      mkdirSafe('/home/pyodide/plots/base64');
    }

    // Setup Python environment

    // Setup Python environment
    await pyodide.runPythonAsync(`
import sys
import os
from pathlib import Path

# Set up home directory
os.environ['HOME'] = '/home/pyodide'

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set default figure parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# Install and configure seaborn if not already available
try:
    import seaborn as sns
    sns.set_theme()  # Set default seaborn theme
except ImportError:
    import micropip
    await micropip.install('seaborn')
    import seaborn as sns
    sns.set_theme()
plt.rcParams['savefig.bbox'] = 'tight'
    `);

    initialized = true;
    sendMessage({ type: 'initialized', success: true });
  } catch (error) {
    sendMessage({ 
      type: 'initialized', 
      success: false, 
      error: error.message,
      stack: error.stack
    });
    process.exit(1);
  }
}

/**
 * Execute Python code in isolated environment
 * @param {string} code - Python code to execute
 * @param {Object} context - Variables to make available in Python scope
 * @param {string|null} executionId - Unique identifier for this execution
 */
async function executeCode(code, context = {}, executionId = null) {
  if (!initialized) {
    sendMessage({
      type: 'result',
      executionId,
      success: false,
      error: 'Pyodide not initialized'
    });
    return;
  }

  const startTime = Date.now();
  
  try {
    // SAFETY: Ensure Python streams are in a good state before execution
    try {
      pyodide.runPython(`
import sys
# Reset streams to their original state if they were redirected
if hasattr(sys, '__stdout__'):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
      `);
    } catch (resetError) {
      // console.error('Warning: Could not reset Python streams before execution');
    }
    
    // Set up context variables and track them for cleanup
    for (const [key, value] of Object.entries(context)) {
      pyodide.globals.set(key, value);
      contextVariableNames.add(key); // Track for cleanup
    }

    // Capture stdout/stderr
    pyodide.runPython(`
import sys
from io import StringIO

# Capture stdout and stderr
_captured_stdout = StringIO()
_captured_stderr = StringIO()
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = _captured_stdout
sys.stderr = _captured_stderr
    `);

    // Execute the user code
    let result;
    try {
      // Check if the code contains await statements or async functions
      const hasAwait = /\bawait\s+/.test(code);
      const hasAsyncDef = /\basync\s+def\s+/.test(code);
      
      if (hasAwait || hasAsyncDef) {
        // Use async execution for code with await
        result = await pyodide.runPythonAsync(code);
      } else {
        // Use sync execution for regular code
        result = pyodide.runPython(code);
      }
    } catch (pythonError) {
      // Try to get captured output before restoring
      let stdout = '';
      let stderr = '';
      
      try {
        stdout = pyodide.runPython('_captured_stdout.getvalue()') || '';
        stderr = pyodide.runPython('_captured_stderr.getvalue()') || '';
      } catch {
        // If getting output fails, use empty strings
        stderr = pythonError.toString();
      }
      
      // CRITICAL: Always restore stdout/stderr even if getting output failed
      try {
        pyodide.runPython(`
sys.stdout = _original_stdout
sys.stderr = _original_stderr
        `);
      } catch (restoreError) {
        // console.error('Failed to restore stdout/stderr after Python error:', restoreError);
        // If restoration fails, try to reinitialize the streams
        try {
          pyodide.runPython(`
import sys
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
          `);
        } catch {
          // console.error('Critical: Could not restore Python streams');
        }
      }
      
      sendMessage({
        type: 'result',
        executionId,
        success: false,
        error: pythonError.message || pythonError.toString(),
        errorName: pythonError.name,
        errorStack: pythonError.stack,
        stdout: stdout,
        stderr: stderr || pythonError.toString(),
        executionTime: Date.now() - startTime
      });
      return;
    }

    // Restore stdout/stderr and get captured output
    pyodide.runPython(`
sys.stdout = _original_stdout
sys.stderr = _original_stderr
    `);
    
    const stdout = pyodide.runPython('_captured_stdout.getvalue()');
    const stderr = pyodide.runPython('_captured_stderr.getvalue()');

    // Handle different result types
    // For plain text endpoint, always return string representation
    let processedResult;
    if (result === undefined) {
      processedResult = 'None';
    } else if (result === null) {
      processedResult = 'None';
    } else if (typeof result === 'bigint') {
      processedResult = result.toString();
    } else if (result && typeof result.toJs === 'function') {
      // PyProxy object - convert to JavaScript object but don't stringify
      try {
        processedResult = result.toJs();
      } catch {
        processedResult = result.toString();
      } finally {
        if (typeof result.destroy === 'function') {
          result.destroy();
        }
      }
    } else if (typeof result === 'object') {
      // Native JavaScript object - shouldn't happen from Python execution
      // But if it does, convert to string
      processedResult = JSON.stringify(result);
    } else {
      // Primitive types - convert to string
      processedResult = String(result);
    }

    // Send message and wait for it to be fully transmitted
    await sendMessage({
      type: 'result',
      executionId,
      success: true,
      result: processedResult,
      stdout: stdout || '',
      stderr: stderr || '',
      executionTime: Date.now() - startTime
    });

    // DETERMINISTIC: Clean up Python memory only after message is fully sent
    try {
      await lightCleanupPythonMemory();
    } catch (error) {
      // Cleanup errors shouldn't affect execution results
      console.warn('Memory cleanup warning:', error.message);
    }

  } catch (error) {
    // Send more detailed error information
    let errorDetails = error.message || error.toString();
    if (error.name) {
      errorDetails = `${error.name}: ${errorDetails}`;
    }
    
    // Send error message and wait for it to be fully transmitted
    await sendMessage({
      type: 'result',
      executionId,
      success: false,
      error: errorDetails,
      errorName: error.name,
      errorStack: error.stack,
      executionTime: Date.now() - startTime
    });

    // DETERMINISTIC: Clean up Python memory only after message is fully sent
    try {
      await lightCleanupPythonMemory();
    } catch (cleanupError) {
      // Cleanup errors shouldn't affect execution results
      console.warn('Memory cleanup warning:', cleanupError.message);
    }
  }
}

/**
 * Clean up Python memory after code execution
 * This is critical to prevent memory leaks when running many tests
 * and ensuring complete isolation between user executions
 */
async function cleanupPythonMemory() {
  try {
    // First, explicitly clear any context variables we tracked
    for (const varName of contextVariableNames) {
      try {
        pyodide.globals.delete(varName);
      } catch {
        // Variable might already be deleted
      }
    }
    contextVariableNames.clear(); // Reset the tracker
    
    await pyodide.runPythonAsync(`
import gc
import sys

# Clear all user-created global variables (keep only built-ins and system modules)
globals_to_delete = []
for name in list(globals().keys()):
    # Skip system variables and modules
    if not name.startswith('_') and name not in ['sys', 'os', 'gc', 'matplotlib', 'plt', 'Path', 'micropip', 'StringIO', 'sns', 'seaborn']:
        globals_to_delete.append(name)

# Delete tracked globals
for name in globals_to_delete:
    try:
        del globals()[name]
    except:
        pass

# Clear all module caches except core ones
modules_to_keep = {
    'sys', 'os', 'gc', 'builtins', '__main__', 
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
    'numpy', 'pandas', 'scipy', 'sklearn',
    'seaborn', 'statsmodels', 'statsmodels.api',
    'pathlib', 'io', 'json', 'datetime', 'time',
    'micropip', '_pyodide', 'pyodide_js'
}

modules_to_remove = []
for module_name in list(sys.modules.keys()):
    if module_name not in modules_to_keep and not module_name.startswith('_'):
        modules_to_remove.append(module_name)

for module_name in modules_to_remove:
    try:
        del sys.modules[module_name]
    except:
        pass

# Force garbage collection
gc.collect()
gc.collect()  # Run twice to ensure cleanup of circular references
gc.collect()  # Third time for deeply nested references

# Clear matplotlib figures if matplotlib is loaded
try:
    import matplotlib.pyplot as plt
    plt.close('all')  # Close all matplotlib figures
    # Clear the figure manager
    import matplotlib._pylab_helpers
    matplotlib._pylab_helpers.Gcf.destroy_all()
except:
    pass

# Clear any pandas dataframes from memory
try:
    import pandas as pd
    # Force pandas to release memory
    pd.DataFrame()._clear_item_cache()
except:
    pass

# Clear any open file handles
try:
    import io
    # Close any BufferedWriter/Reader objects that might be lingering
    for obj in gc.get_objects():
        if isinstance(obj, (io.BufferedWriter, io.BufferedReader, io.TextIOWrapper)):
            try:
                obj.close()
            except:
                pass
except:
    pass

# Reset working directory to default
try:
    os.chdir('/home/pyodide')
except:
    pass

# Clear any temporary variables from previous cleanup attempts
for var in ['_captured_stdout', '_captured_stderr', '_original_stdout', '_original_stderr']:
    if var in globals():
        try:
            del globals()[var]
        except:
            pass
    `);
  } catch (error) {
    // Log but don't fail on cleanup errors
    sendMessage({
      type: 'debug',
      message: 'Memory cleanup warning',
      error: error.message
    });
  }
}

/**
 * Light cleanup Python memory - minimal cleanup to avoid matplotlib interference
 * This is used instead of aggressive cleanup to preserve matplotlib objects
 */
async function lightCleanupPythonMemory() {
  try {
    // Only clear context variables we explicitly tracked
    for (const varName of contextVariableNames) {
      try {
        pyodide.globals.delete(varName);
      } catch {
        // Variable might already be deleted
      }
    }
    contextVariableNames.clear();
    
    // Only run minimal Python cleanup
    await pyodide.runPythonAsync(`
import gc

# Only clear obvious temporary variables that may have been created
temp_vars_to_clear = []
for name in list(globals().keys()):
    # Only clear variables that look like temporary/test variables
    if (name.startswith('temp_') or 
        name.startswith('test_') or
        name.startswith('_temp') or 
        name in ['x', 'y', 'data', 'df'] and 
        not name.startswith('_') and 
        name not in ['sys', 'os', 'gc', 'matplotlib', 'plt', 'np', 'pd', 'sns']):
        temp_vars_to_clear.append(name)

# Clear only obvious temporary variables
for name in temp_vars_to_clear:
    try:
        del globals()[name] 
    except:
        pass

# Single garbage collection (not aggressive)
gc.collect()
    `);
  } catch (error) {
    // Ignore cleanup errors in light mode
  }
}

/**
 * Install Python package via micropip
 * @param {string} packageName - Name of package to install
 * @param {string|null} executionId - Unique identifier for this operation
 */
async function installPackage(packageName, executionId = null) {
  if (!initialized) {
    sendMessage({
      type: 'package_result',
      executionId,
      success: false,
      error: 'Pyodide not initialized'
    });
    return;
  }

  const startTime = Date.now();
  
  try {
    // Use micropip.install directly with await at JavaScript level
    const micropip = pyodide.pyimport('micropip');
    await micropip.install(packageName);
    micropip.destroy(); // Clean up the PyProxy

    sendMessage({
      type: 'package_result',
      executionId,
      success: true,
      package: packageName,
      installTime: Date.now() - startTime
    });
  } catch (error) {
    sendMessage({
      type: 'package_result',
      executionId,
      success: false,
      package: packageName,
      error: error.message,
      installTime: Date.now() - startTime
    });
  }
}

/**
 * Process incoming messages from parent
 * @param {string} message - JSON message from parent process
 */
function processMessage(message) {
  try {
    const data = JSON.parse(message);
    
    switch (data.type) {
      case 'execute':
        executeCode(data.code, data.context, data.executionId);
        break;
      case 'install_package':
        installPackage(data.package, data.executionId);
        break;
      case 'cleanup':
        // Force memory cleanup
        cleanupPythonMemory().then(() => {
          sendMessage({ type: 'cleanup_complete', executionId: data.executionId });
        }).catch(error => {
          sendMessage({ type: 'cleanup_error', executionId: data.executionId, error: error.message });
        });
        break;
      case 'ping':
        sendMessage({ type: 'pong', executionId: data.executionId });
        break;
      case 'shutdown':
        process.exit(0);
        break;
      default:
        sendMessage({
          type: 'error',
          executionId: data.executionId,
          error: `Unknown message type: ${data.type}`
        });
    }
  } catch (error) {
    sendMessage({
      type: 'error',
      error: `Failed to process message: ${error.message}`
    });
  }
}

// Setup stdin processing
let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk.toString();
  
  // Process complete lines
  let lines = buffer.split('\n');
  buffer = lines.pop() || ''; // Keep incomplete line in buffer
  
  for (const line of lines) {
    if (line.trim()) {
      processMessage(line.trim());
    }
  }
});

process.stdin.on('end', () => {
  process.exit(0);
});

// Handle process termination
process.on('SIGTERM', () => {
  sendMessage({ type: 'terminated', reason: 'SIGTERM' });
  process.exit(0);
});

process.on('SIGINT', () => {
  sendMessage({ type: 'terminated', reason: 'SIGINT' });
  process.exit(0);
});

// Start initialization
initializePyodide().catch((error) => {
  console.error('Failed to initialize Pyodide executor:', error);
  process.exit(1);
});