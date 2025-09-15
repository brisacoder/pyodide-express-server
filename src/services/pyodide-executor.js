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

/**
 * Initialize Pyodide with required packages and filesystem setup
 */
async function initializePyodide() {
  try {
    // Load Pyodide with locally available packages
    pyodide = await loadPyodide({
      packages: ['micropip', 'numpy', 'pandas', 'matplotlib', 'scikit-learn']
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

    mkdirSafe('/home');
    mkdirSafe('/home/pyodide');
    mkdirSafe('/home/pyodide/uploads');
    mkdirSafe('/home/pyodide/plots');
    mkdirSafe('/home/pyodide/plots/matplotlib');
    mkdirSafe('/home/pyodide/plots/seaborn');
    mkdirSafe('/home/pyodide/plots/base64');

    // Mount host filesystem directories from pyodide_data
    const projectRoot = path.resolve(__dirname, '../..');
    const pyodideDataDir = path.join(projectRoot, 'pyodide_data');
    const uploadsPath = path.join(pyodideDataDir, 'uploads');
    const plotsPath = path.join(pyodideDataDir, 'plots');
    
    try {
      pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: uploadsPath }, '/home/pyodide/uploads');
      pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: plotsPath }, '/home/pyodide/plots');
    } catch (err) {
      console.warn('Failed to mount filesystem directories:', err.message);
      // Fallback to virtual filesystem if mounting fails
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
    // Set up context variables
    for (const [key, value] of Object.entries(context)) {
      pyodide.globals.set(key, value);
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
      // Restore stdout/stderr
      pyodide.runPython(`
sys.stdout = _original_stdout
sys.stderr = _original_stderr
      `);
      
      const stdout = pyodide.runPython('_captured_stdout.getvalue()');
      const stderr = pyodide.runPython('_captured_stderr.getvalue()');
      
      sendMessage({
        type: 'result',
        executionId,
        success: false,
        error: pythonError.message || pythonError.toString(),
        stdout: stdout || '',
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
    let serializedResult;
    if (result === undefined || result === null) {
      serializedResult = result;
    } else if (typeof result === 'bigint') {
      serializedResult = result.toString() + 'n';
    } else if (result && typeof result.toJs === 'function') {
      // PyProxy object
      try {
        serializedResult = result.toJs();
      } catch {
        serializedResult = result.toString();
      } finally {
        if (typeof result.destroy === 'function') {
          result.destroy();
        }
      }
    } else {
      serializedResult = result;
    }

    sendMessage({
      type: 'result',
      executionId,
      success: true,
      result: serializedResult,
      stdout: stdout || '',
      stderr: stderr || '',
      executionTime: Date.now() - startTime
    });

  } catch (error) {
    sendMessage({
      type: 'result',
      executionId,
      success: false,
      error: error.message,
      executionTime: Date.now() - startTime
    });
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
 * Send message to parent process
 * @param {Object} data - Data to send as JSON message
 */
function sendMessage(data) {
  try {
    process.stdout.write(JSON.stringify(data) + '\n');
  } catch (error) {
    console.error('Failed to send message:', error);
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