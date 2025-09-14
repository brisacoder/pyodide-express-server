# Process Pool Configuration Guide

## Current Process Pool Settings

The Pyodide Express Server uses a **process pool** with **persistent processes** to avoid the overhead of initializing Pyodide for every execution.

### Current Configuration:
- **Pool Size**: 2 processes (configurable)
- **Max Executions per Process**: 50 executions before recycling
- **Initialization Timeout**: 90 seconds
- **Execution Timeout**: 30 seconds (default)

## How to Configure Process Pool Size

### Method 1: Environment Variables (Recommended)
Create a `.env` file in the project root or set environment variables:

```bash
# Process pool configuration
PYODIDE_POOL_SIZE=4                    # Number of persistent processes (default: 2)
PYODIDE_MAX_EXECUTIONS=100             # Executions before process recycling (default: 50)
PYODIDE_INIT_TIMEOUT=120000            # Process initialization timeout in ms (default: 90000)
PYODIDE_IDLE_TIMEOUT=300000            # Process idle timeout in ms (default: 300000)
PYODIDE_HEALTH_CHECK_INTERVAL=30000    # Health check interval in ms (default: 30000)
```

### Method 2: Direct Configuration File Edit
Edit `src/config/constants.js` and modify the `PROCESS_POOL` constants:

```javascript
const PROCESS_POOL = {
  DEFAULT_POOL_SIZE: 4,              // Change from 2 to 4 processes
  MAX_EXECUTIONS_PER_PROCESS: 100,   // Increase executions before recycling
  PROCESS_INIT_TIMEOUT: 120000,      // 2 minutes for initialization
  // ... other settings
};
```

## Pool Size Recommendations

### System Resource Guidelines:
- **2 processes**: Good for development, light usage (default)
- **4 processes**: Recommended for moderate concurrent usage
- **6-8 processes**: High-concurrency production environments
- **1 process**: Minimal resource usage, no concurrency

### Memory Considerations:
Each Pyodide process uses approximately **100-150MB RAM** when loaded with packages like NumPy, Pandas, etc.

- **2 processes**: ~300MB RAM
- **4 processes**: ~600MB RAM  
- **8 processes**: ~1.2GB RAM

### CPU Considerations:
- More processes = better concurrency for CPU-bound Python code
- Each process can handle one execution at a time
- Pool size should not exceed your CPU core count

## Testing Different Pool Sizes

### Quick Test Commands:
```bash
# Test with 4 processes
PYODIDE_POOL_SIZE=4 npm start

# Test with 1 process (minimal)
PYODIDE_POOL_SIZE=1 npm start

# Test with 6 processes (high concurrency)
PYODIDE_POOL_SIZE=6 npm start
```

### Monitoring Pool Performance:
Check the server logs for pool statistics:
```bash
# Look for pool initialization logs
grep "Pyodide process pool initialized" logs/server.log

# Check pool stats via API
curl http://localhost:3000/api/status
```

## Process Pool Behavior

### Process Lifecycle:
1. **Initialization**: Pool creates N processes during server startup (~7 seconds each)
2. **Execution**: Processes handle requests in round-robin fashion
3. **Recycling**: After 50 executions (configurable), process is killed and replaced
4. **Crash Protection**: If a process crashes, it's automatically replaced

### Benefits of Process Pool:
- **⚡ Performance**: 3-6ms execution time (vs 60+ seconds without pool)
- **🛡️ Isolation**: Infinite loops or crashes only affect one process
- **🔄 Concurrency**: Multiple executions can run simultaneously
- **♻️ Resource Management**: Automatic process recycling prevents memory leaks

## Current Status

You can check the current pool configuration and status with:

```bash
# Via API
curl http://localhost:3000/api/status

# Via server logs
tail -f logs/server.log | grep "process-pool"
```

The response will show:
- Pool size
- Active processes
- Available processes  
- Total executions
- Process statistics