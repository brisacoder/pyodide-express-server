# Architecture Overview

This document provides a narrative tour of the Pyodide Express Server with its enhanced security logging system.
It is meant to complement the detailed notes in `pyodide_arch.md`.

## System Architecture

### Core Request Flow
1. **Express server** (`src/server.js`) boots and initializes the `PyodideService`.
2. **PyodideService** (`src/services/pyodide-service.js`) loads the WebAssembly
   runtime and a few common Python packages.
3. **Security Logger** (`src/utils/logger.js`) initializes enhanced logging with crypto hashing
4. Incoming API requests are routed to small handlers in `src/routes` which call
   into the service to execute Python code, install additional packages, or
   reset the runtime.
5. **Security events** are logged with SHA-256 hashing, IP tracking, and timing analysis
6. Results, stdout, and any errors are captured and returned as JSON.
7. **Statistics** are collected in real-time for dashboard visualization

### Enhanced Security Logging System

#### Core Components
- **Logger Service** (`src/utils/logger.js`):
  - SHA-256 code hashing for audit trails
  - Dual logging (server.log + security.log)
  - Real-time statistics collection
  - IP and User-Agent tracking

- **Statistics Dashboard** (`src/routes/stats.js`):
  - Chart.js visualizations with responsive design
  - Real-time metrics API endpoints
  - Interactive HTML dashboard
  - Statistics reset functionality

#### Security Features
- **Code Execution Tracking**: Every Python execution is logged with crypto hash
- **Error Categorization**: Detailed error analysis and trends
- **Performance Monitoring**: Execution time tracking and averaging
- **User Activity**: IP addresses and User-Agent strings logged
- **Hourly Trends**: Time-based analytics for usage patterns

### API Architecture

#### Route Organization
```
src/routes/
├── execute.js          # Python code execution (enhanced with security logging)
├── executeRaw.js       # Raw execution endpoint
├── health.js           # Health monitoring
├── upload.js           # File upload handling
├── files.js            # File management
└── stats.js            # NEW: Security dashboard endpoints
```

#### Enhanced Endpoints
- **Legacy Compatibility**: `/api/stats` maintains original format
- **Enhanced Statistics**: `/api/dashboard/stats` provides detailed security metrics
- **Interactive Dashboard**: `/api/dashboard/stats/dashboard` serves Chart.js visualization
- **Statistics Management**: `/api/dashboard/stats/clear` for resetting metrics

## Why Pyodide?
Pyodide bundles the CPython interpreter for the browser and WebAssembly
environments. Using it on the server allows us to execute Python code in a
sandbox without requiring a system Python installation.

**Benefits:**
- **Sandboxed Execution**: Python code runs in isolated WebAssembly environment
- **No System Dependencies**: No need for system Python installation
- **Package Ecosystem**: Access to most of the Python ecosystem via micropip
- **Security**: Enhanced logging provides audit trails and monitoring

## Security & Monitoring

### Enhanced Logging Architecture
The security logging system provides comprehensive monitoring without disrupting existing functionality:

- **Dual Logging Streams**:
  - `logs/server.log`: General application logging
  - `logs/security.log`: Security-specific events with crypto hashing

- **Real-time Statistics**:
  - Execution counts and success rates
  - Error categorization and trends
  - IP address and User-Agent tracking
  - Hourly execution patterns

- **Dashboard Visualization**:
  - Professional Chart.js interface
  - Responsive design for all devices
  - Real-time updates with live data

### Backward Compatibility
The security enhancements maintain 100% backward compatibility:
- Existing `/api/stats` endpoint unchanged
- No breaking changes to any existing APIs
- Enhanced data available via new endpoints
- Legacy clients continue to work without modification

## Extensibility
- Additional endpoints can be added under `src/routes/`.
- The service exposes hooks for package management and environment resets.
- **Security logging** can be extended with additional metrics and visualizations.
- Because everything runs inside one runtime instance, complex operations such
  as streaming output or loading data files become easy to expose via HTTP.
- **Dashboard components** can be customized or extended with additional Chart.js visualizations.

## Testing Architecture

### Comprehensive Test Coverage
The project includes extensive testing across multiple dimensions:

**Core Functionality** (39 quick tests):
- Basic API operations (16 tests)
- Dynamic module management (13 tests)
- Security logging system (10 tests)

**Extended Testing** (100+ comprehensive tests):
- Error handling and edge cases
- Performance and load testing
- Data science libraries (matplotlib, seaborn, scikit-learn)
- File system mounting and management
- Security and input validation

**Test Infrastructure**:
- **Simple Runner**: Fast development feedback (~1 second)
- **Comprehensive Runner**: Complete validation (~30-60 seconds)
- **Automatic Server Management**: No manual setup required
- **Selective Testing**: Run specific categories for focused debugging

For a line-by-line explanation of the initialization and execution process, see
[`pyodide_arch.md`](../pyodide_arch.md).
