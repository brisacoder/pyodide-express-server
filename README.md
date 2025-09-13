# Pyodide Express Server

A Node.js Express service that exposes a REST API for executing Python code via [Pyodide](https://pyodide.org/). Features comprehensive data science capabilities including matplotlib, seaborn, scikit-learn, and pandas with extensive testing coverage and **enhanced security logging**.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Direct Filesystem Access](#direct-filesystem-access)
- [Enhanced Features](#enhanced-features)
- [API Response Contract](#api-response-contract)

## Features

- **Python Code Execution** - Execute Python code via Pyodide and return stdout/stderr
- **Streaming Execution** - Real-time streaming endpoint for long-running operations
- **Enhanced Security Logging** - SHA-256 hashing, audit trails, and comprehensive monitoring
- **Statistics Dashboard** - Interactive Chart.js dashboard for security monitoring
- **Package Management** - Install additional Python packages at runtime via micropip
- **File Upload & Analysis** - Upload CSV and other data files for processing
- **Data Science Ready** - Built-in support for matplotlib, seaborn, scikit-learn, pandas
- **Health & Monitoring** - Comprehensive health checks and server statistics
- **Environment Management** - Reset Python environment, manage package state
- **Robust Testing** - Extensive test suite covering API, security, performance, and data science
- **Modular Architecture** - Clean, extensible design for adding new routes and features
- **Direct Filesystem Mounting** - Files created in Python appear directly in local filesystem (no API calls needed!)

## Quick Start

```bash
# Clone and install
git clone https://github.com/brisacoder/pyodide-express-server.git
cd pyodide-express-server
npm ci

# Create environment file (optional)
cp .env.example .env   # Edit PORT, NODE_ENV, etc. as needed

# Create required directories
mkdir -p logs uploads plots/matplotlib plots/seaborn

# Start the server
npm start
```

### 🌐 Web Interface

Once the server is running, open your browser and go to:

**http://localhost:3000**

The main UI provides:
- **Interactive Python Code Editor** with syntax highlighting and resizable text area
- **Real-time Python Code Execution** - run data science code directly in your browser
- **Package Management** - install Python packages on-demand
- **File Upload & Analysis** - upload CSV files and analyze them with pandas
- **Environment Controls** - reset Python environment when needed

This is the fastest way to start experimenting with Python data science in your browser!

#### 🐍 Python Code Execution: Implicit Return Values

**Important:** When you execute Python code via the Pyodide API, the server will automatically return the value of the last variable or expression that appears alone on a line (outside of a function or class definition). This acts as an implicit return statement, similar to how interactive Python shells work.

For example, the following code will return the value of `result`:

```python
x = 2
result = x * 5
result  # This value will be returned in the API response
```

You can also return values from functions by calling them as the last line:

```python
def add(a, b):
    return a + b
add(3, 4)  # Returns 7
```

**Note:** You do not need to use an explicit `return` statement at the top level. The last standalone variable or function call will be returned automatically in the API response under the `data.result` field.

#### 📊 Statistics Dashboard

The server includes a comprehensive monitoring dashboard accessible at:

**http://localhost:3000/api/dashboard/stats/dashboard**

Features:
- **Professional UI** with responsive design and gradient styling
- **Real-time metrics**: execution counts, success rates, timing analysis
- **Hourly trend analysis** with interactive charts
- **REST endpoints** for programmatic access to statistics

### 📝 Sample Clients

With the server running, try the sample clients:
```bash
node examples/basic-client.js           # Simple execution example
node examples/data-science-client.js    # Data science workflow with matplotlib/seaborn  
node examples/file-upload-client.js     # File upload and pandas analysis
node examples/execute-raw-client.js     # Raw endpoint for complex Python code
```

For testing and development, also set up the Python environment using uv:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Install Python dependencies using uv (creates virtual environment automatically)
uv sync
```

## 📁 Direct Filesystem Access

**NEW!** The server implements true filesystem mounting per [Pyodide documentation](https://pyodide.org/en/stable/usage/accessing-files.html). When Python code creates files in mounted directories, they appear **directly** in your local filesystem automatically!

```python
# This Python code creates files directly on your local machine:
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 2])
plt.savefig('/plots/matplotlib/my_chart.png')  # Appears instantly in local plots/ folder!
```

**📖 Complete Guide:** See [`docs/FILESYSTEM_MOUNTING_GUIDE.md`](docs/FILESYSTEM_MOUNTING_GUIDE.md) for comprehensive documentation  
**🚀 Quick Reference:** See [`docs/QUICK_REFERENCE_FILESYSTEM.md`](docs/QUICK_REFERENCE_FILESYSTEM.md) for examples

## Enhanced Features

### 🔐 Enhanced Security Logging System
- **SHA-256 code hashing** for security tracking and audit trails
- **Real-time statistics dashboard** with Chart.js visualizations
- **Dual logging** (server.log + security.log) for comprehensive monitoring
- **IP tracking, User-Agent monitoring, error categorization**
- **Interactive dashboard** at `/api/dashboard/stats/dashboard`
- **Backward-compatible APIs** maintaining existing functionality

## Project Structure

The repository includes these key components:

```
pyodide-express-server/
├── 📦 Core Server
│   ├── src/                     # Source code
│   │   ├── server.js            # Main server file
│   │   ├── app.js               # Express app configuration
│   │   ├── services/pyodide-service.js # Pyodide WebAssembly runtime
│   │   ├── controllers/         # Route controllers
│   │   ├── middleware/          # Express middleware
│   │   ├── routes/              # API route definitions
│   │   └── utils/
│   │       ├── logger.js        # 🔐 Enhanced security logging
│   │       ├── metrics.js       # Performance metrics
│   │       └── requestContext.js # Request tracking
│   ├── public/index.html        # Server landing page
│   └── package.json             # Dependencies and scripts
├── 🔐 Security & Monitoring
│   ├── logs/                    # Log files (auto-created)
│   │   ├── server.log           # General application logs
│   │   └── security.log         # Security event logs
│   └── Dashboard at /api/dashboard/stats/dashboard
├── 📊 Data & Files
│   ├── uploads/                 # File uploads
│   ├── plots/                   # Generated plots
│   │   ├── matplotlib/          # Direct matplotlib saves
│   │   └── seaborn/             # Direct seaborn saves
│   └── examples/                # Example client applications
├── 🧪 Testing Infrastructure
│   ├── tests/                   # Comprehensive Python test suite (30+ modules)
│   ├── run_simple_tests.py      # Quick development tests
│   └── run_comprehensive_tests.py # Full validation
└── 📚 Documentation
    ├── docs/                    # Detailed documentation
    ├── TESTING.md               # Testing guide
    └── pyodide_arch.md          # Pyodide integration specifics
```

## Environment Configuration

Create `.env` file for custom configuration (optional):

```bash
# Copy the template  
cp .env.example .env
```

Default configuration values:
```bash
# Server Configuration
PORT=3000
NODE_ENV=development

# Security Logging
SECURITY_LOG_ENABLED=true
SECURITY_LOG_FILE=logs/security.log

# Dashboard Configuration  
DASHBOARD_ENABLED=true
DASHBOARD_TITLE="Pyodide Security Dashboard"

# Performance Settings
REQUEST_TIMEOUT=30000
MAX_FILE_SIZE=10485760
```

## API Endpoints

### Core Execution
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/execute` | Execute Python code and return the output |
| POST | `/api/execute-raw` | Execute Python code with raw text body (no JSON wrapping) |   <--- RECOMMENDED
| POST | `/api/execute-stream` | Execute code and stream results |
| POST | `/api/install-package` | Install a Python package via `micropip` |
| GET  | `/api/packages` | List installed packages |
| POST | `/api/reset` | Reset the Python environment |

### Health & Monitoring
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET  | `/api/status` | Pyodide initialization status |
| GET  | `/api/health` | Python execution health check |
| GET  | `/api/stats` | Server statistics (legacy format) |
| GET  | `/health` | Overall server health |

### 🔐 Security & Analytics (New!)
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/api/dashboard/stats` | Real-time security statistics (JSON) |
| GET | `/api/dashboard/stats/dashboard` | Interactive Chart.js dashboard (HTML) |
| POST | `/api/dashboard/stats/clear` | Clear/reset security statistics |

### File Management
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/upload` | Upload a data file |
| GET | `/api/uploaded-files` | List uploaded files |
| GET | `/api/file-info/:filename` | Get file info for uploaded files |
| DELETE | `/api/uploaded-files/:filename` | Delete uploaded file |
| GET | `/api/pyodide-files` | List files in Pyodide virtual filesystem |
| DELETE | `/api/pyodide-files/:filename` | Delete file from Pyodide filesystem |
| POST | `/api/extract-plots` | Extract plots from Pyodide filesystem |

## API Response Contract

All API endpoints return responses in a standardized format for consistency and ease of integration. The contract is as follows:

```json
{
  "success": true | false,           // Indicates if the operation was successful
  "data": <object|null>,             // Main result data (object or null if error)
  "error": <string|null>,            // Error message (string or null if success)
  "meta": { "timestamp": <string> } // Metadata, always includes ISO timestamp
}
```

- On success, `success` is `true`, `data` contains the result, and `error` is `null`.
- On error, `success` is `false`, `data` is `null`, and `error` contains a descriptive message.
- `meta.timestamp` is always present and uses ISO 8601 format.

**Example: Successful Response**
```json
{
  "success": true,
  "data": { "result": "Package installed successfully" },
  "error": null,
  "meta": { "timestamp": "2025-08-21T19:30:42.972Z" }
}
```

**Example: Error Response**
```json
{
  "success": false,
  "data": null,
  "error": "Can't fetch metadata for 'nonexistent-package-xyz123'. Please make sure you have entered a correct package name and correctly specified index_urls (if you changed them).",
  "meta": { "timestamp": "2025-08-21T19:30:42.972Z" }
}
```

All endpoints (including `/api/execute`, `/api/install-package`, `/api/packages`, etc.) follow this contract.

## Development Workflow

### Node.js Development
- `npm run dev` – start the server with live reload
- `npm run lint` – lint source files
- `npm run format` – apply Prettier formatting
- `npm test` – run basic Node.js tests

### Python Testing Environment
The project includes extensive Python test suites. Set up the environment:

```bash
# Create and activate virtual environment (if not already done)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install test dependencies using uv
uv sync
```

### File Organization
- **Logs:** Live server logs under `logs/`
- **Uploads:** Test and user-uploaded files under `uploads/`
- **Tests:** Comprehensive Python test suite under `tests/`

See [docs/architecture.md](docs/architecture.md), [`pyodide_arch.md`](pyodide_arch.md), and [TESTING.md](TESTING.md) for deeper technical details.

## Testing

The project includes comprehensive Python test suites that validate all functionality including API endpoints, error handling, security, performance, and data science capabilities (matplotlib, seaborn, scikit-learn).

### Quick Testing (Development)
For fast validation during development:

```bash
# Start the server first
npm start

# In another terminal, run quick smoke tests
uv run python run_simple_tests.py
```

### Comprehensive Testing (CI/CD & Pre-commit)
For complete system validation with automatic server management:

```bash
# Run all test categories (recommended)
uv run python run_comprehensive_tests.py

# Run specific test categories
uv run python run_comprehensive_tests.py --categories basic integration security

# Run plotting and data science tests
uv run python run_comprehensive_tests.py --categories matplotlib seaborn sklearn
```

**Test Categories Available:**
- **Basic API** - Core endpoint functionality
- **Error Handling** - Error scenarios and edge cases  
- **Integration** - End-to-end workflows
- **Security** - Authentication and input validation
- **Security Logging** - Enhanced logging system and dashboard (New!)
- **Performance** - Load and stress testing
- **Reset** - Environment reset functionality
- **Extra Non-Happy Paths** - Additional edge cases
- **Scikit-Learn** - Machine learning functionality
- **Matplotlib** - Static plotting capabilities
- **Seaborn** - Statistical visualization
- **Dynamic Modules** - Package management and execution robustness

### Node.js Tests
Basic server functionality tests:

```bash
npm test
```

**Requirements:** The Python tests are managed with `uv` which automatically handles dependencies and virtual environments. See [TESTING.md](TESTING.md) for detailed testing documentation.

## Verification & Health Checks

### Basic Server Health
```bash
# Test server is running
curl http://localhost:3000/health

# Test Pyodide status  
curl http://localhost:3000/api/status

# View security statistics
curl http://localhost:3000/api/dashboard/stats
```

### Interactive Dashboard
Open the security monitoring dashboard in your browser:
```bash
# Windows
start http://localhost:3000/api/dashboard/stats/dashboard

# macOS
open http://localhost:3000/api/dashboard/stats/dashboard

# Linux
xdg-open http://localhost:3000/api/dashboard/stats/dashboard
```

### Test Python Code Execution
```bash
# Test basic execution
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello World!\"); result = 2 + 2; print(f\"Result: {result}\")"}'

# Test raw execution (better for complex code)
curl -X POST http://localhost:3000/api/execute-raw \
  -H "Content-Type: text/plain" \
  -d "import pandas as pd; print(pd.__version__)"
```

## Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check Node.js version (requires 18+)
node --version

# Clear npm cache and reinstall
npm cache clean --force
npm ci

# Check if port is in use
netstat -tulpn | grep :3000  # Linux/macOS
netstat -an | findstr :3000  # Windows
```

**Tests failing:**
```bash
# Ensure uv environment is ready
uv sync

# Check if server is running
curl http://localhost:3000/health

# Run tests with verbose output
uv run python run_simple_tests.py
```

**Python execution errors:**
```bash
# Check Pyodide status
curl http://localhost:3000/api/status

# Check server logs
tail -f logs/server.log

# Test package installation
curl -X POST http://localhost:3000/api/install-package \
  -H "Content-Type: application/json" \
  -d '{"package": "requests"}'
```

**Security logging not working:**
```bash
# Check logs directory exists
ls -la logs/

# Verify security logging is enabled
curl http://localhost:3000/api/dashboard/stats

# Check security log file
tail -f logs/security.log
```

**Dashboard not loading:**
```bash
# Test JSON endpoint first
curl http://localhost:3000/api/dashboard/stats

# Check browser console for JavaScript errors
# Verify Chart.js CDN is accessible
```

### Performance Issues
- **Memory usage**: Monitor with `top` or Task Manager during heavy usage
- **Response times**: Check dashboard performance metrics
- **Log file size**: Clean logs regularly with `npm run clean:logs`
- **File uploads**: Ensure `uploads/` directory has proper permissions

### Getting Help
- **Issue Tracker**: Create GitHub issue with logs and reproduction steps
- **Documentation**: Check all documentation files in `docs/` directory  
- **Test Results**: Include output from `uv run python run_comprehensive_tests.py`

## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is released under the [MIT License](LICENSE).

## Additional Resources

- [Examples](examples/README.md) – sample clients for interacting with the API
- [Testing Guide](TESTING.md) – comprehensive testing documentation and best practices
- [Architecture Overview](docs/architecture.md) – detailed system design walkthrough
- [Pyodide Architecture](pyodide_arch.md) – Pyodide integration specifics
