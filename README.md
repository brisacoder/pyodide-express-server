# Pyodide Express Server

A Node.js Express service that exposes a REST API for executing Python code via [Pyodide](https://pyodide.org/). Features comprehensive data science capabilities including matplotlib, seaborn, scikit-learn, and pandas with extensive testing coverage.

## Features

- **Python Code Execution** - Execute Python code via Pyodide and return stdout/stderr
- **Streaming Execution** - Real-time streaming endpoint for long-running operations
- **Package Management** - Install additional Python packages at runtime via micropip
- **File Upload & Analysis** - Upload CSV and other data files for processing
- **Data Science Ready** - Built-in support for matplotlib, seaborn, scikit-learn, pandas
- **Health & Monitoring** - Comprehensive health checks and server statistics
- **Environment Management** - Reset Python environment, manage package state
- **Robust Testing** - Extensive test suite covering API, security, performance, and data science
- **Modular Architecture** - Clean, extensible design for adding new routes and features
- **Direct Filesystem Mounting** - Files created in Python appear directly in local filesystem (no API calls needed!)

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

## Quick Start

```bash
# Clone and install
git clone https://github.com/brisacoder/pyodide-express-server.git
cd pyodide-express-server
npm ci
cp .env.example .env   # optional

# Start the server
npm start
```

With the server running, try the sample client:
```bash
node examples/basic-client.js
```

For testing and development, also set up the Python environment:
```bash
# Create virtual environment and install test dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
```

## API Endpoints
| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/execute` | Execute Python code and return the output |
| POST | `/api/execute-stream` | Execute code and stream results |
| POST | `/api/install-package` | Install a Python package via `micropip` |
| GET  | `/api/packages` | List installed packages |
| GET  | `/api/status` | Pyodide initialization status |
| GET  | `/api/health` | Python execution health check |
| GET  | `/api/stats` | Server statistics |
| POST | `/api/reset` | Reset the Python environment |
| POST | `/api/upload-csv` | Upload a data file |
| GET  | `/health` | Overall server health |

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

# Install test dependencies
pip install -r requirements.txt  # or use uv for faster installs
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
python run_simple_tests.py
```

### Comprehensive Testing (CI/CD & Pre-commit)
For complete system validation with automatic server management:

```bash
# Run all test categories (recommended)
python run_comprehensive_tests.py

# Run specific test categories
python run_comprehensive_tests.py --categories basic integration security

# Run plotting and data science tests
python run_comprehensive_tests.py --categories matplotlib seaborn sklearn
```

**Test Categories Available:**
- **Basic API** - Core endpoint functionality
- **Error Handling** - Error scenarios and edge cases  
- **Integration** - End-to-end workflows
- **Security** - Authentication and input validation
- **Performance** - Load and stress testing
- **Reset** - Environment reset functionality
- **Extra Non-Happy Paths** - Additional edge cases
- **Scikit-Learn** - Machine learning functionality
- **Matplotlib** - Static plotting capabilities
- **Seaborn** - Statistical visualization

### Node.js Tests
Basic server functionality tests:

```bash
npm test
```

**Requirements:** The Python tests require a virtual environment with `requests` and other dependencies. See [TESTING.md](TESTING.md) for detailed testing documentation.

## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is released under the [MIT License](LICENSE).

## Additional Resources

- [Examples](examples/README.md) – sample clients for interacting with the API
- [Testing Guide](TESTING.md) – comprehensive testing documentation and best practices
- [Architecture Overview](docs/architecture.md) – detailed system design walkthrough
- [Pyodide Architecture](pyodide_arch.md) – Pyodide integration specifics
