# 🚀 Complete Setup Instructions for Pyodide Express Server

Follow these steps to get your Pyodide Express Server with **Enhanced Security Logging** up and running.

## 🎯 Quick Start (Fastest Way)

If you're using the existing repository:

```bash
# Clone the repository
git clone https://github.com/brisacoder/pyodide-express-server.git
cd pyodide-express-server

# Install Node.js dependencies
npm ci

# Set up Python testing environment
python -m venv .venv
.venv\Scripts\activate  # Windows (.venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# Start the server
npm start

# In another terminal, test everything works
python run_simple_tests.py
```

## 📁 Project Structure (Current)

The repository includes these key components:

```
pyodide-express-server/
├── 📦 Core Server
│   ├── src/                     # Source code
│   │   ├── server.js            # Main server file
│   │   ├── app.js               # Express app configuration
│   │   ├── services/
│   │   │   └── pyodide-service.js # Pyodide service
│   │   ├── controllers/         # Route controllers
│   │   ├── middleware/          # Express middleware
│   │   ├── routes/              # API route definitions
│   │   └── utils/
│   │       ├── logger.js        # 🔐 Enhanced security logging
│   │       ├── metrics.js       # Performance metrics
│   │       └── requestContext.js # Request tracking
│   ├── public/                  # Static files
│   │   └── index.html           # Server landing page
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
│   └── Filesystem mounting enabled
├── 🧪 Testing Infrastructure
│   ├── tests/                   # Comprehensive test suite
│   │   ├── test_api.py          # Basic API tests
│   │   ├── test_security_logging.py # 🔐 Security logging tests
│   │   ├── test_matplotlib.py   # Visualization tests
│   │   └── ... (20+ test modules)
│   ├── run_simple_tests.py      # Quick development tests
│   └── run_comprehensive_tests.py # Full validation
└── 📚 Documentation
    ├── docs/                    # Detailed documentation
    ├── README.md                # Main documentation
    ├── TESTING.md               # Testing guide
    ├── TODO.md                  # 🆕 Development roadmap
    └── setup.md                 # This file
```

## 🔐 New Security Features

### Enhanced Security Logging System
- **SHA-256 code hashing** for audit trails
- **Real-time statistics collection** with IP and User-Agent tracking
- **Interactive Chart.js dashboard** at `/api/dashboard/stats/dashboard`
- **Dual logging streams** (server.log + security.log)
- **Comprehensive test coverage** (10 security logging tests)

### Dashboard Features
- **Professional UI** with responsive design
- **Real-time metrics**: execution counts, success rates, timing analysis
- **Hourly trend tracking** with interactive charts
- **REST API access** for programmatic monitoring

## �️ Manual Setup (If Building from Scratch)

### 1. Initialize Project
```bash
mkdir pyodide-express-server
cd pyodide-express-server
npm init -y
```

### 2. Install Dependencies
```bash
# Core dependencies
npm install express multer pyodide cors helmet compression

# Development dependencies  
npm install --save-dev nodemon prettier eslint

# Security and monitoring
npm install crypto fs path
```

### 3. Create Directory Structure
```bash
# Create all required directories
mkdir -p src/{controllers,middleware,routes,services,utils,config}
mkdir -p public docs tests logs uploads plots/{matplotlib,seaborn}
mkdir -p examples
```

### 4. Copy Source Files
You'll need to create all the source files. The key files with enhanced security logging:

**Enhanced Files:**
- `src/utils/logger.js` - Security logging with SHA-256 hashing
- `src/routes/stats.js` - Dashboard endpoints
- `src/routes/execute.js` - Enhanced execution logging
- `tests/test_security_logging.py` - Security test suite

**Core Files:**
- `src/server.js` - Main server
- `src/app.js` - Express configuration  
- `src/services/pyodide-service.js` - Pyodide integration
- `src/swagger-config.js` - API documentation

## 🧪 Testing Setup

### Python Environment (Required for Tests)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install test dependencies
pip install requests pytest unittest-xml-reporting

# Or use UV for faster installs:
pip install uv
uv pip install requests pytest
```

### Test Validation
```bash
# Start the server (keep running)
npm start

# In another terminal, run quick tests (39 tests)
python run_simple_tests.py

# Run comprehensive tests (100+ tests)
python run_comprehensive_tests.py

# Run just security logging tests
python -m unittest tests.test_security_logging -v
```

## 🌍 Environment Configuration

Create `.env` file for custom configuration (optional):

```bash
# Copy the template  
cp .env.example .env
```

Default `.env` contents:
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

## 🚀 Starting the Server

### Development Mode (Recommended)
```bash
# Start with auto-reload
npm run dev

# Start with debugging
npm run dev:inspect
```

### Production Mode
```bash
# Standard start
npm start

# With PM2 (process manager)
npm install -g pm2
pm2 start src/server.js --name pyodide-server
```

## 🔍 Verification Steps

### 1. Basic Server Health
```bash
# Test server is running
curl http://localhost:3000/health

# Test Pyodide status  
curl http://localhost:3000/api/status
```

### 2. Security Dashboard Access
```bash
# View security statistics (JSON)
curl http://localhost:3000/api/dashboard/stats

# Open interactive dashboard in browser
start http://localhost:3000/api/dashboard/stats/dashboard
```

### 3. Execute Python Code
```bash
# Test basic execution
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello World!\"); result = 2 + 2; print(f\"Result: {result}\")"}'
```

### 4. Run Test Suite
```bash
# Quick tests (39 tests, ~1 second)
python run_simple_tests.py

# Full validation (100+ tests, ~60 seconds)  
python run_comprehensive_tests.py
```

## 🔐 Security Features Verification

### Check Security Logging
```bash
# View security log entries
tail -f logs/security.log

# Check statistics accumulation
curl http://localhost:3000/api/dashboard/stats | jq '.stats.overview'
```

### Test Dashboard Features
1. **Execute some Python code** to generate security events
2. **Open dashboard**: `http://localhost:3000/api/dashboard/stats/dashboard`
3. **Verify charts display**: execution counts, success rates, trends
4. **Test statistics reset**: `curl -X DELETE http://localhost:3000/api/dashboard/stats/clear`

## 📚 Next Steps

### Explore the API
- **Swagger Documentation**: `http://localhost:3000/api-docs`
- **API Examples**: See `docs/curl-commands.md`
- **Test Scripts**: Try `examples/basic-client.js`

### Development Workflow
- **Quick Testing**: Use `run_simple_tests.py` during development
- **Pre-commit**: Run `run_comprehensive_tests.py` before commits  
- **Security Focus**: Use `--categories security security_logging` for security testing

### Documentation
- **Architecture**: Read `docs/architecture.md`
- **Testing Guide**: See `TESTING.md`
- **Development Roadmap**: Check `TODO.md`
- **Filesystem Guide**: See `docs/FILESYSTEM_MOUNTING_GUIDE.md`

## 🆘 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check Node.js version (requires 18+)
node --version

# Clear npm cache  
npm cache clean --force
npm ci
```

**Tests failing:**
```bash
# Ensure virtual environment is active
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall test dependencies
pip install -r requirements.txt
```

**Security logging not working:**
```bash
# Check logs directory permissions
ls -la logs/

# Verify security logging is enabled
curl http://localhost:3000/api/dashboard/stats
```

**Dashboard not loading:**
```bash
# Check browser console for errors
# Verify Chart.js CDN is accessible
# Test with: curl http://localhost:3000/api/dashboard/stats/dashboard
```

### Performance Issues
- **Memory usage**: Monitor with `top` or Task Manager
- **Response times**: Check dashboard performance metrics
- **Log file size**: Rotate logs regularly (see `npm run clean:logs`)

### Getting Help
- **Issue Tracker**: Create GitHub issue with logs and steps to reproduce
- **Documentation**: Check all `.md` files in the repository
- **Test Results**: Include `run_comprehensive_tests.py` output for debugging

---

🎉 **Congratulations!** You now have a fully functional Pyodide Express Server with enhanced security logging, interactive dashboard, and comprehensive testing. The server provides a robust foundation for executing Python code via REST API with enterprise-grade monitoring and audit capabilities.
nano .env
```

Default `.env` content:
```
PORT=3000
NODE_ENV=development
LOG_LEVEL=info
```

## 🚀 7. Start the Server

```bash
# Start the server
npm start

# Or for development with auto-restart
npm run dev
```

You should see output like:
```
[2025-01-01T12:00:00.000Z] INFO: Starting Pyodide Express Server...
[2025-01-01T12:00:00.000Z] INFO: Initializing Pyodide...
[2025-01-01T12:00:05.000Z] INFO: Loading common packages...
[2025-01-01T12:00:10.000Z] INFO: Setting up Python environment...
[2025-01-01T12:00:15.000Z] INFO: Pyodide initialization completed!
[2025-01-01T12:00:15.000Z] INFO: 🚀 Server running on port 3000
[2025-01-01T12:00:15.000Z] INFO: 📖 Web interface: http://localhost:3000
[2025-01-01T12:00:15.000Z] INFO: 🔧 API base URL: http://localhost:3000/api
[2025-01-01T12:00:15.000Z] INFO: 📊 Health check: http://localhost:3000/health
```

## ✅ 8. Test the Installation

### Option A: Web Interface
1. Open your browser
2. Go to `http://localhost:3000`
3. You should see a web interface with code editor
4. Try running the sample Python code

### Option B: Test Client
```bash
# In a new terminal window
npm test

# Or run directly
node test-client.js
```

This will run comprehensive tests and show results like:
```
🚀 Starting Pyodide Express Server API Tests
============================================================
🔍 Waiting for server to be ready...
✅ Server is ready!

🧪 Testing Health Endpoint
==================================================
✅ Health check successful

🧪 Testing Basic Python Execution
==================================================
✅ Execution successful

📊 Test Summary
============================================================
✅ server ready
✅ health check
✅ basic execution
✅ async execution
✅ csv processing
✅ package installation
✅ package listing
✅ error handling
============================================================
🎯 Results: 8/8 tests passed
🎉 All tests passed! Your Pyodide Express Server is working correctly.
```

## 🌐 9. Using the API

### Execute Python Code:
```javascript
fetch('http://localhost:3000/api/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: `
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(df)
df.sum().to_dict()
    `
  })
})
```

### Install Packages:
```javascript
fetch('http://localhost:3000/api/install-package', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ package: 'requests' })
})
```

## 🔧 10. Troubleshooting

### Common Issues:

**"Cannot find module 'pyodide'"**
```bash
npm install pyodide
```

**"Port 3000 already in use"**
```bash
# Change port in .env file
echo "PORT=3001" >> .env
```

**"Pyodide initialization failed"**
- Check internet connection (Pyodide downloads from CDN)
- Try restarting the server
- Check firewall settings

**Tests fail**
```bash
# Make sure server is running first
npm start

# Then in another terminal
npm test
```

## 📋 11. Verification Checklist

- [ ] All dependencies installed (`npm install` successful)
- [ ] All source files created with correct code
- [ ] Server starts without errors
- [ ] Web interface loads at `http://localhost:3000`
- [ ] Test client passes all tests
- [ ] Can execute Python code via API
- [ ] Can install packages
- [ ] Error handling works

## 🎉 12. Next Steps

Once everything is working:

1. **Customize**: Modify the code for your specific use case
2. **Deploy**: Use Docker or cloud platforms for production
3. **Secure**: Add authentication and rate limiting for production
4. **Monitor**: Set up logging and monitoring
5. **Scale**: Add load balancing for high traffic

## 📞 Need Help?

If you encounter any issues:

1. Check the server logs for error messages
2. Verify all files are created correctly
3. Make sure all dependencies are installed
4. Test with the provided test client
5. Check the browser console for client-side errors

The setup should take about 5-10 minutes once you have all the files in place!