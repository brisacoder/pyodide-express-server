# 🚀 Complete Setup Instructions for Pyodide Express Server

Follow these steps to get your Pyodide Express Server up and running.

## 📁 1. Directory Structure

Create this exact folder structure:

```
pyodide-express-server/
├── package.json                 # Dependencies and scripts
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── test-client.js               # Test client
├── README.md                    # Documentation
├── src/                         # Source code
│   ├── server.js                # Main server file
│   ├── services/
│   │   └── pyodide-service.js   # Pyodide service
│   ├── middleware/
│   │   └── validation.js        # Input validation
│   └── utils/
│       └── logger.js            # Logging utility
├── uploads/                     # File uploads (will be created)
│   └── .gitkeep                 # Keep empty directory
└── logs/                        # Log files (will be created)
    └── .gitkeep                 # Keep empty directory
```

## 📦 2. Create package.json

Save this as `package.json`:

```json
{
  "name": "pyodide-express-server",
  "version": "1.0.0",
  "description": "Express server for executing Python code using Pyodide",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "node test-client.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5-lts.1",
    "pyodide": "^0.28.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  },
  "keywords": ["pyodide", "python", "express", "api"],
  "author": "Your Name",
  "license": "MIT"
}
```

## 🔧 3. Install Dependencies

```bash
# Navigate to your project directory
cd pyodide-express-server

# Install all dependencies
npm install
```

## 📂 4. Create Required Directories

```bash
# Create directories that might not exist
mkdir -p src/services src/middleware src/utils uploads logs

# Create .gitkeep files to track empty directories
touch uploads/.gitkeep logs/.gitkeep
```

## 📄 5. Copy All Source Files

You now need to create these files with the code I provided:

### Required Files:
1. **`src/services/pyodide-service.js`** - Main Pyodide service
2. **`src/utils/logger.js`** - Logging utility  
3. **`src/server.js`** - Main server file
4. **`src/middleware/validation.js`** - Input validation
5. **`test-client.js`** - Test client (in root directory)

Copy the complete code from the artifacts I created above into each file.

## 🌍 6. Environment Setup (Optional)

Create `.env` file for custom configuration:

```bash
# Copy the template
cp .env.example .env

# Edit if needed (default values work fine)
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