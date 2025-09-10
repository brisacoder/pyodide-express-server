const express = require('express');
const path = require('path');
const multer = require('multer');
const helmet = require('helmet');
const logger = require('./utils/logger');
const config = require('./config');
const { swaggerSpec, swaggerUi, swaggerUiOptions } = require('./swagger-config');
const { requestContextMiddleware } = require('./utils/requestContext');
const { metricsMiddleware, metricsEndpoint } = require('./utils/metrics');
const executeRoutes = require('./routes/execute');
const fileRoutes = require('./routes/files');
const healthRoutes = require('./routes/health');
const executeRawRoutes = require('./routes/executeRaw');
const uploadRoutes = require('./routes/upload');
const statsRoutes = require('./routes/stats');
const DashboardRoutes = require('./routes/dashboard');
const crashReportsRoutes = require('./routes/crashReports');
const app = express();
// **SECURITY HEADERS - Express Team Standard (Helmet)**
// Used by major companies for production security
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ['\'self\''],
        scriptSrc: ['\'self\'', '\'unsafe-inline\'', '\'unsafe-eval\'', '\'unsafe-hashes\''], // Needed for Pyodide and inline handlers
        scriptSrcAttr: [
          '\'unsafe-hashes\'',
          '\'sha256-2i9H2Cj8BRV6t0xqGQegU4EjkwbamSb0ZtsxE7eusWQ=\'', // executeCode()
          '\'sha256-IniV1l/8/fsy5HPmpXv6FT3loH6FFa8uafatyr8GiTQ=\'', // clearResult()
          '\'sha256-fzSa9vFIu9VZ1v3EN7IgyRsJXButA9lbx/ymdE1Dqjw=\'', // resetEnvironment()
          '\'sha256-eN76d5DfNbHADUbeBBvz9DuFVYxk8n9KZWcj8oT4fWc=\'', // uploadFile()
          '\'sha256-vh+zIoc+jRVxRIPQ8oEvFkClGxC+Dg9pvkqjRGtchcY=\'', // listFiles()
          '\'sha256-zd7kcAQAM99C/MFOcwJLH3QigofTQXXnG3XdrIA6QO4=\'', // installPackage()
          '\'sha256-FAyvdkpvLytMGwms2wOhvqsSC7cS8QrItk6goGv2/sQ=\'', // listPackages()
          '\'sha256-a9sZvzEVqGwXEGxOEN97pTFyQ8YWO2SWWyzKQ2GtzlQ=\'', // clearAllFiles()
        ], // Allow inline event handlers with specific hashes
        workerSrc: ['\'self\'', 'blob:'], // Needed for WebAssembly workers
        childSrc: ['\'self\'', 'blob:'], // Needed for WebAssembly
        connectSrc: ['\'self\'', 'https://cdn.jsdelivr.net'], // Pyodide CDN
        styleSrc: ['\'self\'', '\'unsafe-inline\''], // For Swagger UI and inline styles
        imgSrc: ['\'self\'', 'data:', 'https:'], // Allow images from self, data URLs, and HTTPS
        fontSrc: ['\'self\'', 'https:', 'data:'], // Allow fonts from various sources
      },
    },
    crossOriginEmbedderPolicy: false, // Disabled for WebAssembly compatibility
  })
);
app.use(requestContextMiddleware);
app.use(metricsMiddleware);
// JSON parsing with error handling
app.use((req, res, next) => {
  express.json({ limit: '30mb' })(req, res, (err) => {
    if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
      return res.status(400).json({
        success: false,
        error: 'Invalid JSON format',
        timestamp: new Date().toISOString(),
      });
    }
    next(err);
  });
});
app.use(express.urlencoded({ extended: true, limit: '30mb' }));
// Serve static files from public directory
app.use(express.static(path.join(__dirname, '../public')));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', config.corsOrigin);
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header(
    'Access-Control-Allow-Headers',
    'Origin, X-Requested-With, Content-Type, Accept, Authorization'
  );
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});
app.use((req, res, next) => {
  logger.info('Incoming request', {
    method: req.method,
    path: req.path,
    originalUrl: req.originalUrl,
    ip: req.ip,
  });
  // Check for path traversal attempts in the URL
  if (req.originalUrl.includes('../') || req.originalUrl.includes('..\\')) {
    logger.warn('Path traversal attempt detected', {
      originalUrl: req.originalUrl,
      ip: req.ip,
    });
    return res.status(400).json({
      success: false,
      error: 'Invalid filename',
      timestamp: new Date().toISOString(),
    });
  }
  // Check for access to sensitive system paths (likely from path traversal)
  const sensitivePaths = [
    '/etc/',
    '/bin/',
    '/usr/',
    '/sys/',
    '/proc/',
    '/var/',
    '/tmp/',
    '/root/',
    'C:\\',
    'D:\\',
  ];
  if (sensitivePaths.some((p) => req.path.startsWith(p) || req.path.includes(p))) {
    logger.warn('Attempt to access sensitive path', {
      path: req.path,
      ip: req.ip,
    });
    return res.status(400).json({
      success: false,
      error: 'Invalid filename',
      timestamp: new Date().toISOString(),
    });
  }
  next();
});
app.use(express.static(path.join(__dirname, '../public')));
app.use('/docs', swaggerUi.serve);
app.get('/docs', swaggerUi.setup(swaggerSpec, swaggerUiOptions));
app.get('/docs.json', (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});
app.use('/api', executeRoutes);
app.use('/api', fileRoutes);
app.use('/api', executeRawRoutes);
app.use('/api', uploadRoutes);
app.use('/dashboard', DashboardRoutes);
app.use('/api/dashboard', statsRoutes);
app.use('/api/debug', crashReportsRoutes);
app.use('/', healthRoutes);
app.get('/metrics', metricsEndpoint);
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

app.use((err, req, res, _next) => {
  logger.error('Unhandled error:', err);
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: `File size too large. Maximum size: ${config.maxFileSize / 1024 / 1024}MB`,
      });
    }
    if (err.code === 'LIMIT_UNEXPECTED_FILE') {
      return res.status(400).json({
        success: false,
        error: `Unexpected file field. Expected field name: 'file', received: '${err.field}'`,
      });
    }
  }
  // Handle file type validation errors
  if (err.code === 'INVALID_FILE_TYPE') {
    return res.status(400).json({
      success: false,
      error: err.message,
    });
  }
  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message,
    timestamp: new Date().toISOString(),
  });
});
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    timestamp: new Date().toISOString(),
  });
});
module.exports = app;
