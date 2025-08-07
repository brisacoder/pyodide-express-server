const express = require('express');
const path = require('path');
const multer = require('multer');
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

const app = express();

app.use(requestContextMiddleware);
app.use(metricsMiddleware);

// JSON parsing with error handling
app.use((req, res, next) => {
  express.json({ limit: '30mb' })(req, res, (err) => {
    if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
      return res.status(400).json({
        success: false,
        error: 'Invalid JSON format',
        timestamp: new Date().toISOString()
      });
    }
    next(err);
  });
});

app.use(express.urlencoded({ extended: true, limit: '30mb' }));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', config.corsOrigin);
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

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
      ip: req.ip
    });
    return res.status(400).json({
      success: false,
      error: 'Invalid filename',
      timestamp: new Date().toISOString()
    });
  }
  
  // Check for access to sensitive system paths (likely from path traversal)
  const sensitivePaths = ['/etc/', '/bin/', '/usr/', '/sys/', '/proc/', '/var/', '/tmp/', '/root/', 'C:\\', 'D:\\'];
  if (sensitivePaths.some(p => req.path.startsWith(p) || req.path.includes(p))) {
    logger.warn('Attempt to access sensitive path', {
      path: req.path,
      ip: req.ip
    });
    return res.status(400).json({
      success: false,
      error: 'Invalid filename',
      timestamp: new Date().toISOString()
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
app.use('/', healthRoutes);
app.get('/metrics', metricsEndpoint);

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);

  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: `File size too large. Maximum size: ${config.maxFileSize / 1024 / 1024}MB`,
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
