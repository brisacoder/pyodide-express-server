/**
 * Application Constants - Centralized Configuration Values
 *
 * This file contains all magic numbers and configuration constants used
 * throughout the Pyodide Express Server application. Centralizing these
 * values makes the codebase more maintainable and easier to configure.
 *
 * @module constants
 * @since 1.0.0
 */
/**
 * Time-related constants in milliseconds and seconds
 */
const TIME = {
  // Milliseconds
  SECOND: 1000,
  MINUTE: 60 * 1000,
  HOUR: 60 * 60 * 1000,
  DAY: 24 * 60 * 60 * 1000,
  // Seconds (for calculations)
  SECONDS_PER_MINUTE: 60,
  SECONDS_PER_HOUR: 3600,
  SECONDS_PER_DAY: 86400,
  // Hours
  HOURS_PER_DAY: 24,
};
/**
 * File size constants (in bytes)
 */
const FILE_SIZE = {
  KB: 1024,
  MB: 1024 * 1024,
  GB: 1024 * 1024 * 1024,
  // Specific sizes
  LOG_SIZE_5MB: 5 * 1024 * 1024,
  UPLOAD_SIZE_10MB: 10 * 1024 * 1024,
  CODE_SIZE_1MB: 1024 * 1024,
};
/**
 * Logging configuration constants
 */
const LOGGING = {
  MAX_LOG_SIZE: FILE_SIZE.LOG_SIZE_5MB, // 5MB default log rotation size
  ROTATION_COUNT: 5, // Keep 5 rotated log files
  DEFAULT_LEVEL: 'info', // Default log level
  STATS_RETENTION_HOURS: 24, // Keep stats for 24 hours
  MAX_IP_TRACKING: 100, // Track up to 100 unique IPs
};
/**
 * Security configuration constants
 */
const SECURITY = {
  HASH_ALGORITHM: 'sha256', // SHA-256 for code hashing
  STATS_RETENTION_TIME: TIME.DAY, // 24 hours in milliseconds
  MAX_REQUEST_SIZE: FILE_SIZE.UPLOAD_SIZE_10MB, // 10MB max request size
  allowedExts: new Set(['.csv', '.txt', '.py', '.png', '.json']), // adjust as needed
};
/**
 * Python code execution constants
 */
const EXECUTION = {
  DEFAULT_TIMEOUT: 10000, // 10 seconds default timeout (further reduced for stability)
  MAX_TIMEOUT: 30000, // 30 seconds maximum timeout (further reduced)
  MIN_TIMEOUT: 1000, // 1 second minimum timeout
  MAX_CODE_LENGTH: FILE_SIZE.CODE_SIZE_1MB, // 1MB maximum code size
  MEMORY_LIMIT: 128 * FILE_SIZE.MB, // 128MB memory limit (further reduced)
  MAX_ITERATIONS: 50000, // Maximum loop iterations before timeout (reduced)
  MAX_RECURSION_DEPTH: 50, // Maximum recursion depth (reduced)
  HEARTBEAT_INTERVAL: 5000, // Check runtime health every 5 seconds
};
/**
 * Server configuration constants
 */
const SERVER = {
  DEFAULT_PORT: 3000, // Default HTTP port
  SWAGGER_PORT: 3000, // Port for Swagger docs
  SHUTDOWN_TIMEOUT: 10000, // 10 seconds for graceful shutdown
};
/**
 * Network and API constants
 */
const NETWORK = {
  DEFAULT_CORS_ORIGIN: '*', // Default CORS origin
  MAX_UPLOAD_SIZE: FILE_SIZE.UPLOAD_SIZE_10MB, // Maximum upload size
  LOCALHOST: 'localhost', // Default hostname
  DEFAULT_UPLOAD_DIR: 'uploads', // Default upload directory
};
/**
 * Pyodide-specific constants
 */
const PYODIDE = {
  CDN_VERSION: '0.28.2', // Pyodide CDN version
  INDEX_URL: 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/',
  INIT_TIMEOUT: 60000, // 60 seconds for initialization
  ISOLATION_NAMESPACE: '__user_code__', // Namespace for user code isolation
  // Default bases (name = host folder under pyodide-data, urlBase = relative path for URLs)
  DEFAULT_PYODIDE_DATA_DIR: 'pyodide_data',
  DEFAULT_BASES : {
    uploads: { urlBase: 'uploads', subfolders: [] },
    plots:   { urlBase: 'plots',   subfolders: [] },
    in:      { urlBase: 'in',      subfolders: [] },
    admin:   { urlBase: 'admin',   subfolders: [] },
  }
};

/**
 * Process Pool configuration constants
 */
const PROCESS_POOL = {
  DEFAULT_POOL_SIZE: 2,              // Number of persistent Pyodide processes
  MIN_POOL_SIZE: 1,                  // Minimum pool size
  MAX_POOL_SIZE: 8,                  // Maximum pool size (reasonable for most systems)
  MAX_EXECUTIONS_PER_PROCESS: 50,   // Recycle process after this many executions
  PROCESS_INIT_TIMEOUT: 90000,       // 90 seconds for Pyodide initialization
  PROCESS_IDLE_TIMEOUT: 300000,      // 5 minutes idle timeout before process cleanup
  POOL_HEALTH_CHECK_INTERVAL: 30000, // Check pool health every 30 seconds
};
/**
 * Performance and monitoring constants
 */
const PERFORMANCE = {
  STATS_UPDATE_INTERVAL: TIME.MINUTE, // Update stats every minute
  CLEANUP_INTERVAL: TIME.HOUR, // Cleanup old data every hour
  METRIC_PRECISION: 2, // Decimal places for metrics
  MAX_RECENT_EXECUTIONS: 1000, // Keep last 1000 executions in memory
};
/**
 * Dashboard and Chart.js configuration constants
 */
const DASHBOARD = {
  // Chart.js CDN Configuration
  CHARTJS: {
    LOCAL_PATH: '/js/chart.js', // Local Chart.js file path
    CDN_PRIMARY: 'https://cdn.jsdelivr.net/npm/chart.js',
    CDN_FALLBACK: 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js',
    CDN_TERTIARY: 'https://unpkg.com/chart.js@3.9.1/dist/chart.min.js',
    VERSION: '4.5.0', // Chart.js version
    TIMEOUT: 5000, // CDN load timeout in milliseconds
  },
  // Chart Configuration
  CHARTS: {
    ANIMATION_DURATION: 750, // Chart animation duration in ms
    RESPONSIVE: true, // Make charts responsive
    MAINTAIN_ASPECT_RATIO: false, // Allow flexible aspect ratios
    POINT_RADIUS: 4, // Chart point radius
    BORDER_WIDTH: 2, // Chart border width
    COLORS: {
      PRIMARY: '#667eea', // Primary chart color
      SUCCESS: '#28a745', // Success color
      ERROR: '#dc3545', // Error color
      BACKGROUND: 'rgba(102, 126, 234, 0.2)', // Background fill color
      BORDER: '#fff', // Border color
    },
  },
  // Dashboard UI Configuration
  UI: {
    REFRESH_DEBOUNCE: 1000, // Debounce refresh button (ms)
    AUTO_REFRESH_INTERVAL: 30000, // Auto-refresh every 30 seconds
    MAX_TABLE_ROWS: 100, // Maximum rows in recent executions table
    CHART_HEIGHT: 400, // Default chart height in pixels
    CHART_WIDTH: 600, // Default chart width in pixels
  },
  // Data Display Configuration
  DATA: {
    TIME_FORMAT: 'HH:mm:ss', // Time format for displays
    DATE_FORMAT: 'YYYY-MM-DD HH:mm:ss', // Full datetime format
    DECIMAL_PLACES: 2, // Decimal places for metrics
    HOURS_DISPLAYED: 24, // Hours shown in trend chart
    MAX_RECENT_ITEMS: 50, // Max items in recent executions
  },
};

module.exports = {
  TIME,
  FILE_SIZE,
  LOGGING,
  SECURITY,
  EXECUTION,
  SERVER,
  NETWORK,
  PYODIDE,
  PROCESS_POOL,
  PERFORMANCE,
  DASHBOARD,
};
