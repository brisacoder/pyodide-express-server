/**
 * Prometheus-style Metrics Collection for HTTP Performance Monitoring
 * 
 * Collects and exposes HTTP request metrics in Prometheus format for
 * monitoring, alerting, and performance analysis.
 */

/**
 * Express Request object with extended properties
 * @typedef {Object} ExpressRequest
 * @property {string} method - HTTP method
 * @property {string} url - Request URL
 */

/**
 * Express Response object with extended properties  
 * @typedef {Object} ExpressResponse
 * @property {number} statusCode - HTTP status code
 * @property {Function} status - Set status code
 */

/**
 * Express Next function
 * @typedef {Function} ExpressNext
 * @param {Error} [error] - Optional error to pass to error handler
 */

/**
 * In-memory metrics storage object.
 * Tracks counters and timing data for HTTP requests.
 * 
 * @typedef {Object} MetricsStorage
 * @property {number} requestsTotal - Total HTTP requests received
 * @property {number} requestErrorsTotal - Total 5xx error responses
 * @property {number} requestDurationSecondsSum - Sum of all request durations in seconds
 * @property {number} requestDurationSecondsCount - Count of timed requests
 */

/**
 * @type {MetricsStorage}
 */
const metrics = {
  requestsTotal: 0,
  requestErrorsTotal: 0,
  requestDurationSecondsSum: 0,
  requestDurationSecondsCount: 0,
};

/**
 * Express middleware that collects HTTP request metrics.
 * Measures request duration and counts total requests and errors.
 * 
 * @param {ExpressRequest} req - Express request object with extended properties
 * @param {ExpressResponse} res - Express response object with extended properties
 * @param {ExpressNext} next - Express next middleware function
 * @returns {void} Calls next() after setting up metrics collection
 * 
 * @example
 * // In app.js
 * const { metricsMiddleware } = require('./utils/metrics');
 * app.use(metricsMiddleware);
 * 
 * // Automatically collects metrics for all requests:
 * // - Total request count
 * // - Request duration timing
 * // - Error count (5xx responses)
 * 
 * @description
 * Metrics Collected:
 * - http_requests_total: Counter of all HTTP requests
 * - http_request_errors_total: Counter of 5xx error responses
 * - http_request_duration_seconds: Summary of request durations
 * 
 * Performance Characteristics:
 * - Uses high-resolution process.hrtime.bigint() for accurate timing
 * - Minimal overhead (~microseconds per request)
 * - Memory-efficient counters
 * - Non-blocking metrics collection
 * 
 * Integration:
 * - Compatible with Prometheus monitoring
 * - Works with Grafana dashboards
 * - Suitable for alerting rules
 * - Enables SLA/SLO tracking
 */
function metricsMiddleware(req, res, next) {
  metrics.requestsTotal += 1;
  const start = process.hrtime.bigint();
  res.on('finish', () => {
    const end = process.hrtime.bigint();
    const duration = Number(end - start) / 1e9; // seconds
    metrics.requestDurationSecondsSum += duration;
    metrics.requestDurationSecondsCount += 1;
    if (res.statusCode >= 500) {
      metrics.requestErrorsTotal += 1;
    }
  });
  next();
}

/**
 * Express route handler that exposes metrics in Prometheus format.
 * Returns text/plain response with metric data for scraping.
 * 
 * @param {ExpressRequest} req - Express request object
 * @param {ExpressResponse} res - Express response object  
 * @returns {void} Sends Prometheus metrics as plain text response
 * 
 * @example
 * // In routes setup
 * app.get('/metrics', metricsEndpoint);
 * 
 * // GET /metrics returns:
 * // # HELP http_requests_total Total number of HTTP requests
 * // # TYPE http_requests_total counter
 * // http_requests_total 1234
 * // # HELP http_request_errors_total Total number of HTTP error responses (5xx)
 * // # TYPE http_request_errors_total counter
 * // http_request_errors_total 5
 * // # HELP http_request_duration_seconds Summary of HTTP request durations in seconds
 * // # TYPE http_request_duration_seconds summary
 * // http_request_duration_seconds_sum 45.67
 * // http_request_duration_seconds_count 1234
 * 
 * @description
 * Prometheus Format:
 * - Includes HELP and TYPE comments for each metric
 * - Follows Prometheus naming conventions
 * - Uses appropriate metric types (counter, summary)
 * - Compatible with standard Prometheus scrapers
 * 
 * Monitoring Setup:
 * ```yaml
 * # prometheus.yml
 * scrape_configs:
 *   - job_name: 'pyodide-server'
 *     static_configs:
 *       - targets: ['localhost:3000']
 *     metrics_path: '/metrics'
 *     scrape_interval: 15s
 * ```
 * 
 * Calculated Metrics:
 * - Average request duration: sum / count
 * - Error rate: errors / total requests
 * - Request rate: total / uptime
 * - P95/P99 latencies: requires additional histogram
 */
function metricsEndpoint(req, res) {
  res.setHeader('Content-Type', 'text/plain; version=0.0.4');
  let output = '';
  output += '# HELP http_requests_total Total number of HTTP requests\n';
  output += '# TYPE http_requests_total counter\n';
  output += `http_requests_total ${metrics.requestsTotal}\n`;
  output += '# HELP http_request_errors_total Total number of HTTP error responses (5xx)\n';
  output += '# TYPE http_request_errors_total counter\n';
  output += `http_request_errors_total ${metrics.requestErrorsTotal}\n`;
  output += '# HELP http_request_duration_seconds Summary of HTTP request durations in seconds\n';
  output += '# TYPE http_request_duration_seconds summary\n';
  output += `http_request_duration_seconds_sum ${metrics.requestDurationSecondsSum}\n`;
  output += `http_request_duration_seconds_count ${metrics.requestDurationSecondsCount}\n`;
  res.send(output);
}

module.exports = { metricsMiddleware, metricsEndpoint };
