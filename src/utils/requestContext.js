/**
 * Request Context Management for Request ID Tracking
 *
 * Provides request ID generation and async context propagation throughout
 * the request lifecycle for better logging and debugging capabilities.
 */
/**
 * Express Request object with extended properties
 * @typedef {Object} ExpressRequest
 * @property {string} method - HTTP method
 * @property {string} url - Request URL
 * @property {Object} headers - Request headers
 * @property {string} ip - Client IP address
 * @property {string} requestId - Unique request identifier
 */
/**
 * Express Response object with extended properties
 * @typedef {Object} ExpressResponse
 * @property {number} statusCode - HTTP status code
 * @property {Function} status - Set status code
 * @property {Function} json - Send JSON response
 * @property {Function} set - Set response header
 */
/**
 * Express Next function
 * @typedef {Function} ExpressNext
 * @param {Error} [error] - Optional error to pass to error handler
 */
const { AsyncLocalStorage } = require('async_hooks');
const { randomUUID } = require('crypto');
const asyncLocalStorage = new AsyncLocalStorage();
/**
 * Express middleware that generates and tracks unique request IDs.
 * Creates async context that persists throughout the entire request lifecycle.
 *
 * @param {ExpressRequest} req - Express request object with extended properties
 * @param {ExpressResponse} res - Express response object with extended properties
 * @param {ExpressNext} next - Express next middleware function
 * @returns {void} Calls next() after setting up request context
 *
 * @example
 * // In app.js
 * const { requestContextMiddleware } = require('./utils/requestContext');
 * app.use(requestContextMiddleware);
 *
 * // Request headers will include:
 * // X-Request-Id: 550e8400-e29b-41d4-a716-446655440000
 *
 * @description
 * Features:
 * - Generates UUID v4 for each request
 * - Sets X-Request-Id response header
 * - Creates async context for request tracking
 * - Enables request ID access in nested function calls
 * - Persists context across async operations
 *
 * Benefits:
 * - Correlate logs across request lifecycle
 * - Debug distributed operations
 * - Track requests through middleware chain
 * - Identify related log entries
 *
 * Performance:
 * - Minimal overhead using Node.js async_hooks
 * - UUID generation is cryptographically secure
 * - Context propagation is automatic
 */
function requestContextMiddleware(req, res, next) {
  const requestId = randomUUID();
  asyncLocalStorage.run({ requestId }, () => {
    res.setHeader('X-Request-Id', requestId);
    next();
  });
}
/**
 * Retrieves the current request ID from async local storage.
 * Safe to call from any function within a request's execution context.
 *
 * @returns {string|undefined} Current request ID, or undefined if not in request context
 *
 * @example
 * // In any function called during request processing
 * const requestId = getRequestId();
 * if (requestId) {
 *   console.log(`Processing request: ${requestId}`);
 * }
 *
 * // In logger.js
 * function write(level, message, meta) {
 *   const requestId = getRequestId();
 *   if (requestId) {
 *     meta.requestId = requestId;
 *   }
 *   // ... rest of logging
 * }
 *
 * // In error handling
 * catch (error) {
 *   logger.error('Execution failed', {
 *     error: error.message,
 *     requestId: getRequestId() // Automatically includes current request ID
 *   });
 * }
 *
 * @description
 * Use Cases:
 * - Add request ID to log entries
 * - Correlate errors with specific requests
 * - Debug async operation chains
 * - Track performance across request lifecycle
 *
 * Context Safety:
 * - Returns undefined when called outside request context
 * - Safe to call from any depth of function nesting
 * - Works across Promise chains and async/await
 * - Survives setTimeout and setInterval callbacks
 */
function getRequestId() {
  const store = asyncLocalStorage.getStore();
  return store ? store.requestId : undefined;
}
module.exports = { requestContextMiddleware, getRequestId };
