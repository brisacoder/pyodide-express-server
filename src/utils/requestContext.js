const { AsyncLocalStorage } = require('async_hooks');
const { randomUUID } = require('crypto');

const asyncLocalStorage = new AsyncLocalStorage();

function requestContextMiddleware(req, res, next) {
  const requestId = randomUUID();
  asyncLocalStorage.run({ requestId }, () => {
    res.setHeader('X-Request-Id', requestId);
    next();
  });
}

function getRequestId() {
  const store = asyncLocalStorage.getStore();
  return store ? store.requestId : undefined;
}

module.exports = { requestContextMiddleware, getRequestId };
