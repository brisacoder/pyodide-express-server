const metrics = {
  requestsTotal: 0,
  requestErrorsTotal: 0,
  requestDurationSecondsSum: 0,
  requestDurationSecondsCount: 0,
};

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
