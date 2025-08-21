const express = require('express');
const { health, healthcheck, readiness } = require('../controllers/healthController');
const router = express.Router();
// Basic health check
router.get('/health', health);
// Detailed health check for monitoring
router.get('/healthcheck', healthcheck);
// Kubernetes-style readiness probe
router.get('/ready', readiness);
module.exports = router;
