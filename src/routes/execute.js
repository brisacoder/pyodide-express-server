const express = require('express');
const { executePythonCode } = require('../services/pyodide-service');
const { validateCode } = require('../middleware/validation');

const router = express.Router();

router.post('/execute', validateCode, async (req, res) => {
  // Python execution logic
});

module.exports = router;