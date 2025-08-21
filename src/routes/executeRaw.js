const express = require('express');
const { executeRaw } = require('../controllers/executeController');
const router = express.Router();
router.post('/execute-raw', express.text({ limit: '10mb' }), executeRaw);
module.exports = router;
