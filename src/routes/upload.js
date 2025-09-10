const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const config = require('../config');
const { uploadFile } = require('../controllers/uploadController');
const { sanitizeFilenameSegment } = require('./safePath'); // from earlier code
const router = express.Router();
const pyodideService = require('../services/pyodide-service');

/**
 * Guard against double-extension tricks (e.g., "data.csv.exe").
 * Accept only if final ext is allowed AND base (before that ext) has no dot.
 * @param {string} originalName - The original filename to check.
 * @param {Set<string>} allowedExts - Set of allowed file extensions (including dot).
 * @returns {boolean} - True if the extension and base are allowed, false otherwise.
 */
function isAllowedExtStrict(originalName, allowedExts) {
  const ext = path.extname(originalName).toLowerCase();
  if (!allowedExts.has(ext)) return false;
  const base = path.basename(originalName, ext);
  return !base.includes('.');
}

/**
 * Conflict-aware unique filename generator (sync, simple).
 * Produces: base.ext, base-1.ext, base-2.ext, ...
 * @param {string} dir - Directory to check for filename conflicts.
 * @param {string} base - Base filename (without extension).
 * @param {string} ext - File extension (including dot).
 * @returns {string} - Unique filename (with extension).
 */
function uniqueFilenameSync(dir, base, ext) {
  let attempt = 0;
  // Safety: clamp attempts to avoid rare infinite loops
  const MAX_ATTEMPTS = 100;

  while (attempt < MAX_ATTEMPTS) {
    const suffix = attempt === 0 ? '' : `-${attempt}`;
    const candidate = path.join(dir, `${base}${suffix}${ext}`);
    if (!fs.existsSync(candidate)) {
      return path.basename(candidate);
    }
    attempt += 1;
  }
  // Extremely unlikely—surface a meaningful error
  throw new Error('Unable to allocate a unique filename after many attempts');
}


const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, pyodideService.basesReal.uploads.dirReal), // Use the real uploads path

  filename: (req, file, cb) => {
    try {
      const ext = path.extname(file.originalname).toLowerCase();
      const rawBase = path.basename(file.originalname, ext);

      const uploadsDir = pyodideService.basesReal.uploads.dirReal;
      const safeBase = sanitizeFilenameSegment(rawBase) || 'unnamed';
      const finalName = uniqueFilenameSync(uploadsDir, safeBase, ext);

      cb(null, finalName);
    } catch (err) {
      cb(err);
    }
  },
});

const upload = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1 },
  fileFilter: (req, file, cb) => {
    try {
      // 1) Extension allow-list (strict)
      if (!isAllowedExtStrict(file.originalname, config.allowedExts)) {
        return cb(new Error(`Only ${[...config.allowedExts].join(', ')} files are allowed`), false);
      }
      // 2) Filename sanity gate (full name — belt & suspenders)
      const safeName = sanitizeFilenameSegment(file.originalname);
      if (!safeName) {
        return cb(new Error('Unsafe filename'), false);
      }

      // Optional: stash sanitized original for logs/audit
      file.safeOriginalName = safeName;

      return cb(null, true);
    } catch (err) {
      return cb(err);
    }
  },
});

router.post('/upload-csv', upload.single('file'), uploadFile);
module.exports = router;
