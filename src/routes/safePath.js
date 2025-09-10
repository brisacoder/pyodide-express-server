// safe-path.js (CommonJS; run with: node safe-path.js)
const fs = require('node:fs/promises');
const path = require('node:path');

/* ----------------------- Filename sanitization (no deps) -------------------- */

/**
 * Make a single filename segment safe across platforms.
 * - Normalizes Unicode (NFC)
 * - Removes control / forbidden path chars
 * - Collapses whitespace and trims
 * - Avoids Windows-reserved names by suffixing an underscore
 * - Truncates to <=255 bytes in UTF-8 (common FS limit for a segment)
 * @param {string} input - The filename segment to sanitize.
 * @param {Object} [root0] - Optional configuration object.
 * @param {string} [root0.replacement='-'] - Replacement character for forbidden characters.
 * @returns {string} - The sanitized filename segment.
 */
function sanitizeFilenameSegment(input, { replacement = '-' } = {}) {
  const forbiddenRe = /[<>:"/\\|?*\u0000-\u001F]/g; // Windows + control chars
  const reservedWinNames = new Set([
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',
    'LPT1','LPT2','LPT3','LPT4','LPT5','LPT6','LPT7','LPT8','LPT9',
  ]);

  let name = String(input)
    .normalize('NFC')
    .replace(forbiddenRe, replacement)
    .replace(/\s+/g, ' ')
    .trim();

  // Disallow trailing spaces/dots (Windows quirk)
  name = name.replace(/[ .]+$/g, '');

  if (name.length === 0) name = 'unnamed';

  if (reservedWinNames.has(name.toUpperCase())) {
    name = `${name}_`;
  }

  // Keep single segment <= 255 bytes
  if (Buffer.byteLength(name, 'utf8') > 255) {
    const chars = Array.from(name);
    while (Buffer.byteLength(chars.join(''), 'utf8') > 255) {
      chars.pop();
    }
    name = chars.join('');
  }

  return name;
}


/**
 * Safely join an untrusted relative path to a trusted base directory.
 * - Canonicalizes the base (resolving symlinks)
 * - Resolves the candidate path
 * - Verifies it remains inside the base
 * @param {string} baseDir - Trusted base directory.
 * @param {string} userPath - Untrusted user-supplied path.
 * @returns {Promise<string>} - Resolves to the safely joined absolute path.
 */
async function safeJoin(baseDir, userPath) {
  const baseReal = await fs.realpath(baseDir);
  const target = path.resolve(baseReal, userPath);
  const rel = path.relative(baseReal, target);

  // If rel starts with '..' or is absolute, it's outside base
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error('Path traversal attempt blocked');
  }
  return target;
}

/**
 * Safely create a file from untrusted pieces (folder + filename).
 * @param {string} baseDir - Trusted base directory for file creation.
 * @param {string} unsafeFolder - Untrusted folder name segment.
 * @param {string} unsafeFile - Untrusted filename segment.
 * @returns {Promise<string>} - Resolves to the full path of the created file.
 */
async function createUserFile(baseDir, unsafeFolder, unsafeFile) {
  // Sanitize each segment you intend to treat as a folder or filename
  const cleanFolder = sanitizeFilenameSegment(unsafeFolder);
  const cleanFile = sanitizeFilenameSegment(unsafeFile);

  // Compose a relative subpath from sanitized segments
  const relative = path.join(cleanFolder, cleanFile);

  // Join safely against a trusted base and write
  const finalPath = await safeJoin(baseDir, relative);
  await fs.mkdir(path.dirname(finalPath), { recursive: true });
  await fs.writeFile(finalPath, 'hello, safe world\n', { flag: 'wx' }); // 'wx' avoids overwriting
  return finalPath;
}

// Demo when run directly:
if (require.main === module) {
  (async () => {
    const base = path.join(process.cwd(), 'uploads');
    const saved = await createUserFile(base, '../evil/..//', 'NUL:.txt');
    console.log('Saved at:', saved);
  })().catch(err => {
    console.error(err.message);
    process.exit(1);
  });
}

module.exports = { sanitizeFilenameSegment, safeJoin, createUserFile };
