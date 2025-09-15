// layout.js
const fs = require('node:fs/promises');
const path = require('node:path');

/**
 * @typedef {Object} BaseConfig
 * @property {string} urlBase
 *   Relative path for URLs (will be prefixed with '/' automatically), e.g. "uploads".
 * @property {string[]} [subfolders]
 *   Optional list of subdirectory names to create under this base directory on disk.
 *   Each name must match /^[A-Za-z0-9._-]+$/.
 */

/**
 * @typedef {Object<string, BaseConfig>} BasesConfig
 * An object keyed by base name. Example:
 * {
 *   uploads: { urlBase: '/uploads', subfolders: [] },
 *   plots:   { urlBase: '/plots',   subfolders: ['matplotlib','seaborn'] }
 * }
 */

/**
 * @typedef {Object} BaseRealized
 * @property {string} name
 *   The base key (e.g. "uploads", "plots").
 * @property {string} urlBase
 *   The same public URL prefix provided in the config.
 * @property {string} dirReal
 *   Canonical (symlink-resolved) absolute filesystem path for this base.
 */

/**
 * @typedef {Object<string, BaseRealized>} BasesRealized
 * Object keyed by base name mapping to realized base info.
 */

/**
 * Ensure a directory exists and return its canonical (symlink-resolved) absolute path.
 *
 * Behavior:
 * - If the path does not exist, it is created recursively.
 * - If the path exists but is not a directory, the function throws (never overwrites files).
 * - Always returns the canonical absolute path (via `fs.realpath`).
 *
 * @param {string} dirPath
 *   Path to the directory to ensure (may be relative or absolute).
 * @returns {Promise<string>}
 *   Canonical absolute path to the ensured directory.
 * @throws {Error}
 *   If the path exists but is not a directory, or creation fails.
 */
async function ensureDirReal(dirPath) {
  try {
    const st = await fs.stat(dirPath).catch(() => null);
    if (st && !st.isDirectory()) {
      throw new Error(`Path exists and is not a directory: ${dirPath}`);
    }
    if (!st) {
      await fs.mkdir(dirPath, { recursive: true });
    }
    return await fs.realpath(dirPath);
  } catch (err) {
    err.message = `Failed to ensure directory "${dirPath}": ${err.message}`;
    throw /** @type {Error} */ (err);
  }
}

/**
 * Initialize a generic on-disk layout for Pyodide-visible bases.
 *
 * Creates the `rootDir`, then for each key in `bases`:
 * - Validates base name (must match `/^[A-Za-z0-9._-]+$/`)
 * - Creates `<root>/<baseName>` and resolves it to a canonical path
 * - Creates each subfolder `<root>/<baseName>/<subfolder>`
 *   (each subfolder name must match `/^[A-Za-z0-9._-]+$/`)
 *
 * Returns canonical paths so downstream code can safely mount/serve and perform
 * containment checks without symlink surprises.
 *
 * @param {string} rootDir
 *   Root directory under which all bases will be created (can be relative; will be resolved).
 * @param {BasesConfig} bases
 *   Object keyed by base name with URL prefix and optional subfolders.
 * @returns {Promise<{ rootReal: string, basesReal: BasesRealized }>}
 *   Canonical root path and an object keyed by base name containing realized info.
 * @throws {Error}
 *   If any base name or subfolder is invalid, or directory creation fails.
 *
 * @example
 * // Given:
 * // const bases = {
 * //   uploads: { urlBase: 'uploads', subfolders: [] },
 * //   plots:   { urlBase: 'plots',   subfolders: ['matplotlib','seaborn'] },
 * // };
 * //
 * // const { rootReal, basesReal } = await initLayout('pyodide-data', bases);
 * // console.log(rootReal);                // "/abs/path/to/pyodide-data"
 * // console.log(basesReal.uploads.dirReal); // "/abs/path/to/pyodide-data/uploads"
 * // console.log(basesReal.plots.urlBase);   // "/plots"
 */
async function initLayout(rootDir, bases) {
  const rootReal = await ensureDirReal(rootDir);

  /** @type {BasesRealized} */
  const out = {};

  for (const [name, cfg] of Object.entries(bases)) {
    if (!/^[A-Za-z0-9._-]+$/.test(name)) {
      throw new Error(`Invalid base name "${name}" (allowed: A-Z a-z 0-9 . _ -)`);
    }
    if (typeof cfg?.urlBase !== 'string' || cfg.urlBase.length === 0) {
      throw new Error(`Base "${name}" must have a non-empty urlBase string: got ${String(cfg?.urlBase)}`);
    }

    const baseDir = path.join(rootReal, name);
    const baseReal = await ensureDirReal(baseDir);

    const subs = Array.isArray(cfg.subfolders) ? cfg.subfolders : [];
    for (const sub of subs) {
      if (!/^[A-Za-z0-9._-]+$/.test(sub)) {
        throw new Error(`Invalid subfolder "${sub}" in base "${name}" (allowed: A-Z a-z 0-9 . _ -)`);
      }
      await ensureDirReal(path.join(baseReal, sub));
    }

    // Ensure urlBase starts with '/' for consistency
    const urlBase = cfg.urlBase.startsWith('/') ? cfg.urlBase : `/${cfg.urlBase}`;
    out[name] = { name, urlBase, dirReal: baseReal };
  }

  return { rootReal, basesReal: out };
}

module.exports = {
  ensureDirReal,
  initLayout,
};
