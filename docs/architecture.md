# Architecture Overview

This document provides a narrative tour of the Pyodide Express Server.
It is meant to complement the detailed notes in `pyodide_arch.md`.

## Request Flow
1. **Express server** (`src/server.js`) boots and initializes the `PyodideService`.
2. **PyodideService** (`src/services/pyodide-service.js`) loads the WebAssembly
   runtime and a few common Python packages.
3. Incoming API requests are routed to small handlers in `src/routes` which call
   into the service to execute Python code, install additional packages, or
   reset the runtime.
4. Results, stdout, and any errors are captured and returned as JSON.

## Why Pyodide?
Pyodide bundles the CPython interpreter for the browser and WebAssembly
environments. Using it on the server allows us to execute Python code in a
sandbox without requiring a system Python installation.

## Extensibility
- Additional endpoints can be added under `src/routes/`.
- The service exposes hooks for package management and environment resets.
- Because everything runs inside one runtime instance, complex operations such
  as streaming output or loading data files become easy to expose via HTTP.

For a line-by-line explanation of the initialization and execution process, see
[`pyodide_arch.md`](../pyodide_arch.md).
