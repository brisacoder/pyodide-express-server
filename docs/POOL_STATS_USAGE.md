# Pool & Memory Stats Usage

This document explains how to query the Pyodide process pool and memory statistics to diagnose memory growth or leaking tests.

## Endpoint

`GET /api/pool-stats`

### Query Parameters
- `verbose=1` (optional): include raw `process.memoryUsage()` data from the parent Node process.

## Response Contract
```json
{
  "success": true,
  "data": {
    "pool": { "poolSize": 2, "activeProcesses": 2, "availableProcesses": 2, ... },
    "memory": { "timestamp": "2025-09-17T10:00:00.000Z", "nodeProcess": { "rss": 123456789, ... } },
    "uptimeSeconds": 1234,
    "adaptive": { "finalPoolSize": 2, "reason": "requested_ok", ... },
    "node": { "rss": 123456789, ... } // only if verbose=1
  },
  "error": null,
  "meta": { "timestamp": "2025-09-17T10:00:00.000Z" }
}
```

## Basic curl
```bash
curl -s http://localhost:3000/api/pool-stats | jq
```

## Verbose (includes raw Node memory usage)
```bash
curl -s "http://localhost:3000/api/pool-stats?verbose=1" | jq '.data | {pool: .pool.poolSize, active: .pool.activeProcesses, avail: .pool.availableProcesses, adaptive: .adaptive.reason, rssMB: (.node.rss/1024/1024|floor)}'
```

## Compact one-line logging (good for watch loops)
```bash
curl -s "http://localhost:3000/api/pool-stats?verbose=1" \
 | jq -r '"POOL | ts=", .meta.timestamp, " pool=", .data.pool.poolSize, " active=", .data.pool.activeProcesses, " avail=", .data.pool.availableProcesses, " rssMB=", ((.data.node.rss/1024/1024)|floor), " execTotal=", .data.pool.totalExecutions, " reason=", .data.adaptive.reason' | paste -sd '' -
```

## Monitor continuously during a long test run
```bash
watch -n 5 'curl -s "http://localhost:3000/api/pool-stats?verbose=1" | jq "{ts: .meta.timestamp, pool: .data.pool.poolSize, active: .data.pool.activeProcesses, avail: .data.pool.availableProcesses, rssMB: (.data.node.rss/1024/1024|floor), execTotal: .data.pool.totalExecutions, reason: .data.adaptive.reason}"'
```

## After-each-test Capture (already enabled)
Pytest `conftest.py` includes an `autouse` fixture printing lines like:
```
MEMSTATS | test=tests/test_example.py::test_case | poolSize=2 active=2 avail=2 rssMB=345.1 execTotal=120
```
You can aggregate by test to find memory growth deltas.

## Identifying Culprits
1. Run the full suite.
2. Capture output: `pytest -q > test_run.log 2>&1`.
3. Extract memory lines: `grep '^MEMSTATS' test_run.log > mem.log`.
4. Compare sequential RSS values to spot jumps.

## Environment Tuning
- `PYODIDE_POOL_SIZE` : Requested pool size (before adaptation)
- `PYODIDE_EST_PROC_MB` : Estimated per-process memory (default 250)
- `PYODIDE_MAX_POOL_MEM_FRACTION` : Fraction of total RAM for pool (default 0.5)
- `PYODIDE_MAX_POOL_SIZE` : Hard upper cap

Reduce `PYODIDE_EST_PROC_MB` only if empirical measurements show lower stable usage; otherwise leave default to stay safe.

## Troubleshooting
| Symptom | Cause | Action |
|---------|-------|--------|
| rssMB steadily climbs | Leaking Python objects retained across executions | Add recycling thresholds or investigate specific test code |
| avail=0 for long periods | Pool saturation | Increase pool size (if memory permits) |
| reason=capped_memory | Adaptive sizing reduced requested size | Increase RAM or lower per-process estimate |

## Example jq Field Map
```bash
curl -s http://localhost:3000/api/pool-stats?verbose=1 | jq '.data | {pool: .pool.poolSize, active: .pool.activeProcesses, avail: .pool.availableProcesses, exec: .pool.totalExecutions, rssMB: (.node.rss/1024/1024|floor)}'
```

---
Use this endpoint liberally during performance and leak investigations; it's read-only and lightweight.
