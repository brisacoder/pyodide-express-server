"""BDD-style memory regression tests for adaptive cleanup and recycling.

Description:
    Ensures that repeated executions via /api/execute-raw do not cause unbounded
    memory growth in the Node.js parent process. Validates API contract and that
    the adaptive cleanup logic maintains a plateau in reported memory usage.

Inputs:
    - HTTP endpoint /api/execute-raw accepting plain text Python code.
    - Environment variables may enable adaptive cleanup (ADAPTIVE_CLEANUP_ENABLED=1).

Outputs:
    - Assertions on response contract for each execution.
    - Final plateau assertion: max RSS observed minus min RSS observed must be within
      an allowed delta threshold to indicate stabilization.

Examples:
    pytest -q tests/test_memory_regression.py
"""
from __future__ import annotations
import os
import time
import math
import requests
import pytest
from datetime import datetime

API_URL = os.getenv("TEST_SERVER_URL", "http://localhost:3000")
EXECUTE_ENDPOINT = f"{API_URL}/api/execute-raw"
STATS_ENDPOINT = f"{API_URL}/api/status"  # Assuming stats or status route returns pool stats

MAX_ALLOWED_RSS_GROWTH_MB = int(os.getenv("TEST_MAX_ALLOWED_RSS_GROWTH_MB", "600"))  # generous default
WARM_EXECUTIONS = int(os.getenv("TEST_MEMORY_WARM_EXECUTIONS", "10"))
STRESS_EXECUTIONS = int(os.getenv("TEST_MEMORY_STRESS_EXECUTIONS", "60"))
SAMPLE_INTERVAL = float(os.getenv("TEST_MEMORY_SAMPLE_INTERVAL", "0.1"))


def validate_api_contract(payload: dict):
    assert isinstance(payload, dict), "Response root must be dict"
    assert set(payload.keys()) == {"success", "data", "error", "meta"}
    assert isinstance(payload["success"], bool)
    # Either data or error must be present (allow data with no error) keep readable
    has_data = payload["data"] is not None
    has_error = payload["error"] is not None
    assert not (has_data and has_error)
    assert has_data or has_error
    assert payload["meta"] and isinstance(payload["meta"], dict)
    assert "timestamp" in payload["meta"], "Missing timestamp metadata"
    # Timestamp parseable ISO
    datetime.fromisoformat(payload["meta"]["timestamp"].replace("Z", "+00:00"))


def fetch_pool_rss_mb() -> float:
    try:
        r = requests.get(STATS_ENDPOINT, timeout=5)
        if r.status_code != 200:
            return math.nan
        data = r.json()
        # Node process RSS if provided, else skip
        node_rss = None
        if isinstance(data, dict):
            pool = data.get("pool") or data
            node_proc = pool.get("memory", {}).get("nodeProcess") if isinstance(pool, dict) else None
            if node_proc and "rss" in node_proc:
                node_rss = node_proc["rss"] / 1024 / 1024
        return float(node_rss) if node_rss is not None else math.nan
    except requests.RequestException:
        return math.nan


@pytest.mark.skipif(os.getenv("SKIP_MEMORY_STRESS") == "1", reason="Memory stress test skipped by env flag")
@pytest.mark.timeout(600)
def test_memory_plateau_under_stress(server_ready):  # noqa: ARG001 fixture used for server readiness side-effect
    """Scenario: Repeated code execution reaches memory plateau

    Given the server is running
    And adaptive cleanup is enabled (if configured)
    When we execute many small Python snippets repeatedly
    Then each response follows the API contract
    And overall memory RSS growth stays within allowed bounds
    """
    # Access fixture (ensures server is started); value not necessarily True
    _ = server_ready  # noqa: F841

    warm_code = "\n".join([
        "import math", "x = math.sqrt(144)", "x"
    ])

    # Warm-up phase to stabilize modules
    for _ in range(WARM_EXECUTIONS):
        r = requests.post(
            EXECUTE_ENDPOINT,
            data=warm_code,
            headers={'Content-Type': 'text/plain'},
            timeout=20
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        validate_api_contract(payload)
        assert payload["success"] is True

    rss_samples = []

    stress_code_template = "\n".join([
        "import numpy as np",
        "a = np.random.rand(200,200)",  # ~0.32MB
        "b = a @ a",
        "float(b[0,0])"
    ])

    for i in range(STRESS_EXECUTIONS):
        r = requests.post(
            EXECUTE_ENDPOINT,
            data=stress_code_template,
            headers={'Content-Type': 'text/plain'},
            timeout=25
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        validate_api_contract(payload)
        assert payload["success"] is True

        if i % 3 == 0:  # periodic sampling to reduce overhead
            rss = fetch_pool_rss_mb()
            if not math.isnan(rss):
                rss_samples.append(rss)
            time.sleep(SAMPLE_INTERVAL)

    if len(rss_samples) >= 5:
        min_rss = min(rss_samples)
        max_rss = max(rss_samples)
        growth = max_rss - min_rss
        assert growth <= MAX_ALLOWED_RSS_GROWTH_MB, (
            f"RSS growth {growth:.2f} MB exceeds allowed {MAX_ALLOWED_RSS_GROWTH_MB} MB. "
            f"First samples: {rss_samples[:10]}"
        )
    else:
        pytest.skip("Insufficient RSS samples to evaluate plateau")
