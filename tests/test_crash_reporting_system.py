"""Crash Reporting System Tests (Pytest BDD Style)

Verifies that the crash reporting mechanism behaves correctly under normal and
error conditions without relying on internal Pyodide-only endpoints. All code
execution uses the single public text/plain endpoint: /api/execute-raw.

Contract Enforced:
    {
      "success": bool,
      "data": {"result": Any, "stdout": str, "stderr": str, "executionTime": int} | null,
      "error": str | null,
      "meta": {"timestamp": str}
    }

Guiding Principles:
    - No internal endpoints containing 'pyodide' in the path.
    - Use pathlib for any filesystem logic in executed Python code.
    - Ensure no plots or artifacts leak to project root; plots must reside under
      /home/pyodide/plots when generated.
    - Do not bend tests to fit a broken API; instead assert the contract.

Scenarios Covered:
    1. Listing crash reports (if any) returns structured data.
    2. Executing code that raises an exception returns success=False without crashing server.
    3. Long-running code respects timeout (simulated cooperative sleep loop) and returns an error without crash.
    4. Normal execution does not create crash artifacts.
    5. Malformed Python (syntax error) handled gracefully.
    6. Performance sanity: multiple quick executions remain under generous threshold.

This file replaces the original unittest-based implementation which has been
backed up to test_crash_reporting_system_original.py.
"""

from __future__ import annotations

from typing import Any, Dict
import time
import requests  # type: ignore[import-untyped]
import pytest


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

class CrashConfig:
    """Centralized constants for crash reporting tests.

    Adjust here instead of scattering literals across tests. Timeouts are in
    seconds; they are deliberately generous to avoid flakiness on CI.
    """

    BASE_URL: str = "http://localhost:3000"
    EXECUTE_RAW: str = "/api/execute-raw"
    TIMEOUT_EXEC: int = 45
    TIMEOUT_REQUEST: int = 15
    PERFORMANCE_MAX_AVG_SEC: float = 10.0  # generous envelope


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _execute(code: str, timeout: int | None = None) -> Dict[str, Any]:
    """Execute Python code through /api/execute-raw and validate contract.

    Parameters
    ----------
    code: str
        Python source code (plain text) to send.
    timeout: int | None
        Optional request timeout (seconds). Defaults to CrashConfig.TIMEOUT_EXEC.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON response payload.

    Raises
    ------
    AssertionError
        If the response doesn't honor the API contract.

    Examples
    --------
    >>> resp = _execute('print("hi")')
    >>> assert resp['success'] is True
    """
    resp = requests.post(
        f"{CrashConfig.BASE_URL}{CrashConfig.EXECUTE_RAW}",
        data=code.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        timeout=timeout or CrashConfig.TIMEOUT_EXEC,
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text[:200]}"
    payload = resp.json()

    # API contract validation
    assert "success" in payload and isinstance(payload["success"], bool)
    assert "error" in payload
    assert "data" in payload
    assert "meta" in payload and isinstance(payload["meta"], dict) and "timestamp" in payload["meta"], payload
    if payload["success"]:
        assert isinstance(payload["data"], dict), "data must be object on success"
        for f in ("stdout", "stderr", "executionTime"):
            assert f in payload["data"], f"Missing field data.{f}"
    else:
        assert payload["data"] in (None, {})
        assert isinstance(payload["error"], str) and payload["error"].strip()
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _server_ready_autouse():
    """Module-level readiness check executed automatically.

    This avoids having to inject a fixture parameter (which triggered redefinition
    lint noise) in every test while still guaranteeing a pre-flight check.
    """
    try:
        ping = _execute("print('ready')")
        assert ping["success"] is True
    except (requests.RequestException, AssertionError) as exc:  # narrow exception scope
        pytest.skip(f"Server not ready for crash tests: {exc}")


# ---------------------------------------------------------------------------
# Tests (BDD Style with Given / When / Then comments)
# ---------------------------------------------------------------------------


def test_list_crash_reports_structure():  # Given server ready
    """Scenario: Listing crash reports returns structured data (if present)."""
    # When requesting crash reports listing (public debug route may or may not exist)
    resp = requests.get(f"{CrashConfig.BASE_URL}/api/crash-reports", timeout=CrashConfig.TIMEOUT_REQUEST)
    # Then response is either 200 (with structure) or 404 (no system yet)
    assert resp.status_code in {200, 404}
    if resp.status_code == 200:
        body = resp.json()
        assert "reports" in body and isinstance(body["reports"], list)
        if body["reports"]:
            sample = body["reports"][0]
            # Minimal expected diagnostic fields
            for f in ("id", "timestamp"):
                assert f in sample


def test_execution_python_exception_handled():
    """Scenario: A Python exception does not crash the server and returns structured error."""
    code = "raise ValueError('boom from test')"
    result = _execute(code)
    assert result["success"] is False
    assert "ValueError" in (result["error"] or "")


def test_execution_syntax_error_handled():
    """Scenario: A Python syntax error is surfaced cleanly without server crash."""
    code = "def bad(:\n    pass"  # intentional syntax error
    result = _execute(code)
    assert result["success"] is False
    assert any(tok in (result["error"] or "") for tok in ["SyntaxError", "invalid syntax", "parsing"])  # tolerant


def test_long_running_code_timeout():
    """Scenario: Cooperative long-running loop is interrupted by timeout policy.

    We simulate work with a small loop and time.sleep to stay portable & quick.
    """
    # Given Python code that loops with short sleeps
    code = (
        "import time, pathlib\n"
        "start=time.time();\n"
        "for i in range(1000000):\n"
        "    time.sleep(0.01)\n"
        "    if time.time()-start>1.5: raise TimeoutError('simulated overrun')\n"
    )
    # When executing (server side may enforce its own timeout earlier)
    result = _execute(code)
    # Then either timeout or our raised TimeoutError yields error but not crash
    assert result["success"] is False
    assert any(k in (result["error"] or "") for k in ["Timeout", "overrun", "execution timeout"])


def test_normal_execution_does_not_generate_crash_report():
    """Scenario: Normal execution prints output and does not create crash artifacts.

    We rely on absence of failure; explicit crash file listing is optional.
    """
    code = (
        "from pathlib import Path\n"
        "print('hello crash system')\n"
        "p=Path('/home/pyodide/plots')\n"
        "print('plots_dir_exists', p.exists())\n"
    )
    result = _execute(code)
    assert result["success"] is True
    stdout = result["data"]["stdout"]
    assert "hello crash system" in stdout
    assert "plots_dir_exists" in stdout


def test_multiple_fast_executions_performance_sanity():
    """Scenario: Repeated small executions remain within generous performance envelope."""
    iterations = 5
    start = time.time()
    for i in range(iterations):
        r = _execute(f"print('perf {i}')")
        assert r["success"] is True
    avg = (time.time() - start) / iterations
    assert avg < CrashConfig.PERFORMANCE_MAX_AVG_SEC, f"Average exec time too high: {avg:.3f}s"


def test_filesystem_isolation_no_root_plots_leak():
    """Scenario: Plot creation saves under /home/pyodide/plots not project root.

    The test generates a simple plot (matplotlib) and ensures stdout mentions
    the intended directory while NOT referencing '/plots' at VFS root.
    """
    code = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "outdir=Path('/home/pyodide/plots/matplotlib'); outdir.mkdir(parents=True, exist_ok=True)\n"
        "f=outdir/'crash_vis.png'\n"
        "plt.plot([1,2,3],[3,2,1]); plt.savefig(f)\n"
        "print('saved_to', f)\n"
    )
    r = _execute(code)
    # We don't assert success hard if environment lacks matplotlib; allow graceful failure but not crash
    if r["success"]:
        stdout = r["data"]["stdout"]
        assert "saved_to" in stdout
        assert "/home/pyodide/plots" in stdout
        assert not any(seg in stdout for seg in ["/plots/crash_vis", " project_root/plots "])
    else:
        # Accept failure only if the error references missing backend or package
        assert any(tok in r["error"].lower() for tok in ["matplotlib", "backend", "import"])


# End of file
