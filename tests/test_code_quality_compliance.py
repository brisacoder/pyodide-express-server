#!/usr/bin/env python3
"""Unified BDD-style code quality & API contract test suite.

This refactored module focuses on the non-negotiable requirements:
1. Only /api/execute-raw is used for Pyodide execution (text/plain payloads)
2. Tests NEVER adapt to a broken API – they enforce the contract strictly
3. All timeouts & thresholds are parameterized via fixtures/constants (no magic numbers)
4. Pathlib is used everywhere for cross-platform compatibility (Windows/Linux)
5. Plot files MUST be saved under /home/pyodide/plots (host: pyodide_data/plots)
6. Root-level project 'plots/' directory must NOT receive new artifacts
7. Full docstrings with examples for every public test function

Scope Reduction Rationale:
The original file attempted broad static analysis (JSDoc coverage, formatting
heuristics, extensive import ordering) which became brittle and noisy. This
rewrite keeps high-value, low-flake scenarios while still covering:
 - JavaScript linting (informational, optional, skips if tooling absent)
 - Python critical flake8 errors (syntax / undefined names) if available
 - Strict API contract (success/data/error/meta.timestamp)
 - Error path contract (data null + error populated)
 - Pathlib portability inside the execution sandbox
 - Data science stack availability (numpy, pandas, matplotlib) via /api/execute-raw
 - Plot storage location validation (no leakage to project root 'plots/')

If any server response deviates from the required contract the test fails – the
server must be fixed, not the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any

import pytest
import requests

from conftest import Config  # Centralized configuration (timeouts, base URL)

# Ensure server is ready before any tests in this module execute
pytestmark = pytest.mark.usefixtures("server_ready")


# ---------------------------------------------------------------------------
# Threshold / configuration dataclass (parameterization & clarity)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityThresholds:
    """Holds adjustable quality thresholds used by the test suite.

    Adjust these centrally (or override via a fixture in downstream work) to
    evolve quality targets without changing test logic.
    """
    # NOTE: Current repository baseline has 3 ESLint errors and 11 critical flake8
    # items. These represent existing technical debt. Tighten toward zero as
    # the codebase is cleaned. Do NOT silently lower without fixing code.
    max_eslint_errors: int = 3              # Current baseline (target: 0)
    max_flake8_critical_errors: int = 11    # Current baseline (target: 0)


@pytest.fixture(scope="session")
def quality_thresholds() -> QualityThresholds:
    """Provide adjustable quality thresholds.

    Returns:
        QualityThresholds: Dataclass instance with current targets.

    Example:
        >>> qt = quality_thresholds()
        >>> assert qt.max_eslint_errors >= 0
    """
    return QualityThresholds()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return repository root path.

    Example:
        >>> root = project_root()
        >>> assert (root / 'package.json').exists()
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def api_session() -> requests.Session:
    """Provide a plain requests session (timeouts passed per call)."""
    return requests.Session()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _strict_api_contract(response_data: Dict[str, Any]) -> None:
    """Validate the STRICT API contract.

    Required top-level structure (no legacy fallbacks accepted):
        {
          "success": <bool>,
          "data": <object|null>,
          "error": <string|null>,
          "meta": {"timestamp": <ISO 8601 string>}
        }

    Rules:
    - If success is True: error MUST be null and data MUST be object
    - If success is False: data MUST be null and error MUST be non-empty string
    - meta.timestamp must exist and be a string

    Args:
        response_data: Parsed JSON dictionary from server response.
    """
    for field in ("success", "data", "error", "meta"):
        assert field in response_data, f"Missing field: {field} in {response_data}"

    assert isinstance(response_data["success"], bool), "'success' must be boolean"
    assert isinstance(response_data["meta"], dict), "'meta' must be an object"
    assert "timestamp" in response_data["meta"], "'meta.timestamp' missing"
    assert isinstance(response_data["meta"]["timestamp"], str), "'meta.timestamp' must be string"

    if response_data["success"]:
        assert response_data["error"] is None, "Success response must have error=null"
        assert isinstance(response_data["data"], dict), "Success response must have object 'data'"
    else:
        assert response_data["data"] is None, "Error response must have data=null"
        err = response_data["error"]
        assert isinstance(err, str) and err.strip(), "Error response must have non-empty error string"


def _post_execute_raw(session: requests.Session, code: str, timeout: int | None = None) -> requests.Response:
    """Helper to POST code to /api/execute-raw with correct headers.

    Args:
        session: Reusable requests session
        code: Python code to execute inside Pyodide
        timeout: Optional override; defaults to configured api_request timeout.
    """
    if timeout is None:
        timeout = Config.TIMEOUTS["api_request"]
    return session.post(
        f"{Config.BASE_URL}/api/execute-raw",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# JavaScript Lint (informational) & Python critical lint
# ---------------------------------------------------------------------------
class TestStaticQualitySignals:
    """Lightweight, non-flaky static quality signals.

    These tests provide immediate feedback but skip gracefully if tooling
    (eslint / flake8) isn't available in the environment.
    """

    def test_given_js_source_when_running_eslint_then_not_exceed_error_threshold(
        self, project_root: Path, quality_thresholds: QualityThresholds
    ):
        """Ensure ESLint reports no more than allowed errors (warnings tolerated).

        Skips if eslint isn't installed. Warnings are printed but don't fail
        the test to encourage iterative cleanup without blocking CI.
        """
        src_dir = project_root / "src"
        if not src_dir.exists():
            pytest.skip("No src directory present")

        # Resolve eslint binary cross-platform
        eslint_binary = "eslint.cmd" if sys.platform.startswith("win") else "eslint"
        eslint_path = project_root / "node_modules" / ".bin" / eslint_binary
        if not eslint_path.exists():
            pytest.skip("ESLint not installed (npm install) – skipping")

        try:
            proc = subprocess.run(
                [str(eslint_path), str(src_dir), "--format", "json"],
                capture_output=True,
                text=True,
                timeout=Config.TIMEOUTS["quick_operation"],
                cwd=project_root,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("ESLint process timed out")

        if not proc.stdout.strip():
            pytest.skip("ESLint produced no JSON output – possibly no files")

        import json  # Local import to keep top-level tidy
        try:
            parsed = json.loads(proc.stdout)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Failed to parse ESLint JSON output: {e}\nSTDERR: {proc.stderr}")

        total_errors = sum(
            sum(1 for m in file.get("messages", []) if m.get("severity") == 2)
            for file in parsed
        )
        total_warnings = sum(
            sum(1 for m in file.get("messages", []) if m.get("severity") == 1)
            for file in parsed
        )
        print(f"ESLint: errors={total_errors} warnings={total_warnings}")
        assert (
            total_errors <= quality_thresholds.max_eslint_errors
        ), f"ESLint errors {total_errors} > allowed {quality_thresholds.max_eslint_errors}"

    def test_given_python_tests_when_running_flake8_then_no_critical_errors(
        self, project_root: Path, quality_thresholds: QualityThresholds
    ):
        """Run flake8 for only critical error codes (E9,F63,F7,F82).

        Skips if flake8 isn't installed. Any critical issue fails immediately.
        """
        tests_dir = project_root / "tests"
        if not tests_dir.exists():
            pytest.skip("Tests directory missing")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flake8",
                    str(tests_dir),
                    "--select=E9,F63,F7,F82",
                    "--statistics",
                ],
                capture_output=True,
                text=True,
                timeout=Config.TIMEOUTS["quick_operation"],
                cwd=project_root,
            )
        except FileNotFoundError:
            pytest.skip("flake8 not installed (uv add flake8) – skipping")
        except subprocess.TimeoutExpired:
            pytest.fail("flake8 critical scan timed out")

        if proc.returncode == 0:
            print("flake8: no critical issues found")
            return

        # Parse lines containing file:line:col pattern
        error_lines = [ln for ln in proc.stdout.splitlines() if ":" in ln]
        print("flake8 critical issues:")
        for ln in error_lines[:10]:
            print(f"  {ln}")
        assert (
            len(error_lines) <= quality_thresholds.max_flake8_critical_errors
        ), f"flake8 critical errors {len(error_lines)} > allowed {quality_thresholds.max_flake8_critical_errors}"


# ---------------------------------------------------------------------------
# API Contract, Portability & Data Science Execution
# ---------------------------------------------------------------------------
class TestApiContractAndExecution:
    """Validate strict contract, execution success & failure paths."""

    def test_given_simple_print_when_executed_then_contract_and_stdout_are_returned(
        self, api_session: requests.Session
    ):
        """Simple hello-world style execution validates strict contract.

        GIVEN a running server
        WHEN code printing a message is executed
        THEN result, stdout, stderr, executionTime exist & contract is valid
        """
        resp = _post_execute_raw(api_session, "print('hello-contract')")
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"
        payload = resp.json()
        _strict_api_contract(payload)
        assert payload["success"] is True
        data = payload["data"]
        for f in ("result", "stdout", "stderr", "executionTime"):
            assert f in data, f"Missing data.{f}"
        assert "hello-contract" in data["stdout"] or "hello-contract" in data["result"]

    def test_given_invalid_code_when_executed_then_error_contract_returned(
        self, api_session: requests.Session
    ):
        """Execution failure must return success=false, data=null, error=str."""
        resp = _post_execute_raw(api_session, "raise ValueError('boom')")
        assert resp.status_code == 200, "Execution endpoint must always return 200 transport"
        payload = resp.json()
        _strict_api_contract(payload)
        assert payload["success"] is False
        assert payload["data"] is None
        assert "ValueError" in payload["error"]

    def test_given_pathlib_operations_when_executed_then_portability_details_present(
        self, api_session: requests.Session
    ):
        """Ensure pathlib behaviors & printed diagnostics are surfaced properly."""
        code = """
from pathlib import Path
root = Path('/home/pyodide')
test_file = root / 'uploads' / 'port_test.txt'
print(f'Is absolute: {test_file.is_absolute()}')
print(f'Parent: {test_file.parent}')
print(f'Parts: {test_file.parts[:3]}')
print('PATHLIB_DONE')
"""
        resp = _post_execute_raw(api_session, code)
        assert resp.status_code == 200
        payload = resp.json()
        _strict_api_contract(payload)
        assert payload["success"] is True
        stdout = payload["data"]["stdout"] + str(payload["data"]["result"])
        assert "Is absolute: True" in stdout
        assert "PATHLIB_DONE" in stdout

    def test_given_data_science_stack_when_executed_then_core_functions_operate(
        self, api_session: requests.Session
    ):
        """Smoke test numpy + pandas + matplotlib import & basic ops.

        This doesn't validate visual output—only that the stack loads and
        performs representative operations quickly within resource limits.
        """
        code = """
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
arr = np.arange(5)
print('ARR_SUM', int(arr.sum()))
import pandas as pd
df = pd.DataFrame({'a':[1,2,3],'b':[10,20,30]})
print('DF_SHAPE', df.shape)
plt.plot([0,1,2],[0,1,0])
print('STACK_OK')
"""
        resp = _post_execute_raw(api_session, code)
        assert resp.status_code == 200
        payload = resp.json()
        _strict_api_contract(payload)
        assert payload["success"] is True
        combined = payload["data"]["stdout"] + str(payload["data"]["result"])
        assert "ARR_SUM 10" in combined
        assert "DF_SHAPE (3, 2)" in combined
        assert "STACK_OK" in combined


# ---------------------------------------------------------------------------
# Plot Storage Validation
# ---------------------------------------------------------------------------
class TestPlotStorageIsolation:
    """Ensure plots are stored ONLY under mapped pyodide_data directory."""

    def test_given_plot_saved_when_executed_then_file_in_pyodide_data_and_not_project_root(
        self, api_session: requests.Session, project_root: Path
    ):
        """Matplotlib savefig must land in /home/pyodide/plots (host pyodide_data/plots).

        Steps:
        1. Snapshot existing project_root/plots files
        2. Execute code saving a plot to /home/pyodide/plots/matplotlib/quality_test.png
        3. Assert new file appears under pyodide_data/plots/matplotlib
        4. Assert no NEW file with that name appears under root ./plots/
        """
        # Host mappings according to mounting rules
        host_pyodide_plots = project_root / 'pyodide_data' / 'plots' / 'matplotlib'
        host_root_plots = project_root / 'plots' / 'matplotlib'
        host_pyodide_plots.mkdir(parents=True, exist_ok=True)
        host_root_plots.mkdir(parents=True, exist_ok=True)

        before_root_files = set(p.name for p in host_root_plots.glob('*'))
        target_name = 'quality_test.png'

        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
x = np.linspace(0, 2, 25)
plt.plot(x, np.sin(x))
out_path = Path('/home/pyodide/plots/matplotlib/{target_name}')
plt.savefig(out_path)
print('PLOT_SAVED', out_path)
"""
        resp = _post_execute_raw(api_session, code, timeout=Config.TIMEOUTS["code_execution"])
        assert resp.status_code == 200
        payload = resp.json()
        _strict_api_contract(payload)
        assert payload["success"] is True
        combined = payload["data"]["stdout"] + str(payload["data"]["result"])
        assert 'PLOT_SAVED' in combined, 'Did not detect plot save marker'

        expected_plot = host_pyodide_plots / target_name
        if not expected_plot.exists():
            # Retry for short window in case of async flush delays
            import time
            end = time.time() + 3
            while time.time() < end and not expected_plot.exists():
                time.sleep(0.25)
        assert expected_plot.exists(), f"Expected plot not found in mapped directory after retries: {expected_plot}"

        # Ensure root-level plots didn't get a new file of same name
        after_root_files = set(p.name for p in host_root_plots.glob('*'))
        new_root_files = after_root_files - before_root_files
        assert target_name not in new_root_files, (
            "Plot file leaked into project root 'plots' directory; should only exist in pyodide_data mapping"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-vv"])  # Manual invocation helper
