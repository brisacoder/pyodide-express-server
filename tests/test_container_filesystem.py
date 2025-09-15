#!/usr/bin/env python3
"""
Pytest BDD Container Filesystem Tests for Pyodide Express Server.

This test suite validates containerized Pyodide execution maintains full filesystem
compatibility using pytest with BDD (Behavior-Driven Development) patterns.

Key Features:
- BDD Given-When-Then structure for all test scenarios
- Comprehensive API contract validation per requirements
- Cross-platform pathlib usage for all file operations
- Only uses /api/execute-raw endpoint (no internal pyodide APIs)
- Parameterized constants and fixtures for maintainability
- Full docstrings with inputs, outputs, and examples

API Contract Validation:
{
  "success": true | false,
  "data": {
    "result": <execution_output>,
    "stdout": <captured_stdout>,
    "stderr": <captured_stderr>,
    "executionTime": <milliseconds>
  } | null,
  "error": <string> | null,
  "meta": { "timestamp": <ISO_string> }
}

Test Categories:
- Basic container execution validation
- Matplotlib plot generation and filesystem persistence
- Complex multi-plot dashboard creation
- Environment and package verification
- Error handling and timeout scenarios
- Security and resource validation
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pytest
import requests


# ==================== TEST CONFIGURATION CONSTANTS ====================


class TestConfig:
    """Centralized test configuration constants for maintainability."""

    # Server configuration
    BASE_URL: str = "http://localhost:3000"
    API_ENDPOINT: str = "/api/execute-raw"
    HEALTH_ENDPOINT: str = "/health"

    # Timeout values (seconds)
    TIMEOUTS = {
        "health_check": 10,
        "basic_execution": 30,
        "plot_creation": 45,
        "complex_dashboard": 60,
        "environment_check": 30,
        "api_request": 30,
    }

    # File size expectations (bytes)
    FILE_SIZES = {
        "min_plot_size": 1000,
        "min_dashboard_size": 50000,
        "max_reasonable_size": 10_000_000,
    }

    # Test data patterns
    PLOT_DIMENSIONS = {
        "simple_plot": (8, 6),
        "dashboard": (12, 10),
        "default_dpi": 150,
    }

    # Expected environment packages
    REQUIRED_PACKAGES = [
        "numpy",
        "pandas",
        "matplotlib",
    ]

    # Virtual filesystem paths
    FILESYSTEM_PATHS = [
        "/plots",
        "/uploads",
        "/logs",
        "/plots/matplotlib",
    ]


class APIContract:
    """API contract validation utilities."""

    @staticmethod
    def validate_response_structure(response_data: Dict[str, Any]) -> bool:
        """
        Validate API response follows exact contract structure.

        Args:
            response_data: JSON response from API

        Returns:
            bool: True if structure is valid

        Example:
            >>> response = {"success": True, "data": {...}, "error": None, "meta": {"timestamp": "..."}}
            >>> APIContract.validate_response_structure(response)
            True
        """
        required_keys = {"success", "data", "error", "meta"}
        if not all(key in response_data for key in required_keys):
            return False

        # Validate success is boolean
        if not isinstance(response_data["success"], bool):
            return False

        # Validate meta has timestamp
        if not isinstance(response_data.get("meta"), dict):
            return False
        if "timestamp" not in response_data["meta"]:
            return False

        # If success=True, data should not be null
        if response_data["success"] and response_data["data"] is None:
            return False

        # If success=False, error should not be null
        if not response_data["success"] and response_data["error"] is None:
            return False

        return True

    @staticmethod
    def validate_execution_data(data: Optional[Dict[str, Any]]) -> bool:
        """
        Validate execution data structure contains required fields.

        Args:
            data: Data section from API response

        Returns:
            bool: True if data structure is valid

        Example:
            >>> data = {"result": "output", "stdout": "...", "stderr": "", "executionTime": 123}
            >>> APIContract.validate_execution_data(data)
            True
        """
        if data is None:
            return False

        required_fields = {"result", "stdout", "stderr", "executionTime"}
        if not all(field in data for field in required_fields):
            return False

        # Validate executionTime is numeric
        if not isinstance(data["executionTime"], (int, float)):
            return False

        # Validate text fields are strings
        for field in ["result", "stdout", "stderr"]:
            if not isinstance(data[field], (str, type(None))):
                return False

        return True


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def api_client() -> requests.Session:
    """
    Create a configured requests session for API testing.

    Returns:
        requests.Session: Configured session with proper timeouts

    Example:
        >>> def test_something(api_client):
        ...     response = api_client.get("/health")
        ...     assert response.status_code == 200
    """
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "text/plain",
            "User-Agent": "pytest-container-filesystem-tests/1.0",
        }
    )
    return session


@pytest.fixture(scope="function")
def temp_file_tracker() -> Generator[List[Path], None, None]:
    """
    Track temporary files created during tests for automatic cleanup.

    Yields:
        List[Path]: List to append temporary file paths for cleanup

    Example:
        >>> def test_file_creation(temp_file_tracker):
        ...     temp_file = Path("/tmp/test.txt")
        ...     temp_file.write_text("test")
        ...     temp_file_tracker.append(temp_file)
        ...     # File will be automatically cleaned up
    """
    temp_files: List[Path] = []
    yield temp_files

    # Cleanup after test
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up {temp_file}: {e}")


@pytest.fixture(scope="function")
def server_health_check(api_client: requests.Session) -> None:
    """
    Verify server is healthy before running tests.

    Args:
        api_client: Configured requests session

    Raises:
        pytest.skip: If server is not available or unhealthy

    Example:
        >>> def test_something(server_health_check):
        ...     # Server is guaranteed to be healthy at this point
        ...     pass
    """
    try:
        response = api_client.get(
            f"{TestConfig.BASE_URL}{TestConfig.HEALTH_ENDPOINT}",
            timeout=TestConfig.TIMEOUTS["health_check"],
        )
        if response.status_code != 200:
            pytest.skip(f"Server unhealthy: HTTP {response.status_code}")
    except requests.RequestException as e:
        pytest.skip(f"Server not reachable: {e}")


@pytest.fixture(scope="function")
def unique_timestamp() -> int:
    """
    Generate unique timestamp for test isolation.

    Returns:
        int: Millisecond timestamp for unique test identification

    Example:
        >>> def test_plot_creation(unique_timestamp):
        ...     filename = f"plot_{unique_timestamp}.png"
        ...     # Use filename for test-specific files
    """
    return int(time.time() * 1000)


# ==================== BDD TEST SCENARIOS ====================


@pytest.mark.api
@pytest.mark.integration
def test_scenario_basic_container_python_execution(
    api_client: requests.Session, server_health_check: None
) -> None:
    """
    Scenario: Execute basic Python code in containerized Pyodide environment

    Given: A running Pyodide Express Server with container filesystem
    When: I send a simple Python print statement to /api/execute-raw
    Then: The response should follow API contract with success=True
    And: The stdout should contain the expected output
    And: The response time should be reasonable

    Args:
        api_client: Configured HTTP client session
        server_health_check: Ensures server is healthy before test

    Inputs:
        - Python code: "print('Hello from containerized Pyodide!')"
        - Content-Type: text/plain
        - Timeout: 30 seconds

    Expected Output:
        {
          "success": true,
          "data": {
            "result": "Hello from containerized Pyodide!\n",
            "stdout": "Hello from containerized Pyodide!\n",
            "stderr": "",
            "executionTime": <reasonable_milliseconds>
          },
          "error": null,
          "meta": {"timestamp": "<ISO_timestamp>"}
        }

    Example:
        This test validates the most basic container functionality by executing
        a simple print statement and verifying the API contract compliance.
    """
    # Given: A running server (verified by server_health_check fixture)
    execution_code = "print('Hello from containerized Pyodide!')"
    expected_output = "Hello from containerized Pyodide!"

    # When: I execute basic Python code via /api/execute-raw
    response = api_client.post(
        f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
        data=execution_code,
        timeout=TestConfig.TIMEOUTS["basic_execution"],
    )

    # Then: Response should be successful with proper HTTP status
    assert response.status_code == 200

    # And: Response should follow exact API contract
    result_data = response.json()
    assert APIContract.validate_response_structure(result_data)
    assert result_data["success"] is True
    assert result_data["error"] is None

    # And: Execution data should be properly structured
    assert APIContract.validate_execution_data(result_data["data"])

    # And: Output should contain expected content
    execution_data = result_data["data"]
    assert expected_output in execution_data["stdout"]
    assert expected_output in execution_data["result"]
    assert execution_data["stderr"] == "" or execution_data["stderr"] is None

    # And: Execution time should be reasonable (under 10 seconds)
    assert execution_data["executionTime"] < 10000


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.matplotlib
def test_scenario_container_matplotlib_plot_creation_and_persistence(
    api_client: requests.Session,
    server_health_check: None,
    temp_file_tracker: List[Path],
    unique_timestamp: int,
) -> None:
    """
    Scenario: Create matplotlib plot in container with filesystem persistence

    Given: A containerized Pyodide environment with matplotlib support
    When: I execute Python code that creates and saves a matplotlib plot
    Then: The plot should be successfully created and saved to filesystem
    And: The API response should indicate success with proper contract
    And: The generated file should exist and have reasonable size
    And: All Python code should use pathlib for cross-platform compatibility

    Args:
        api_client: HTTP client for API requests
        server_health_check: Server availability validation
        temp_file_tracker: File cleanup tracking
        unique_timestamp: Unique identifier for test isolation

    Inputs:
        - Matplotlib plot creation code using pathlib
        - Plot dimensions: 8x6 inches at 150 DPI
        - Target path: /plots/matplotlib/container_test_{timestamp}.png
        - Timeout: 45 seconds

    Expected Output:
        - API success response with execution data
        - Plot file created in expected location
        - File size > 1KB indicating actual plot content
        - Console output confirming file creation and size

    Example:
        Creates a sine wave plot using numpy and matplotlib, saves it to the
        mounted filesystem, and verifies both API response and file persistence.
    """
    # Given: Server is available (via fixture) and matplotlib environment ready
    plot_filename = f"container_test_{unique_timestamp}.png"
    host_plot_path = Path(f"plots/matplotlib/{plot_filename}")
    temp_file_tracker.append(host_plot_path)

    # When: I execute matplotlib plot creation code using pathlib
    plot_creation_code = f"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for container compatibility
import matplotlib.pyplot as plt
import numpy as np

# Generate test data using numpy
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot with pathlib-based file handling
plt.figure(figsize={TestConfig.PLOT_DIMENSIONS["simple_plot"]})
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Container Test Plot - {unique_timestamp}')
plt.xlabel('X values')
plt.ylabel('sin(X)')
plt.grid(True, alpha=0.3)
plt.legend()

# Use pathlib for cross-platform file operations
plot_path = Path('/plots/matplotlib/{plot_filename}')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi={TestConfig.PLOT_DIMENSIONS["default_dpi"]}, bbox_inches='tight')
plt.close()

# Output file information for verification
print(f'Plot saved to: {{plot_path}}')
print(f'File exists: {{plot_path.exists()}}')
if plot_path.exists():
    print(f'File size: {{plot_path.stat().st_size}} bytes')
    print(f'Parent directory: {{plot_path.parent}}')
""".strip()

    response = api_client.post(
        f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
        data=plot_creation_code,
        timeout=TestConfig.TIMEOUTS["plot_creation"],
    )

    # Then: API response should indicate successful execution
    assert response.status_code == 200
    result_data = response.json()
    assert APIContract.validate_response_structure(result_data)
    assert result_data["success"] is True
    assert result_data["error"] is None

    # And: Execution data should be properly structured
    assert APIContract.validate_execution_data(result_data["data"])
    execution_data = result_data["data"]

    # And: Console output should confirm plot creation
    stdout_content = execution_data["stdout"]
    assert "Plot saved to:" in stdout_content
    assert "File exists: True" in stdout_content
    assert "File size:" in stdout_content

    # And: Plot file should exist on host filesystem
    # (Allow some time for filesystem sync in container environments)
    time.sleep(1)
    assert host_plot_path.exists(), f"Plot file not found on host: {host_plot_path}"

    # And: File should have reasonable size indicating actual plot content
    file_size = host_plot_path.stat().st_size
    assert (
        file_size > TestConfig.FILE_SIZES["min_plot_size"]
    ), f"Plot file too small: {file_size} bytes (expected > {TestConfig.FILE_SIZES['min_plot_size']})"
    assert (
        file_size < TestConfig.FILE_SIZES["max_reasonable_size"]
    ), f"Plot file suspiciously large: {file_size} bytes"


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.matplotlib
@pytest.mark.slow
def test_scenario_container_complex_dashboard_creation(
    api_client: requests.Session,
    server_health_check: None,
    temp_file_tracker: List[Path],
    unique_timestamp: int,
) -> None:
    """
    Scenario: Create complex multi-plot dashboard in container environment

    Given: A containerized Pyodide environment with full matplotlib/numpy support
    When: I execute complex Python code creating a 2x2 subplot dashboard
    Then: All subplots should be successfully created and rendered
    And: The dashboard file should be saved with substantial size
    And: The API should respond with success and proper execution data
    And: All file operations should use pathlib for portability

    Args:
        api_client: HTTP session for API communication
        server_health_check: Server readiness verification
        temp_file_tracker: File cleanup management
        unique_timestamp: Test isolation identifier

    Inputs:
        - Complex matplotlib code with 4 subplots:
          * Trigonometric functions (sin/cos)
          * Random scatter plot with colors
          * Bar chart with categories
          * Histogram of normal distribution
        - Dashboard dimensions: 12x10 inches
        - High DPI output: 150 DPI
        - Target: /plots/matplotlib/container_dashboard_{timestamp}.png
        - Extended timeout: 60 seconds

    Expected Output:
        - API success response with complete execution data
        - Dashboard file > 50KB (substantial multi-plot content)
        - Console confirmation of successful save operation
        - Proper error handling if any subplot fails

    Example:
        This test validates the container's ability to handle complex matplotlib
        operations including multiple subplots, different plot types, random data
        generation, and sophisticated styling options.
    """
    # Given: Server ready and environment capable of complex matplotlib operations
    dashboard_filename = f"container_dashboard_{unique_timestamp}.png"
    host_dashboard_path = Path(f"plots/matplotlib/{dashboard_filename}")
    temp_file_tracker.append(host_dashboard_path)

    # When: I execute complex dashboard creation code with pathlib
    dashboard_creation_code = f"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for container
import matplotlib.pyplot as plt
import numpy as np

# Create complex 2x2 dashboard using matplotlib subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize={TestConfig.PLOT_DIMENSIONS["dashboard"]})
fig.suptitle('Container Dashboard - {unique_timestamp}', fontsize=16)

# Subplot 1: Trigonometric Functions
x = np.linspace(0, 4*np.pi, 200)
ax1.plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
ax1.plot(x, np.cos(x), 'r-', linewidth=2, label='cos(x)')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X values')
ax1.set_ylabel('Y values')

# Subplot 2: Random Scatter Plot
np.random.seed(42)  # Reproducible randomness for testing
x_rand = np.random.randn(100)
y_rand = np.random.randn(100)
colors = np.random.rand(100)
scatter = ax2.scatter(x_rand, y_rand, c=colors, alpha=0.7, cmap='viridis')
ax2.set_title('Random Scatter Plot')
ax2.set_xlabel('X random')
ax2.set_ylabel('Y random')
plt.colorbar(scatter, ax=ax2)

# Subplot 3: Categorical Bar Chart
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [23, 45, 56, 78, 32]
bars = ax3.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'plum'])
ax3.set_title('Sample Bar Chart')
ax3.set_ylabel('Values')
ax3.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{value}',
             ha='center', va='bottom')

# Subplot 4: Normal Distribution Histogram
data = np.random.normal(0, 1, 1000)
counts, bins, patches = ax4.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
ax4.set_title('Normal Distribution Histogram')
ax4.set_xlabel('Value')
ax4.set_ylabel('Frequency')
ax4.axvline(0, color='red', linestyle='--', alpha=0.8, label='Mean')
ax4.legend()

# Apply tight layout for proper spacing
plt.tight_layout()

# Save dashboard using pathlib for cross-platform compatibility
dashboard_path = Path('/plots/matplotlib/{dashboard_filename}')
dashboard_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(dashboard_path, dpi={TestConfig.PLOT_DIMENSIONS["default_dpi"]}, 
           bbox_inches='tight', facecolor='white')
plt.close()

# Output verification information
print(f'Dashboard saved to: {{dashboard_path}}')
print(f'File exists: {{dashboard_path.exists()}}')
if dashboard_path.exists():
    file_size = dashboard_path.stat().st_size
    print(f'File size: {{file_size}} bytes')
    print(f'File size MB: {{file_size / 1024 / 1024:.2f}}')
    print(f'Parent directory exists: {{dashboard_path.parent.exists()}}')
""".strip()

    response = api_client.post(
        f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
        data=dashboard_creation_code,
        timeout=TestConfig.TIMEOUTS["complex_dashboard"],
    )

    # Then: API should respond successfully for complex operation
    assert response.status_code == 200
    result_data = response.json()
    assert APIContract.validate_response_structure(result_data)
    assert result_data["success"] is True
    assert result_data["error"] is None

    # And: Execution data should indicate successful completion
    assert APIContract.validate_execution_data(result_data["data"])
    execution_data = result_data["data"]

    # And: Console output should confirm dashboard creation
    stdout_content = execution_data["stdout"]
    assert "Dashboard saved to:" in stdout_content
    assert "File exists: True" in stdout_content
    assert "File size:" in stdout_content

    # And: Dashboard file should exist on host with substantial size
    time.sleep(1)  # Allow filesystem sync
    assert (
        host_dashboard_path.exists()
    ), f"Dashboard file not found: {host_dashboard_path}"

    # And: File size should indicate complex multi-plot content
    file_size = host_dashboard_path.stat().st_size
    assert (
        file_size > TestConfig.FILE_SIZES["min_dashboard_size"]
    ), f"Dashboard file too small: {file_size} bytes (expected > {TestConfig.FILE_SIZES['min_dashboard_size']})"
    assert (
        file_size < TestConfig.FILE_SIZES["max_reasonable_size"]
    ), f"Dashboard file suspiciously large: {file_size} bytes"


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.environment
def test_scenario_container_environment_and_package_verification(
    api_client: requests.Session, server_health_check: None
) -> None:
    """
    Scenario: Verify container environment has required packages and filesystem access

    Given: A containerized Pyodide environment
    When: I execute Python code to inspect the environment and available packages
    Then: All required packages should be available with proper versions
    And: Essential filesystem paths should be accessible
    And: Python version should meet minimum requirements
    And: Package import operations should complete successfully

    Args:
        api_client: HTTP client for API requests
        server_health_check: Server availability validation

    Inputs:
        - Environment inspection Python code
        - Package version queries for numpy, pandas, matplotlib
        - Filesystem accessibility tests using pathlib
        - Platform and Python version detection
        - Timeout: 30 seconds

    Expected Output:
        - Python version 3.11+ (Pyodide requirement)
        - All required packages with version information
        - Accessible filesystem paths with ✅ indicators
        - Successful import statements for all packages
        - Platform information for debugging

    Example:
        This test ensures the container environment is properly configured
        with all necessary data science packages and filesystem mounts.
    """
    # Given: Container environment is ready (via server_health_check fixture)

    # When: I execute comprehensive environment inspection code
    environment_inspection_code = f"""
import sys
import platform
from pathlib import Path
import importlib

# Core environment information
print(f"Python version: {{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}")
print(f"Platform: {{platform.platform()}}")
print(f"Python executable: {{sys.executable}}")
print(f"Python path entries: {{len(sys.path)}}")

# Test package availability and versions
required_packages = {TestConfig.REQUIRED_PACKAGES!r}
package_status = []

for package_name in required_packages:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {{package_name}} version: {{version}}")
        package_status.append(True)
    except ImportError as e:
        print(f"❌ {{package_name}} import failed: {{e}}")
        package_status.append(False)

# Test specific imports that are commonly used
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✅ All critical imports successful")
except ImportError as e:
    print(f"❌ Critical import failed: {{e}}")

# Test filesystem path accessibility using pathlib
print("\\nFilesystem accessibility test:")
test_paths = {TestConfig.FILESYSTEM_PATHS!r}
accessible_count = 0

for path_str in test_paths:
    path = Path(path_str)
    is_accessible = path.exists() and path.is_dir()
    status = "✅" if is_accessible else "❌"
    print(f"{{status}} {{path_str}}: exists={{path.exists()}}, is_dir={{path.is_dir()}}")
    if is_accessible:
        accessible_count += 1

print(f"\\nSummary:")
print(f"- Packages available: {{sum(package_status)}}/{{len(package_status)}}")
print(f"- Paths accessible: {{accessible_count}}/{{len(test_paths)}}")
print(f"- Environment ready: {{all(package_status) and accessible_count >= 3}}")

# Test pathlib file creation capability
temp_test_path = Path('/tmp/environment_test.txt')
try:
    temp_test_path.write_text('Environment test successful')
    print(f"✅ File creation test passed: {{temp_test_path.exists()}}")
    temp_test_path.unlink()  # Clean up
except Exception as e:
    print(f"❌ File creation test failed: {{e}}")
""".strip()

    response = api_client.post(
        f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
        data=environment_inspection_code,
        timeout=TestConfig.TIMEOUTS["environment_check"],
    )

    # Then: API should respond successfully
    assert response.status_code == 200
    result_data = response.json()
    assert APIContract.validate_response_structure(result_data)
    assert result_data["success"] is True
    assert result_data["error"] is None

    # And: Execution should complete with proper data structure
    assert APIContract.validate_execution_data(result_data["data"])
    execution_data = result_data["data"]
    stdout_content = execution_data["stdout"]

    # And: Python version should meet minimum requirements (3.11+ for Pyodide)
    assert "Python version: 3.1" in stdout_content, "Python version should be 3.11+"

    # And: All required packages should be available
    for package in TestConfig.REQUIRED_PACKAGES:
        assert (
            f"✅ {package} version:" in stdout_content
        ), f"Package {package} not available"

    # And: Critical imports should succeed
    assert (
        "✅ All critical imports successful" in stdout_content
    ), "Critical imports failed"

    # And: Essential filesystem paths should be accessible
    accessible_paths_count = (
        stdout_content.count("✅") - len(TestConfig.REQUIRED_PACKAGES) - 2
    )  # Subtract package checks and import check
    assert (
        accessible_paths_count >= 3
    ), f"Not enough accessible paths: {accessible_paths_count}"

    # And: File creation capability should work
    assert (
        "✅ File creation test passed:" in stdout_content
    ), "File creation test failed"

    # And: Environment should be reported as ready
    assert "Environment ready: True" in stdout_content, "Environment not ready"


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.error_handling
def test_scenario_api_contract_validation_for_error_conditions(
    api_client: requests.Session, server_health_check: None
) -> None:
    """
    Scenario: Validate API contract compliance during error conditions

    Given: A running Pyodide server
    When: I send Python code that generates various types of errors
    Then: The API should respond with proper error contract structure
    And: The success field should be false for all error cases
    And: The error field should contain descriptive error messages
    And: The data field should be null for failed executions
    And: The meta field should always contain a valid timestamp

    Args:
        api_client: HTTP session for API requests
        server_health_check: Server availability check

    Inputs:
        - Syntax error Python code
        - Import error Python code
        - Runtime error Python code
        - Various malformed requests

    Expected Output:
        - Consistent API contract structure for all error types
        - success: false in all error cases
        - error: non-null descriptive messages
        - data: null for failed executions
        - meta.timestamp: valid ISO timestamp

    Example:
        Tests that the server maintains API contract even when Python
        execution fails, ensuring consistent client experience.
    """
    # Given: Server is available (via fixture)

    error_test_cases = [
        {
            "name": "syntax_error",
            "code": "print('unclosed string",
            "expected_error_keywords": ["syntax", "error", "EOF"],
        },
        {
            "name": "import_error",
            "code": "import nonexistent_module_xyz123",
            "expected_error_keywords": ["import", "module", "not", "found"],
        },
        {
            "name": "runtime_error",
            "code": "x = 1 / 0",
            "expected_error_keywords": ["division", "zero"],
        },
        {
            "name": "name_error",
            "code": "print(undefined_variable_xyz)",
            "expected_error_keywords": ["name", "not", "defined"],
        },
    ]

    for test_case in error_test_cases:
        # When: I execute code that should generate an error
        response = api_client.post(
            f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
            data=test_case["code"],
            timeout=TestConfig.TIMEOUTS["basic_execution"],
        )

        # Then: Response should have proper HTTP status (200 even for execution errors)
        assert response.status_code == 200, f"Wrong HTTP status for {test_case['name']}"

        # And: Response should follow API contract structure
        result_data = response.json()
        assert APIContract.validate_response_structure(
            result_data
        ), f"Invalid API structure for {test_case['name']}"

        # And: Success should be false for execution errors
        assert (
            result_data["success"] is False
        ), f"Success should be false for {test_case['name']}"

        # And: Error field should contain descriptive message
        assert (
            result_data["error"] is not None
        ), f"Error field should not be null for {test_case['name']}"
        assert isinstance(
            result_data["error"], str
        ), f"Error should be string for {test_case['name']}"
        assert (
            len(result_data["error"]) > 0
        ), f"Error message should not be empty for {test_case['name']}"

        # And: Data field should be null for failed executions
        assert (
            result_data["data"] is None
        ), f"Data should be null for failed execution {test_case['name']}"

        # And: Meta should contain valid timestamp
        assert (
            "timestamp" in result_data["meta"]
        ), f"Missing timestamp for {test_case['name']}"
        timestamp = result_data["meta"]["timestamp"]
        assert isinstance(
            timestamp, str
        ), f"Timestamp should be string for {test_case['name']}"
        assert (
            len(timestamp) > 0
        ), f"Timestamp should not be empty for {test_case['name']}"

        # And: Error message should contain relevant keywords
        error_message = result_data["error"].lower()
        keyword_found = any(
            keyword.lower() in error_message
            for keyword in test_case["expected_error_keywords"]
        )
        assert (
            keyword_found
        ), f"Error message should contain relevant keywords for {test_case['name']}: {result_data['error']}"


@pytest.mark.api
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.timeout
def test_scenario_execution_timeout_and_resource_limits(
    api_client: requests.Session, server_health_check: None
) -> None:
    """
    Scenario: Validate server behavior under resource-intensive operations

    Given: A Pyodide server with execution limits
    When: I execute Python code with varying resource requirements
    Then: The server should handle operations within reasonable time limits
    And: Memory-intensive operations should complete without crashes
    And: All responses should maintain API contract compliance
    And: Execution times should be tracked and reported accurately

    Args:
        api_client: HTTP client for API communication
        server_health_check: Server readiness verification

    Inputs:
        - CPU-intensive computation (fibonacci calculation)
        - Memory allocation test with numpy arrays
        - File I/O operations using pathlib
        - Mathematical computations with matplotlib

    Expected Output:
        - All operations complete within timeout limits
        - Memory allocations succeed for reasonable sizes
        - Execution time tracking in response data
        - Proper error handling if limits exceeded

    Example:
        Tests server stability under various load conditions while
        ensuring the API contract remains consistent.
    """
    # Given: Server is ready and has resource management

    performance_test_cases = [
        {
            "name": "cpu_intensive_computation",
            "timeout": 30,
            "code": """
from pathlib import Path
import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# CPU-intensive calculation
start_time = time.time()
result = fibonacci(25)  # Reasonable size for testing
end_time = time.time()

print(f"Fibonacci(25) = {result}")
print(f"Computation time: {end_time - start_time:.3f} seconds")
print(f"CPU test completed successfully")
""".strip(),
        },
        {
            "name": "memory_allocation_test",
            "timeout": 45,
            "code": """
from pathlib import Path
import numpy as np
import time

# Memory allocation test
start_time = time.time()

# Create reasonably sized arrays for testing
arrays = []
for i in range(10):
    arr = np.random.rand(1000, 100)  # 100K elements each
    arrays.append(arr)
    if i % 3 == 0:
        print(f"Allocated array {i+1}: shape {arr.shape}")

# Test mathematical operations
combined = np.concatenate(arrays, axis=0)
result = np.mean(combined)
std_dev = np.std(combined)

end_time = time.time()

print(f"Total memory allocated: {len(arrays)} arrays")
print(f"Combined shape: {combined.shape}")
print(f"Mean: {result:.6f}")
print(f"Standard deviation: {std_dev:.6f}") 
print(f"Memory test time: {end_time - start_time:.3f} seconds")
print("Memory allocation test completed successfully")
""".strip(),
        },
        {
            "name": "file_io_operations",
            "timeout": 30,
            "code": """
from pathlib import Path
import json
import time

# File I/O performance test using pathlib
start_time = time.time()

# Create test data
test_data = {
    'numbers': list(range(1000)),
    'squares': [i**2 for i in range(1000)],
    'timestamp': time.time()
}

# Test file operations
temp_dir = Path('/tmp')
temp_dir.mkdir(exist_ok=True)

files_created = []
for i in range(5):
    file_path = temp_dir / f'test_file_{i}.json'
    file_path.write_text(json.dumps(test_data, indent=2))
    files_created.append(file_path)
    print(f"Created file {i+1}: {file_path.stat().st_size} bytes")

# Read back and verify
total_size = 0
for file_path in files_created:
    content = file_path.read_text()
    loaded_data = json.loads(content)
    total_size += file_path.stat().st_size
    # Clean up
    file_path.unlink()

end_time = time.time()

print(f"Files processed: {len(files_created)}")
print(f"Total data size: {total_size} bytes")
print(f"File I/O time: {end_time - start_time:.3f} seconds")
print("File I/O test completed successfully")
""".strip(),
        },
    ]

    for test_case in performance_test_cases:
        # When: I execute resource-intensive code
        start_request_time = time.time()

        response = api_client.post(
            f"{TestConfig.BASE_URL}{TestConfig.API_ENDPOINT}",
            data=test_case["code"],
            timeout=test_case["timeout"],
        )

        end_request_time = time.time()
        total_request_time = end_request_time - start_request_time

        # Then: Response should be successful within timeout
        assert response.status_code == 200, f"HTTP error for {test_case['name']}"

        # And: API contract should be maintained under load
        result_data = response.json()
        assert APIContract.validate_response_structure(
            result_data
        ), f"Invalid API structure for {test_case['name']}"

        # And: Execution should succeed for reasonable operations
        assert (
            result_data["success"] is True
        ), f"Execution failed for {test_case['name']}: {result_data.get('error')}"

        # And: Execution data should be properly structured
        assert APIContract.validate_execution_data(
            result_data["data"]
        ), f"Invalid execution data for {test_case['name']}"

        # And: Output should confirm successful completion
        execution_data = result_data["data"]
        assert (
            "completed successfully" in execution_data["stdout"]
        ), f"Test did not complete successfully for {test_case['name']}"

        # And: Execution time should be reasonable and tracked
        execution_time_ms = execution_data["executionTime"]
        assert (
            execution_time_ms > 0
        ), f"Execution time should be positive for {test_case['name']}"
        assert (
            execution_time_ms < test_case["timeout"] * 1000
        ), f"Execution time exceeded timeout for {test_case['name']}"

        # And: Total request time should be reasonable
        assert (
            total_request_time < test_case["timeout"]
        ), f"Total request time exceeded for {test_case['name']}"

        print(
            f"✅ Performance test passed: {test_case['name']} (exec: {execution_time_ms}ms, total: {total_request_time:.1f}s)"
        )


# ==================== TEST EXECUTION CONFIGURATION ====================

if __name__ == "__main__":
    """
    Run tests directly with pytest when executed as main module.

    Usage:
        python test_container_filesystem.py
        pytest test_container_filesystem.py -v
        pytest test_container_filesystem.py -m "api and integration"
        pytest test_container_filesystem.py -m "not slow"
    """
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
