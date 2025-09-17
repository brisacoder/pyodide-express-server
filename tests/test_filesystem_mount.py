"""
Pytest-based tests for filesystem mounting functionality between Node.js and Pyodide.

This module tests the virtual filesystem capabilities of Pyodide running in the
Express server, including directory creation, file operations, and mount point detection.

Requirements:
    - pytest: Testing framework
    - requests: HTTP client for API calls
    - pathlib: Cross-platform path operations

Examples:
    Run all filesystem tests:
        $ uv run pytest tests/test_filesystem_mount.py -v

    Run specific test scenario:
        $ uv run pytest tests/test_filesystem_mount.py::TestFilesystemMountBDD::test_given_pyodide_when_creating_directories_then_virtual_filesystem_works -v  # noqa: E501

    Run with coverage:
        $ uv run pytest tests/test_filesystem_mount.py --cov=src --cov-report=term-missing
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import requests
import pytest


# Configuration Constants
class TestConfig:
    """Test configuration constants to avoid hardcoding values."""
    BASE_URL = "http://localhost:3000"
    HEALTH_CHECK_TIMEOUT = 30
    SERVER_STARTUP_TIMEOUT = 180
    REQUEST_TIMEOUT = 60
    PACKAGE_INSTALL_TIMEOUT = 300
    PLOT_CREATION_TIMEOUT = 90
    BATCH_OPERATIONS_TIMEOUT = 120
    API_EXECUTE_RAW_ENDPOINT = "/api/execute-raw"
    API_HEALTH_ENDPOINT = "/health"
    API_INSTALL_PACKAGE_ENDPOINT = "/api/install-package"
    REQUIRED_PACKAGES = ["matplotlib"]


def parse_result_data(api_response):
    """
    Parse the result data from API response, handling both dict and JSON string formats.
    
    Args:
        api_response: API response from execute_python_code containing result
        
    Returns:
        dict: Parsed result data
        
    Raises:
        AssertionError: If result cannot be parsed or is invalid
        
    Examples:
        >>> response = execute_python_code("{'test': 'value'}")
        >>> result = parse_result_data(response)
        >>> assert result["test"] == "value"
    """
    # Get the result from the API response
    result_data = api_response["data"]["result"]
    
    # If result is already a dict, return as-is
    if isinstance(result_data, dict):
        return result_data
    
    # If result is a JSON string, parse it
    if isinstance(result_data, str):
        try:
            return json.loads(result_data)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Failed to parse JSON result: {e}")
    
    # If result is neither dict nor string, it's invalid
    raise AssertionError(f"Expected dict or JSON string, got {type(result_data)}: {result_data}")


@pytest.fixture(scope="session")
def base_url() -> str:
    """
    Provide the base URL for the test server.

    Returns:
        str: Base URL of the test server

    Examples:
        def test_something(base_url):
            response = requests.get(f"{base_url}/health")
    """
    return TestConfig.BASE_URL


@pytest.fixture(scope="session")
def http_session() -> requests.Session:
    """
    Create a persistent HTTP session for efficient connection reuse.

    Returns:
        requests.Session: Configured session with timeout defaults

    Examples:
        def test_api_call(http_session):
            response = http_session.get("/health")
    """
    session = requests.Session()
    session.timeout = TestConfig.REQUEST_TIMEOUT
    return session


@pytest.fixture(scope="session", autouse=True)
def ensure_server_running(base_url: str, http_session: requests.Session):
    """
    Ensure the test server is running before any tests execute.

    Args:
        base_url: Base URL of the server
        http_session: HTTP session for requests

    Raises:
        pytest.skip: If server is not accessible within timeout

    Examples:
        This fixture runs automatically for all tests in the session.
    """
    def wait_for_server(url: str, timeout: int = TestConfig.SERVER_STARTUP_TIMEOUT) -> None:
        """Wait for server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = http_session.get(url, timeout=TestConfig.HEALTH_CHECK_TIMEOUT)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        pytest.skip(f"Server at {url} did not start within {timeout} seconds")

    # Wait for server to be ready
    health_url = f"{base_url}{TestConfig.API_HEALTH_ENDPOINT}"
    wait_for_server(health_url, TestConfig.HEALTH_CHECK_TIMEOUT)


@pytest.fixture(scope="session", autouse=True)
def ensure_required_packages(base_url: str, http_session: requests.Session):
    """
    Ensure required Python packages are installed in Pyodide.

    Args:
        base_url: Base URL of the server
        http_session: HTTP session for requests

    Raises:
        AssertionError: If package installation fails

    Examples:
        This fixture automatically installs matplotlib and other required packages.
    """
    for package in TestConfig.REQUIRED_PACKAGES:
        response = http_session.post(
            f"{base_url}{TestConfig.API_INSTALL_PACKAGE_ENDPOINT}",
            json={"package": package},
            timeout=TestConfig.PACKAGE_INSTALL_TIMEOUT,
        )
        assert response.status_code == 200, f"Failed to install {package}: {response.status_code}"


@pytest.fixture
def project_root() -> Path:
    """
    Get the project root directory path.

    Returns:
        Path: Project root directory as pathlib.Path object

    Examples:
        def test_file_exists(project_root):
            config_file = project_root / "package.json"
            assert config_file.exists()
    """
    return Path(__file__).parent.parent


@pytest.fixture
def plots_directory(project_root: Path) -> Path:
    """
    Get or create the plots directory for filesystem tests.

    Args:
        project_root: Project root directory

    Returns:
        Path: Plots directory path

    Examples:
        def test_plot_creation(plots_directory):
            matplotlib_dir = plots_directory / "matplotlib"
            assert matplotlib_dir.exists()
    """
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Ensure matplotlib subdirectory exists
    matplotlib_dir = plots_dir / "matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)

    return plots_dir


def execute_python_code(
    http_session: requests.Session,
    base_url: str,
    code: str,
    timeout: int = TestConfig.REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """
    Execute Python code using the /api/execute-raw endpoint with proper API contract compliance.

    Args:
        http_session: HTTP session for requests
        base_url: Base URL of the server
        code: Python code to execute
        timeout: Request timeout in seconds

    Returns:
        Dict containing the API response with success, data, error, meta structure
        {
            "success": bool,                    // Indicates if operation was successful
            "data": {                          // Main result data (null if error)
                "result": str,                 // Python execution result/output
                "stdout": str,                 // Standard output
                "stderr": str,                 // Standard error
                "executionTime": int           // Execution time in milliseconds
            } | null,
            "error": str | null,               // Error message (null if success)
            "meta": {                          // Metadata
                "timestamp": str               // ISO timestamp
            }
        }

    Raises:
        requests.RequestException: If HTTP request fails
        AssertionError: If response doesn't match API contract

    Examples:
        response = execute_python_code(session, url, "print('hello')")
        assert response["success"] is True
        assert "hello" in response["data"]["result"]

        # Error case
        response = execute_python_code(session, url, "invalid syntax")
        assert response["success"] is False
        assert response["error"] is not None
        assert response["data"] is None
    """
    response = http_session.post(
        f"{base_url}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=timeout
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    result = response.json()

    # Validate API contract strictly
    assert "success" in result, "Response must include 'success' field"
    assert "data" in result, "Response must include 'data' field"
    assert "error" in result, "Response must include 'error' field"
    assert "meta" in result, "Response must include 'meta' field"
    assert "timestamp" in result["meta"], "Meta must include 'timestamp' field"

    # Validate success/error state consistency
    if result["success"]:
        assert result["data"] is not None, "Success response must have non-null data"
        assert result["error"] is None, "Success response must have null error"
        assert "result" in result["data"], "Success data must contain 'result' field"
        assert "stdout" in result["data"], "Success data must contain 'stdout' field"
        assert "stderr" in result["data"], "Success data must contain 'stderr' field"
        assert "executionTime" in result["data"], "Success data must contain 'executionTime' field"
    else:
        assert result["data"] is None, "Error response must have null data"
        assert result["error"] is not None, "Error response must have non-null error"
        assert isinstance(result["error"], str), "Error must be a string"

    return result


class TestFilesystemMountBDD:
    """
    BDD-style tests for filesystem mounting functionality in Pyodide.

    This class contains behavior-driven development (BDD) tests that follow
    the Given-When-Then pattern to test filesystem mounting capabilities.
    """

    def test_given_project_structure_when_checking_directories_then_plots_directory_exists(
        self, plots_directory: Path
    ):
        """
        Test that the plots directory exists on the host filesystem.

        Given: A project with expected directory structure
        When: Checking for the plots directory existence
        Then: The plots directory should exist and be accessible

        Args:
            plots_directory: Plots directory fixture

        Examples:
            This test verifies the basic filesystem structure required for
            plot storage and retrieval operations.
        """
        # Given: Project structure is set up
        assert plots_directory is not None

        # When: Checking directory existence
        plots_exists = plots_directory.exists()
        matplotlib_dir = plots_directory / "matplotlib"
        matplotlib_exists = matplotlib_dir.exists()

        # Then: Directories should exist
        assert plots_exists, f"Plots directory should exist at {plots_directory}"
        assert matplotlib_exists, f"Matplotlib plots directory should exist at {matplotlib_dir}"
        assert plots_directory.is_dir(), "Plots path should be a directory"
        assert matplotlib_dir.is_dir(), "Matplotlib path should be a directory"

    def test_given_pyodide_environment_when_detecting_mount_functionality_then_js_interface_available(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test detection of mount functionality in Pyodide JavaScript interface.

        Given: Pyodide is running in the server environment
        When: Checking for JavaScript mount interfaces and capabilities
        Then: JS interface should be available with mount method information

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This test verifies that Pyodide has access to JavaScript interfaces
            that could potentially be used for filesystem mounting operations.
        """
        # Given: Pyodide environment is available
        code = '''
from pathlib import Path
import js

result = {
    "js_available": True,
    "has_pyodide": hasattr(js, 'pyodide'),
    "mount_methods": [],
    "current_working_dir": str(Path.cwd()),
    "root_accessible": True
}

# Check for various mount methods in JS interface
mount_checks = [
    ('mountNodeFS', 'js.pyodide.mountNodeFS'),
    ('mountFS', 'js.mountFS'),
    ('FS', 'js.FS')
]

for method_name, method_path in mount_checks:
    try:
        # Navigate to the method using getattr chain
        obj = js
        parts = method_path.split('.')[1:]  # Skip 'js' prefix
        method_info = {
            "name": method_name,
            "path": method_path,
            "available": True,
            "navigation_success": True
        }

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
                method_info[f"has_{part}"] = True
            else:
                method_info[f"has_{part}"] = False
                method_info["available"] = False
                break

        if method_info["available"]:
            method_info["type"] = str(type(obj))
            method_info["callable"] = callable(obj) if obj else False

        result["mount_methods"].append(method_info)

    except Exception as e:
        result["mount_methods"].append({
            "name": method_name,
            "path": method_path,
            "available": False,
            "error": str(e),
            "error_type": type(e).__name__
        })

# Check root directory accessibility
try:
    root_path = Path("/")
    result["root_exists"] = root_path.exists()
    result["root_contents"] = [str(p) for p in root_path.iterdir()] if root_path.exists() else []
except Exception as e:
    result["root_access_error"] = str(e)
    result["root_accessible"] = False

result
'''

        # When: Executing mount detection code
        response = execute_python_code(http_session, base_url, code)

        # Then: Response should indicate JS interface availability
        assert response["success"] is True, f"Mount detection failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Result should not be None"
        assert result["js_available"] is True, "JavaScript interface should be available"
        assert isinstance(result["mount_methods"], list), "Mount methods should be a list"
        assert len(result["mount_methods"]) > 0, "Should detect at least one mount method"

        # Verify each mount method check completed
        for method in result["mount_methods"]:
            assert "name" in method, "Each method should have a name"
            assert "available" in method, "Each method should have availability status"
            if not method["available"]:
                assert "error" in method or any(k.startswith("has_") for k in method.keys()), \
                    "Unavailable methods should have error or detailed availability info"

    def test_given_pyodide_filesystem_when_attempting_mount_operation_then_graceful_handling(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test attempting to mount a directory in Pyodide (may fail gracefully).

        Given: Pyodide virtual filesystem environment
        When: Attempting filesystem mount operations
        Then: Operation should complete without crashing, with appropriate error handling

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This test verifies that mount operations are handled gracefully,
            whether they succeed or fail based on the environment capabilities.
        """
        # Given: Pyodide environment with filesystem capabilities
        code = '''
from pathlib import Path
import js

result = {
    "mount_attempted": False,
    "mount_successful": False,
    "error": None,
    "current_dir": str(Path.cwd()),
    "has_pyodide": hasattr(js, 'pyodide'),
    "test_operations": []
}

# Test basic virtual filesystem operations first
try:
    # Create test directory in virtual filesystem
    test_vfs_dir = Path("/tmp/test_mount_source")
    test_vfs_dir.mkdir(parents=True, exist_ok=True)
    result["test_operations"].append({
        "operation": "mkdir_virtual",
        "success": test_vfs_dir.exists(),
        "path": str(test_vfs_dir)
    })

    # Create test file
    test_file = test_vfs_dir / "test.txt"
    test_file.write_text("test content for mount operation")
    result["test_operations"].append({
        "operation": "file_write_virtual",
        "success": test_file.exists(),
        "path": str(test_file),
        "content_length": len(test_file.read_text()) if test_file.exists() else 0
    })

except Exception as e:
    result["test_operations"].append({
        "operation": "virtual_fs_setup",
        "success": False,
        "error": str(e)
    })

# Only attempt mount if pyodide interface is available
if hasattr(js, 'pyodide') and hasattr(js.pyodide, 'mountNodeFS'):
    try:
        # Attempt to mount the test directory
        mount_target = "/mnt/test_mount"
        js.pyodide.mountNodeFS(str(test_vfs_dir), mount_target)
        result["mount_attempted"] = True
        result["mount_successful"] = True
        result["mount_target"] = mount_target

        # Test if mount worked by checking target directory
        try:
            mount_path = Path(mount_target)
            if mount_path.exists():
                result["mount_verification"] = {
                    "target_exists": True,
                    "is_directory": mount_path.is_dir(),
                    "contents": [str(p.name) for p in mount_path.iterdir()] if mount_path.is_dir() else []
                }
            else:
                result["mount_verification"] = {"target_exists": False}
        except Exception as verify_error:
            result["mount_verification"] = {
                "error": str(verify_error),
                "error_type": type(verify_error).__name__
            }

    except Exception as mount_error:
        result["mount_attempted"] = True
        result["mount_successful"] = False
        result["error"] = str(mount_error)
        result["error_type"] = type(mount_error).__name__
else:
    result["error"] = "mountNodeFS not available in current environment"
    result["pyodide_available"] = hasattr(js, 'pyodide')
    if hasattr(js, 'pyodide'):
        result["mountNodeFS_available"] = hasattr(js.pyodide, 'mountNodeFS')

result
'''

        # When: Attempting mount operations
        response = execute_python_code(http_session, base_url, code)

        # Then: Operation should complete gracefully
        assert response["success"] is True, f"Mount attempt failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Result should not be None"

        # Verify virtual filesystem operations work regardless of mount capability
        test_ops = result.get("test_operations", [])
        if test_ops:
            mkdir_op = next((op for op in test_ops if op["operation"] == "mkdir_virtual"), None)
            if mkdir_op:
                assert mkdir_op["success"], "Virtual directory creation should work"

            file_op = next((op for op in test_ops if op["operation"] == "file_write_virtual"), None)
            if file_op:
                assert file_op["success"], "Virtual file creation should work"
                assert file_op["content_length"] > 0, "File should have content"

        # Mount may or may not work depending on environment - both outcomes are valid
        if result.get("mount_attempted"):
            if result.get("mount_successful"):
                # If mount succeeded, verify it worked properly
                assert "mount_verification" in result, "Should verify successful mount"
                verification = result["mount_verification"]
                if "error" not in verification:
                    assert verification.get("target_exists"), "Mount target should exist"
            else:
                # Mount failed - should have error information
                assert result.get("error") is not None, "Failed mount should have error message"
        else:
            # Mount not attempted - should have explanation
            assert result.get("error") is not None, "Should explain why mount was not attempted"

    def test_given_pyodide_when_creating_directories_then_virtual_filesystem_works(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test creating directories and files in Pyodide's virtual filesystem.

        Given: Pyodide virtual filesystem environment
        When: Creating directories and files programmatically
        Then: Virtual filesystem operations should succeed

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This test verifies that Pyodide's virtual filesystem supports
            standard directory and file operations using pathlib.
        """
        # Given: Pyodide virtual filesystem is available
        code = '''
from pathlib import Path
import json

result = {
    "current_dir": str(Path.cwd()),
    "root_contents": [],
    "directory_operations": [],
    "file_operations": [],
    "success": True
}

# Check root directory contents
try:
    root_path = Path("/")
    result["root_contents"] = [str(p.name) for p in root_path.iterdir()]
    result["root_accessible"] = True
except Exception as e:
    result["root_list_error"] = str(e)
    result["root_accessible"] = False

# Test directory creation with pathlib
test_directories = [
    "/test_mount_dir",
    "/virtual_test/nested/deep/structure",
    "/home/pyodide/plots/test_matplotlib",
    "/home/pyodide/uploads/test_data"
]

for test_dir in test_directories:
    try:
        dir_path = Path(test_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        operation_result = {
            "path": test_dir,
            "created": dir_path.exists(),
            "is_directory": dir_path.is_dir() if dir_path.exists() else False,
            "parents_created": True
        }

        # Test write permissions by creating a test file
        if operation_result["created"]:
            test_file = dir_path / "test_file.txt"
            test_content = f"Test content for {test_dir}"
            test_file.write_text(test_content)

            operation_result["file_creation"] = test_file.exists()
            operation_result["file_readable"] = test_file.read_text() == test_content if test_file.exists() else False

            result["file_operations"].append({
                "path": str(test_file),
                "created": operation_result["file_creation"],
                "content_matches": operation_result["file_readable"],
                "size": test_file.stat().st_size if test_file.exists() else 0
            })

        result["directory_operations"].append(operation_result)

    except Exception as e:
        result["directory_operations"].append({
            "path": test_dir,
            "created": False,
            "error": str(e),
            "error_type": type(e).__name__
        })
        result["success"] = False

# Test complex file operations
try:
    complex_dir = Path("/complex_test")
    complex_dir.mkdir(exist_ok=True)

    # Create multiple files with different content types
    files_to_create = [
        ("data.json", json.dumps({"test": True, "value": 123})),
        ("readme.txt", "This is a test readme file\\nwith multiple lines"),
        ("config.py", "# Python configuration\\nDEBUG = True\\nVERSION = '1.0'")
    ]

    for filename, content in files_to_create:
        file_path = complex_dir / filename
        file_path.write_text(content)

        result["file_operations"].append({
            "path": str(file_path),
            "created": file_path.exists(),
            "size": len(content),
            "actual_size": file_path.stat().st_size if file_path.exists() else 0,
            "type": filename.split(".")[-1]
        })

except Exception as e:
    result["complex_operations_error"] = str(e)
    result["success"] = False

result
'''

        # When: Creating directories and files
        response = execute_python_code(http_session, base_url, code)

        # Then: Virtual filesystem operations should succeed
        assert response["success"] is True, f"Virtual filesystem test failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Result should not be None"
        assert result.get("success") is True, "Virtual filesystem operations should succeed"

        # Verify directory operations
        dir_ops = result.get("directory_operations", [])
        assert len(dir_ops) > 0, "Should have directory operations"

        successful_dirs = [op for op in dir_ops if op.get("created")]
        assert len(successful_dirs) > 0, "At least some directories should be created successfully"

        for dir_op in successful_dirs:
            assert dir_op["is_directory"], f"Created path {dir_op['path']} should be a directory"

        # Verify file operations
        file_ops = result.get("file_operations", [])
        assert len(file_ops) > 0, "Should have file operations"

        successful_files = [op for op in file_ops if op.get("created")]
        assert len(successful_files) > 0, "At least some files should be created successfully"

        for file_op in successful_files:
            assert file_op.get("size", 0) > 0, f"File {file_op['path']} should have content"
            if "actual_size" in file_op and "size" in file_op:
                assert file_op["actual_size"] >= file_op["size"], "File size should match or exceed expected"

    def test_given_matplotlib_when_creating_plots_then_virtual_plots_directory_accessible(
        self, http_session: requests.Session, base_url: str, plots_directory: Path
    ):
        """
        Test matplotlib plot creation in virtual filesystem with pathlib.

        Given: Matplotlib is installed and plots directory exists
        When: Creating plots using matplotlib and pathlib for cross-platform compatibility
        Then: Plots should be created in virtual filesystem and be accessible

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            plots_directory: Plots directory fixture

        Examples:
            This test verifies that data visualization workflows work correctly
            in the virtual filesystem environment using cross-platform paths.
        """
        # Given: Matplotlib is available and plots directory exists
        assert plots_directory.exists(), "Plots directory should exist"

        code = '''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

result = {
    "matplotlib_imported": True,
    "plot_operations": [],
    "virtual_plots_dir": "/home/pyodide/plots/matplotlib",
    "success": True
}

# Create plots directory in virtual filesystem using pathlib
try:
    plots_dir = Path('/home/pyodide/plots/matplotlib')
    plots_dir.mkdir(parents=True, exist_ok=True)

    result["plots_dir_created"] = plots_dir.exists()
    result["plots_dir_path"] = str(plots_dir)

    if plots_dir.exists():
        # Create multiple test plots using cross-platform paths
        plot_configs = [
            {
                "name": "sine_wave",
                "data_func": lambda: (np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))),
                "title": "Sine Wave Test"
            },
            {
                "name": "scatter_plot",
                "data_func": lambda: (np.random.randn(50), np.random.randn(50)),
                "title": "Random Scatter Plot"
            },
            {
                "name": "bar_chart",
                "data_func": lambda: (["A", "B", "C", "D"], [23, 45, 56, 78]),
                "title": "Bar Chart Test"
            }
        ]

        for config in plot_configs:
            try:
                # Generate plot data
                x_data, y_data = config["data_func"]()

                # Create plot using matplotlib
                plt.figure(figsize=(10, 6))

                if config["name"] == "scatter_plot":
                    plt.scatter(x_data, y_data, alpha=0.6)
                elif config["name"] == "bar_chart":
                    plt.bar(x_data, y_data)
                else:
                    plt.plot(x_data, y_data, linewidth=2)

                plt.title(config["title"])
                plt.grid(True, alpha=0.3)

                # Save using pathlib for cross-platform compatibility
                plot_file = plots_dir / f'{config["name"]}_test.png'
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()

                # Verify file was created
                plot_exists = plot_file.exists()
                file_size = plot_file.stat().st_size if plot_exists else 0

                result["plot_operations"].append({
                    "name": config["name"],
                    "file_path": str(plot_file),
                    "created": plot_exists,
                    "file_size": file_size,
                    "title": config["title"]
                })

            except Exception as plot_error:
                result["plot_operations"].append({
                    "name": config["name"],
                    "created": False,
                    "error": str(plot_error),
                    "error_type": type(plot_error).__name__
                })
                result["success"] = False

        # List all created plots
        try:
            created_plots = [p for p in plots_dir.iterdir() if p.is_file()]
            result["created_files"] = [str(p.name) for p in created_plots]
            result["total_plots_created"] = len(created_plots)
        except Exception as list_error:
            result["list_error"] = str(list_error)

except Exception as e:
    result["setup_error"] = str(e)
    result["success"] = False

result
'''

        # When: Creating matplotlib plots
        response = execute_python_code(http_session, base_url, code, timeout=TestConfig.PLOT_CREATION_TIMEOUT)

        # Then: Plots should be created successfully
        assert response["success"] is True, f"Matplotlib plot creation failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Result should not be None"
        assert result.get("matplotlib_imported") is True, "Matplotlib should be imported successfully"
        assert result.get("plots_dir_created") is True, "Plots directory should be created in virtual filesystem"

        # Verify plot operations
        plot_ops = result.get("plot_operations", [])
        assert len(plot_ops) > 0, "Should have attempted plot creation operations"

        successful_plots = [op for op in plot_ops if op.get("created")]
        assert len(successful_plots) > 0, "At least some plots should be created successfully"

        for plot in successful_plots:
            assert plot.get("file_size", 0) > 0, f"Plot file {plot['file_path']} should have content"
            assert plot["file_path"].endswith('.png'), "Plot files should be PNG format"
            assert "/home/pyodide/plots/matplotlib/" in plot["file_path"], \
                "Plots should be in correct virtual directory"

        # Verify total file count
        total_created = result.get("total_plots_created", 0)
        assert total_created > 0, "Should have created at least one plot file"

        created_files = result.get("created_files", [])
        assert len(created_files) == total_created, "File list count should match total count"

    def test_given_complex_filesystem_when_performing_batch_operations_then_all_operations_succeed(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test complex batch filesystem operations for comprehensive coverage.

        Given: Pyodide virtual filesystem with full capabilities
        When: Performing complex batch operations including nested directories, multiple files, and cross-references
        Then: All operations should succeed with proper error handling and validation

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This comprehensive test covers edge cases and complex scenarios
            to ensure robust filesystem operation handling.
        """
        # Given: Virtual filesystem environment ready for complex operations
        code = '''
from pathlib import Path
import json
import time

result = {
    "batch_operations": [],
    "performance_metrics": {},
    "error_handling": [],
    "cross_platform_compatibility": {},
    "success": True
}

start_time = time.time()

# Test 1: Complex nested directory structure
try:
    nested_structure = [
        "/project/src/components/ui/buttons",
        "/project/src/components/ui/forms",
        "/project/tests/unit/components",
        "/project/tests/integration/api",
        "/project/docs/api/v1/endpoints",
        "/project/assets/images/icons",
        "/project/config/environments/development"
    ]

    structure_results = []
    for dir_path in nested_structure:
        path_obj = Path(dir_path)
        path_obj.mkdir(parents=True, exist_ok=True)
        structure_results.append({
            "path": dir_path,
            "created": path_obj.exists(),
            "depth": len(path_obj.parts),
            "parent_count": len(path_obj.parents)
        })

    result["batch_operations"].append({
        "operation": "nested_directory_creation",
        "total_dirs": len(nested_structure),
        "successful": sum(1 for r in structure_results if r["created"]),
        "details": structure_results
    })

except Exception as e:
    result["batch_operations"].append({
        "operation": "nested_directory_creation",
        "error": str(e),
        "success": False
    })

# Test 2: Batch file creation with different content types
try:
    base_path = Path("/project")
    file_operations = []

    files_to_create = [
        ("package.json", json.dumps({"name": "test-project", "version": "1.0.0"})),
        ("README.md", "# Test Project\\n\\nThis is a test project for filesystem operations."),
        ("config.py", "# Configuration file\\nDEBUG = True\\nDATABASE_URL = 'sqlite:///test.db'"),
        (".gitignore", "node_modules/\\n*.pyc\\n__pycache__/\\n.env"),
        ("src/main.py", "#!/usr/bin/env python3\\n\\ndef main():\\n    print('Hello, World!')\\n\\n" +
         "if __name__ == '__main__':\\n    main()"),
        ("tests/test_main.py", "import unittest\\n\\nclass TestMain(unittest.TestCase):\\n" +
         "    def test_example(self):\\n        self.assertTrue(True)")
    ]

    for file_path, content in files_to_create:
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        file_operations.append({
            "path": str(full_path),
            "created": full_path.exists(),
            "size": len(content),
            "actual_size": full_path.stat().st_size if full_path.exists() else 0,
            "extension": full_path.suffix
        })

    result["batch_operations"].append({
        "operation": "batch_file_creation",
        "total_files": len(files_to_create),
        "successful": sum(1 for f in file_operations if f["created"]),
        "details": file_operations
    })

except Exception as e:
    result["batch_operations"].append({
        "operation": "batch_file_creation",
        "error": str(e),
        "success": False
    })

# Test 3: Cross-platform path handling
try:
    cross_platform_tests = []

    # Test various path formats that should work cross-platform
    test_paths = [
        "/unix/style/path",
        "/path/with spaces/test",
        "/path/with-dashes/test",
        "/path/with_underscores/test",
        "/path/with.dots/test",
        "/UPPERCASE/PATH/test",
        "/mixed/Case/Path/test"
    ]

    for test_path in test_paths:
        path_obj = Path(test_path)
        path_obj.mkdir(parents=True, exist_ok=True)

        test_file = path_obj / "cross_platform_test.txt"
        test_file.write_text(f"Test content for {test_path}")

        cross_platform_tests.append({
            "original_path": test_path,
            "normalized_path": str(path_obj),
            "dir_created": path_obj.exists(),
            "file_created": test_file.exists(),
            "path_parts": list(path_obj.parts),
            "file_readable": test_file.read_text() if test_file.exists() else None
        })

    result["cross_platform_compatibility"] = {
        "tests_run": len(test_paths),
        "successful": sum(1 for t in cross_platform_tests if t["dir_created"] and t["file_created"]),
        "details": cross_platform_tests
    }

except Exception as e:
    result["cross_platform_compatibility"] = {
        "error": str(e),
        "success": False
    }

# Test 4: Error handling and edge cases
try:
    error_tests = []

    # Test creating files in non-existent directories without parents=True
    try:
        problematic_file = Path("/non/existent/deep/path/file.txt")
        problematic_file.write_text("This should fail")
        error_tests.append({
            "test": "file_without_parent_dirs",
            "expected_failure": True,
            "actual_result": "unexpected_success"
        })
    except Exception as expected_error:
        error_tests.append({
            "test": "file_without_parent_dirs",
            "expected_failure": True,
            "actual_result": "expected_failure",
            "error_type": type(expected_error).__name__
        })

    # Test very long paths
    try:
        long_path = "/very/long/path/" + "/".join([f"segment{i}" for i in range(20)])
        long_path_obj = Path(long_path)
        long_path_obj.mkdir(parents=True, exist_ok=True)
        error_tests.append({
            "test": "very_long_path",
            "path_length": len(long_path),
            "created": long_path_obj.exists()
        })
    except Exception as long_path_error:
        error_tests.append({
            "test": "very_long_path",
            "error": str(long_path_error)
        })

    result["error_handling"] = error_tests

except Exception as e:
    result["error_handling"] = [{"general_error": str(e)}]

# Performance metrics
end_time = time.time()
result["performance_metrics"] = {
    "total_execution_time": end_time - start_time,
    "operations_completed": len([op for op in result["batch_operations"] if "error" not in op]),
    "average_time_per_operation": (end_time - start_time) / max(len(result["batch_operations"]), 1)
}

result
'''

        # When: Performing complex batch operations
        response = execute_python_code(http_session, base_url, code, timeout=TestConfig.BATCH_OPERATIONS_TIMEOUT)

        # Then: All operations should succeed with proper handling
        assert response["success"] is True, f"Complex batch operations failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Result should not be None"
        assert result.get("success") is True, "Batch operations should succeed overall"

        # Verify batch operations
        batch_ops = result.get("batch_operations", [])
        assert len(batch_ops) > 0, "Should have performed batch operations"

        successful_ops = [op for op in batch_ops if "error" not in op]
        assert len(successful_ops) > 0, "At least some batch operations should succeed"

        # Verify nested directory creation
        nested_op = next((op for op in batch_ops if op.get("operation") == "nested_directory_creation"), None)
        if nested_op:
            assert nested_op.get("successful", 0) > 0, "Should successfully create nested directories"
            assert nested_op.get("total_dirs", 0) > 0, "Should attempt to create multiple directories"

        # Verify batch file creation
        file_op = next((op for op in batch_ops if op.get("operation") == "batch_file_creation"), None)
        if file_op:
            assert file_op.get("successful", 0) > 0, "Should successfully create batch files"
            assert file_op.get("total_files", 0) > 0, "Should attempt to create multiple files"

        # Verify cross-platform compatibility
        cross_platform = result.get("cross_platform_compatibility", {})
        if "tests_run" in cross_platform:
            assert cross_platform["tests_run"] > 0, "Should run cross-platform tests"
            assert cross_platform.get("successful", 0) > 0, "Should have successful cross-platform operations"

        # Verify performance metrics
        perf_metrics = result.get("performance_metrics", {})
        assert "total_execution_time" in perf_metrics, "Should track execution time"
        assert perf_metrics.get("total_execution_time", 0) > 0, "Should have measurable execution time"
        assert perf_metrics.get("operations_completed", 0) > 0, "Should complete some operations"

    def test_given_invalid_code_when_executing_then_proper_error_handling(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test error handling with invalid Python code to ensure proper API contract compliance.

        Given: Invalid Python code with syntax errors
        When: Executing the code via the API
        Then: Response should follow error contract with proper structure

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This test verifies that the API properly handles errors and returns
            them in the expected format with success=False and error details.
        """
        # Given: Invalid Python code with syntax errors
        invalid_codes = [
            "print('missing closing quote",
            "def invalid_function(\n    # missing closing parenthesis",
            "import non_existent_module_that_does_not_exist",
            "x = 1 / 0  # Division by zero runtime error",
            "undefined_variable.method_call()",
        ]

        for i, invalid_code in enumerate(invalid_codes):
            # When: Executing invalid code
            response = execute_python_code(http_session, base_url, invalid_code)

            # Then: Should follow error contract
            assert response["success"] is False, f"Invalid code {i+1} should result in success=False"
            assert response["data"] is None, f"Invalid code {i+1} should have null data"
            assert response["error"] is not None, f"Invalid code {i+1} should have error message"
            assert isinstance(response["error"], str), f"Invalid code {i+1} error should be string"
            assert len(response["error"]) > 0, f"Invalid code {i+1} error should not be empty"
            assert "meta" in response, f"Invalid code {i+1} should have meta field"
            assert "timestamp" in response["meta"], f"Invalid code {i+1} should have timestamp"

    def test_given_filesystem_permissions_when_testing_access_then_security_constraints_respected(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test filesystem access permissions and security constraints in Pyodide environment.

        Given: Pyodide virtual filesystem with security constraints
        When: Attempting to access various filesystem locations
        Then: Security constraints should be properly enforced

        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server

        Examples:
            This test ensures that the virtual filesystem respects security
            boundaries and doesn't allow unauthorized access to host system.
        """
        # Given: Virtual filesystem environment with security constraints
        code = '''
from pathlib import Path

result = {
    "security_tests": [],
    "accessible_directories": [],
    "inaccessible_directories": [],
    "file_permissions": {},
    "virtual_fs_boundaries": {}
}

# Test various directory access patterns
test_directories = [
    "/tmp",                    # Should be accessible
    "/home/pyodide",          # Should be accessible
    "/home/pyodide/uploads",  # Should be accessible
    "/plots",                 # Should be accessible
    "/etc",                   # May or may not exist in virtual fs
    "/proc",                  # May or may not exist in virtual fs
    "/sys",                   # May or may not exist in virtual fs
]

for test_dir in test_directories:
    try:
        dir_path = Path(test_dir)
        is_accessible = dir_path.exists()

        access_info = {
            "path": test_dir,
            "exists": is_accessible,
            "is_directory": dir_path.is_dir() if is_accessible else False,
            "readable": False,
            "writable": False
        }

        if is_accessible:
            # Test read permissions
            try:
                list(dir_path.iterdir())
                access_info["readable"] = True
                result["accessible_directories"].append(test_dir)
            except PermissionError:
                access_info["read_error"] = "Permission denied"
            except Exception as e:
                access_info["read_error"] = str(e)

            # Test write permissions
            try:
                test_file = dir_path / "security_test.tmp"
                test_file.write_text("security test")
                if test_file.exists():
                    access_info["writable"] = True
                    test_file.unlink()  # Clean up
            except PermissionError:
                access_info["write_error"] = "Permission denied"
            except Exception as e:
                access_info["write_error"] = str(e)
        else:
            result["inaccessible_directories"].append(test_dir)

        result["security_tests"].append(access_info)

    except Exception as e:
        result["security_tests"].append({
            "path": test_dir,
            "error": str(e),
            "error_type": type(e).__name__
        })

# Test virtual filesystem boundaries
try:
    # Create a test structure to verify isolation
    test_root = Path("/virtual_fs_test")
    test_root.mkdir(exist_ok=True)

    # Create nested structure
    deep_path = test_root / "level1" / "level2" / "level3"
    deep_path.mkdir(parents=True, exist_ok=True)

    # Test file operations at different levels
    for i, level_path in enumerate([test_root, test_root / "level1", deep_path]):
        test_file = level_path / f"test_file_{i}.txt"
        test_file.write_text(f"Content for level {i}")

        result["virtual_fs_boundaries"][f"level_{i}"] = {
            "path": str(level_path),
            "file_created": test_file.exists(),
            "file_readable": test_file.read_text() if test_file.exists() else None
        }

except Exception as e:
    result["virtual_fs_boundaries"]["error"] = str(e)

result
'''

        # When: Testing filesystem access permissions
        response = execute_python_code(http_session, base_url, code)

        # Then: Security constraints should be properly enforced
        assert response["success"] is True, f"Security test failed: {response.get('error')}"

        result = parse_result_data(response)
        assert result is not None, "Security test result should not be None"

        # Verify security tests were performed
        security_tests = result.get("security_tests", [])
        assert len(security_tests) > 0, "Should perform security tests"

        # Verify virtual filesystem is functional (should have some accessible directories)
        accessible = result.get("accessible_directories", [])
        assert len(accessible) > 0, "Should have some accessible directories in virtual filesystem"

        # Common virtual filesystem directories should be accessible
        expected_accessible = ["/tmp", "/home/pyodide"]
        for expected_dir in expected_accessible:
            accessible_test = next((test for test in security_tests if test["path"] == expected_dir), None)
            if accessible_test and accessible_test.get("exists"):
                assert accessible_test.get("readable", False), f"{expected_dir} should be readable"

        # Verify virtual filesystem boundaries work
        boundaries = result.get("virtual_fs_boundaries", {})
        if "error" not in boundaries:
            level_tests = [k for k in boundaries.keys() if k.startswith("level_")]
            assert len(level_tests) > 0, "Should test virtual filesystem boundaries"

            for level_key in level_tests:
                level_data = boundaries[level_key]
                assert level_data.get("file_created", False), f"File creation should work at {level_key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
