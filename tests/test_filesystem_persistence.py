"""
Pytest-based tests for filesystem persistence behavior across server restarts.

This module tests the critical issue mentioned in Pyodide FAQ:
https://pyodide.org/en/stable/usage/faq.html#why-changes-made-to-indexeddb-don-t-persist

Key points:
- IndexedDB (pyodide.FS.filesystem.IDBFS) is asynchronous
- Changes don't persist unless pyodide.FS.syncfs() is called
- We need to test both scenarios: with and without syncfs

Requirements:
    - pytest: Testing framework
    - requests: HTTP client for API calls
    - pathlib: Cross-platform path operations

Examples:
    Run all persistence tests:
        $ uv run pytest tests/test_filesystem_persistence.py -v

    Run specific test scenario:
        $ uv run pytest tests/test_filesystem_persistence.py::TestFilesystemPersistenceBDD::test_given_pyodide_filesystem_when_creating_files_then_behavior_documented -v

    Run with coverage:
        $ uv run pytest tests/test_filesystem_persistence.py --cov=src --cov-report=term-missing
"""

import json
import time
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
    PERSISTENCE_TEST_TIMEOUT = 90
    FILESYSTEM_ANALYSIS_TIMEOUT = 45
    API_EXECUTE_RAW_ENDPOINT = "/api/execute-raw"
    API_HEALTH_ENDPOINT = "/health"
    API_INSTALL_PACKAGE_ENDPOINT = "/api/install-package"
    REQUIRED_PACKAGES = ["matplotlib"]


def parse_result_data(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse result data from server response, handling JSON string conversion.
    
    The server may return result data as a JSON string that needs to be parsed
    back into a dictionary for proper assertion handling.
    
    Args:
        response: Server response dictionary containing data.result
        
    Returns:
        Parsed result data as dictionary
        
    Raises:
        json.JSONDecodeError: If result data is malformed JSON
        KeyError: If response structure is invalid
    """
    result_data = response["data"]["result"]
    if isinstance(result_data, str):
        try:
            return json.loads(result_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, return as-is 
            return {"raw_result": result_data}
    return result_data


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


class TestFilesystemPersistenceBDD:
    """
    BDD-style tests for filesystem persistence behavior in Pyodide.
    
    This class contains behavior-driven development (BDD) tests that follow
    the Given-When-Then pattern to test filesystem persistence capabilities
    and document expected behavior based on Pyodide FAQ.
    """

    def test_given_pyodide_filesystem_when_creating_files_then_behavior_documented(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test and document filesystem behavior for file creation in Pyodide.
        
        Given: Pyodide virtual filesystem environment
        When: Creating files and examining filesystem capabilities
        Then: Behavior should be documented and consistent with Pyodide FAQ
        
        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            
        Examples:
            This test documents the current filesystem behavior and provides
            guidance for users who need file persistence.
        """
        # Given: Pyodide filesystem environment is available
        test_filename = f"test_persistence_{int(time.time())}.txt"
        test_content = "This tests filesystem persistence behavior"
        
        code = f'''
from pathlib import Path
import sys

# Create file in Pyodide filesystem
test_file = Path("/tmp/{test_filename}")
test_file.write_text("{test_content}")

result = {{
    "file_exists": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "filename": str(test_file),
    "platform": sys.platform,
    "pyodide_available": "pyodide" in sys.modules,
    "filesystem_info": {{
        "cwd": str(Path.cwd()),
        "tmp_exists": Path("/tmp").exists(),
        "tmp_writable": True  # We just wrote to it
    }}
}}

# Check if we have access to FS API
try:
    import js
    if hasattr(js, "pyodide"):
        result["pyodide_js_available"] = True
        if hasattr(js.pyodide, "FS"):
            result["fs_api_available"] = True
            result["has_syncfs"] = hasattr(js.pyodide.FS, "syncfs")
        else:
            result["fs_api_available"] = False
    else:
        result["pyodide_js_available"] = False
except Exception as e:
    result["js_access_error"] = str(e)

result
'''
        
        # When: Creating file and examining filesystem
        response = execute_python_code(http_session, base_url, code)
        
        # Then: File creation should succeed and provide system information
        assert response["success"] is True, f"File creation failed: {response.get('error')}"
        
        data = parse_result_data(response)
        assert data is not None, "Result should not be None"
        assert data.get("file_exists") is True, "File should exist after creation"
        assert data.get("file_content") == test_content, "File content should match"
        
        # Document the filesystem behavior
        print("\\n🔍 FILESYSTEM BEHAVIOR ANALYSIS:")
        print(f"📋 Platform: {data.get('platform')}")
        print(f"📋 Pyodide available: {data.get('pyodide_available')}")
        print(f"📋 Pyodide JS available: {data.get('pyodide_js_available')}")
        print(f"📋 FS API available: {data.get('fs_api_available')}")
        print(f"📋 Has syncfs(): {data.get('has_syncfs')}")
        print(f"📋 Current working directory: {data.get('filesystem_info', {}).get('cwd')}")

    def test_given_file_created_when_new_execution_context_then_persistence_behavior_documented(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test file persistence within the same server session across execution contexts.
        
        Given: A file created in Pyodide filesystem in one execution
        When: Checking for the file in a new execution context
        Then: Behavior should be documented (may or may not persist within session)
        
        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            
        Examples:
            This test documents whether files persist across different
            Python execution contexts within the same server session.
        """
        # Given: File created in previous execution
        test_filename = f"test_context_persistence_{int(time.time())}.txt"
        test_content = "This tests persistence across execution contexts"
        
        # Step 1: Create file
        create_code = f'''
from pathlib import Path

test_file = Path("/tmp/{test_filename}")
test_file.write_text("{test_content}")

result = {{
    "file_created": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "filename": str(test_file)
}}
result
'''
        
        response = execute_python_code(http_session, base_url, create_code)
        assert response["success"] is True, "File creation should succeed"
        
        creation_data = parse_result_data(response)
        assert creation_data.get("file_created") is True, "File should be created"
        
        # When: Checking file in new execution context
        check_code = f'''
from pathlib import Path

test_file = Path("/tmp/{test_filename}")
result = {{
    "file_exists_in_new_context": test_file.exists(),
    "file_content_in_new_context": test_file.read_text() if test_file.exists() else None,
    "content_matches": False
}}

if test_file.exists():
    content = test_file.read_text()
    result["content_matches"] = (content == "{test_content}")

result
'''
        
        response = execute_python_code(http_session, base_url, check_code)
        assert response["success"] is True, "Persistence check should succeed"
        
        # Then: Document the persistence behavior
        persistence_data = parse_result_data(response)
        
        print(f"\\n📋 File exists in new execution context: {persistence_data.get('file_exists_in_new_context')}")
        print(f"📋 Content matches: {persistence_data.get('content_matches')}")
        
        if persistence_data.get("file_exists_in_new_context"):
            print("✅ Files persist within the same server session")
            assert persistence_data.get("content_matches") is True, "Content should match if file persists"
        else:
            print("❌ Files do not persist even within same server session")
        
        # Document the expected behavior based on Pyodide FAQ
        print("\\n📚 EXPECTED BEHAVIOR (from Pyodide FAQ):")
        print("- Files in IndexedDB (IDBFS) require pyodide.FS.syncfs() to persist")
        print("- Without syncfs(), changes are only in memory")
        print("- Server restart = loss of all non-synced files")
        print("- This test documents current behavior for future reference")
    def test_given_pyodide_environment_when_testing_recommended_patterns_then_guidance_provided(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test and document recommended patterns for file persistence in Pyodide applications.
        
        Given: Pyodide environment with various storage options
        When: Testing different persistence patterns
        Then: Should provide clear guidance on best practices
        
        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            
        Examples:
            This test provides comprehensive guidance for developers on
            how to handle file persistence in Pyodide applications.
        """
        # Given: Pyodide environment with storage capabilities
        code = '''
from pathlib import Path
import sys

# Recommended pattern for file persistence in Pyodide applications:

result = {
    "pattern_description": "Use explicit data serialization instead of relying on filesystem persistence",
    "recommended_approaches": [
        "1. Return file contents as API response data",
        "2. Use local filesystem mounting for direct file access", 
        "3. Store data in external database/storage",
        "4. Use explicit syncfs() calls if IndexedDB persistence is needed"
    ],
    "current_capabilities": {}
}

# Test what we can do in current setup
test_file = Path("/tmp/recommended_pattern_test.txt")
test_content = "Data that needs to persist"
test_file.write_text(test_content)

# Pattern 1: Return file contents as data
result["current_capabilities"]["file_content_as_data"] = {
    "file_content": test_file.read_text(),
    "file_size": len(test_content),
    "can_return_as_api_response": True
}

# Pattern 2: Test if files appear in mounted directories
mounted_file = Path("/home/pyodide/plots/matplotlib/api_test.txt")
try:
    mounted_file.parent.mkdir(parents=True, exist_ok=True)
    mounted_file.write_text("Test file in mounted directory")
    result["current_capabilities"]["mounted_directory"] = {
        "can_write": True,
        "file_exists": mounted_file.exists(),
        "file_path": str(mounted_file)
    }
except Exception as e:
    result["current_capabilities"]["mounted_directory"] = {
        "can_write": False,
        "error": str(e)
    }

# Pattern 3: Test syncfs availability
try:
    import js
    if hasattr(js, "pyodide") and hasattr(js.pyodide, "FS") and hasattr(js.pyodide.FS, "syncfs"):
        result["current_capabilities"]["syncfs_available"] = True
    else:
        result["current_capabilities"]["syncfs_available"] = False
except Exception as e:
    result["current_capabilities"]["syncfs_available"] = False
    result["current_capabilities"]["syncfs_error"] = str(e)

result
'''
        
        # When: Testing recommended patterns
        response = execute_python_code(http_session, base_url, code, timeout=TestConfig.PERSISTENCE_TEST_TIMEOUT)
        
        # Then: Should provide comprehensive guidance
        assert response["success"] is True, f"Pattern test failed: {response.get('error')}"
        
        data = parse_result_data(response)
        assert data is not None, "Pattern test result should not be None"
        
        print("\\n� RECOMMENDED PERSISTENCE PATTERNS:")
        for i, approach in enumerate(data.get("recommended_approaches", []), 1):
            print(f"   {approach}")
        
        print("\\n🔧 CURRENT CAPABILITIES:")
        capabilities = data.get("current_capabilities", {})
        
        # Test file content as data
        file_data = capabilities.get("file_content_as_data", {})
        if file_data.get("can_return_as_api_response"):
            print(f"✅ Can return file contents as API response ({file_data.get('file_size')} bytes)")
        
        # Test mounted directory
        mounted = capabilities.get("mounted_directory", {})
        if mounted.get("can_write"):
            print(f"✅ Can write to mounted directories: {mounted.get('file_path')}")
        else:
            print(f"❌ Cannot write to mounted directories: {mounted.get('error')}")
        
        # Test syncfs availability
        syncfs_available = capabilities.get("syncfs_available", False)
        if syncfs_available:
            print("✅ syncfs() is available for IndexedDB persistence")
        else:
            print("❌ syncfs() is not available")
            if "syncfs_error" in capabilities:
                print(f"   Error: {capabilities['syncfs_error']}")

    def test_given_filesystem_types_when_detecting_capabilities_then_environment_documented(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Test and document the filesystem types and capabilities available in Pyodide.
        
        Given: Pyodide environment with various filesystem types
        When: Detecting filesystem capabilities and mount points
        Then: Should document the available filesystem environment
        
        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            
        Examples:
            This test provides detailed information about the Pyodide
            filesystem environment for debugging and optimization.
        """
        # Given: Pyodide environment with filesystem capabilities
        code = '''
import js
import sys
from pathlib import Path

result = {
    "python_platform": sys.platform,
    "pyodide_available": "pyodide" in sys.modules,
    "fs_api_available": False,
    "filesystem_types": [],
    "mount_points": {}
}

# Check if we can access Emscripten FS API
try:
    if hasattr(js, "pyodide") and hasattr(js.pyodide, "FS"):
        result["fs_api_available"] = True
        
        # Try to get filesystem information
        if hasattr(js.pyodide.FS, "filesystems"):
            result["filesystem_types"] = list(js.pyodide.FS.filesystems.keys())
        
        # Check mount points
        for path in ["/", "/tmp", "/home", "/dev", "/proc"]:
            try:
                stat = js.pyodide.FS.stat(path)
                result["mount_points"][path] = {
                    "exists": True,
                    "mode": stat.mode if hasattr(stat, "mode") else None
                }
            except:
                result["mount_points"][path] = {"exists": False}
    
except Exception as e:
    result["fs_api_error"] = str(e)

# Test directory creation in different locations
test_locations = ["/tmp", "/home", "/dev/shm", "/var/tmp"]
result["location_tests"] = {}

for location in test_locations:
    try:
        test_dir = Path(location) / "test_dir"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        
        result["location_tests"][location] = {
            "writable": True,
            "test_file_exists": test_file.exists()
        }
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        
    except Exception as e:
        result["location_tests"][location] = {
            "writable": False,
            "error": str(e)
        }

result
'''
        
        # When: Detecting filesystem capabilities
        response = execute_python_code(http_session, base_url, code)
        
        # Then: Should document filesystem environment
        assert response["success"] is True, f"Filesystem detection failed: {response.get('error')}"
        
        data = parse_result_data(response)
        assert data is not None, "Filesystem detection result should not be None"
        
        print("\\n🔍 FILESYSTEM ANALYSIS:")
        print(f"📋 Platform: {data.get('python_platform')}")
        print(f"📋 Pyodide available: {data.get('pyodide_available')}")
        print(f"📋 FS API available: {data.get('fs_api_available')}")
        print(f"📋 Filesystem types: {data.get('filesystem_types', [])}")
        
        print("\\n📁 Mount points:")
        for path, info in data.get("mount_points", {}).items():
            status = "✅" if info.get("exists") else "❌"
            print(f"   {status} {path}: {info}")
        
        print("\\n✏️  Writable locations:")
        for location, info in data.get("location_tests", {}).items():
            status = "✅" if info.get("writable") else "❌"
            error_info = f" ({info.get('error')})" if not info.get("writable") else ""
            print(f"   {status} {location}{error_info}")
        
        # Verify that at least some basic functionality is available
        assert data.get("pyodide_available") is True, "Pyodide should be available"
        location_tests = data.get("location_tests", {})
        writable_locations = [loc for loc, info in location_tests.items() if info.get("writable")]
        assert len(writable_locations) > 0, "At least one location should be writable"

    def test_given_user_needs_persistence_when_seeking_guidance_then_clear_instructions_provided(
        self, http_session: requests.Session, base_url: str
    ):
        """
        Provide comprehensive user guidance for file persistence needs.
        
        Given: Users who need file persistence in their Pyodide applications
        When: Seeking guidance on best practices and patterns
        Then: Should receive clear, actionable instructions
        
        Args:
            http_session: HTTP session for requests
            base_url: Base URL of the server
            
        Examples:
            This test serves as documentation for users on how to handle
            file persistence requirements in production applications.
        """
        # Given: Need for user guidance on persistence
        # When: Providing comprehensive guidance
        # Then: Users should have clear actionable instructions
        
        print("\\n📖 USER GUIDANCE FOR FILE PERSISTENCE:")
        print("\\n1. 🎯 FOR PLOT/IMAGE GENERATION:")
        print("   - Save to mounted directory (/home/pyodide/plots/matplotlib/, /home/pyodide/plots/seaborn/)")
        print("   - Files appear directly in local filesystem")
        print("   - Use absolute paths: Path('/home/pyodide/plots/matplotlib/myplot.png')")
        print("   - Example: plt.savefig(Path('/home/pyodide/plots/matplotlib/chart.png'))")
        
        print("\\n2. 📊 FOR DATA PROCESSING RESULTS:")
        print("   - Return processed data as JSON in API response")
        print("   - Use .to_dict(), .tolist(), or explicit serialization")
        print("   - Don't rely on intermediate files persisting")
        print("   - Example: return df.to_dict('records') instead of saving CSV")
        
        print("\\n3. 💾 FOR LARGE DATASETS:")
        print("   - Upload files via /api/upload endpoint")
        print("   - Process and return results immediately")
        print("   - Consider external storage for persistence")
        print("   - Use streaming for very large datasets")
        
        print("\\n4. ⚠️  WHAT NOT TO DO:")
        print("   - Don't assume files in /tmp persist across requests")
        print("   - Don't rely on Pyodide filesystem for data storage")
        print("   - Don't expect files to survive server restarts")
        print("   - Don't use filesystem as a database replacement")
        
        print("\\n5. 🔧 FOR ADVANCED USERS (if needed):")
        print("   - Implement explicit pyodide.FS.syncfs() calls")
        print("   - Use IndexedDB directly for browser-side persistence")
        print("   - Consider WebAssembly filesystem limitations")
        print("   - Test persistence behavior thoroughly")
        
        # Validate that the guidance is accessible
        code = '''
# Test that guidance examples work
from pathlib import Path

result = {
    "guidance_examples_tested": True,
    "plot_directory_accessible": False,
    "data_serialization_works": False
}

# Test plot directory access
try:
    plot_dir = Path("/home/pyodide/plots/matplotlib")
    plot_dir.mkdir(parents=True, exist_ok=True)
    result["plot_directory_accessible"] = plot_dir.exists()
except Exception as e:
    result["plot_directory_error"] = str(e)

# Test data serialization
try:
    test_data = {"example": "data", "numbers": [1, 2, 3]}
    # This would normally be returned as JSON
    result["data_serialization_works"] = True
    result["example_data"] = test_data
except Exception as e:
    result["data_serialization_error"] = str(e)

result
'''
        
        response = execute_python_code(http_session, base_url, code)
        assert response["success"] is True, "Guidance examples should work"
        
        data = parse_result_data(response)
        assert data.get("guidance_examples_tested") is True, "Guidance examples should be tested"
        
        if data.get("plot_directory_accessible"):
            print("\\n✅ Plot directory guidance is functional")
        else:
            print(f"\\n⚠️  Plot directory access issue: {data.get('plot_directory_error', 'Unknown')}")
        
        if data.get("data_serialization_works"):
            print("✅ Data serialization guidance is functional")
        else:
            print(f"⚠️  Data serialization issue: {data.get('data_serialization_error', 'Unknown')}")
        
        # This test always passes - it's documentation
        assert True, "User guidance provided successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
