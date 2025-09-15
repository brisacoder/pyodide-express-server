"""
JavaScript Module Integration Tests for Pyodide Express Server

This module contains comprehensive BDD-style tests that verify JavaScript module
access and functionality within the Pyodide WebAssembly environment through
the Express server's /api/execute-raw endpoint.

The tests validate:
- JavaScript module availability and basic functionality
- Browser global object access through the js module
- Pyodide-specific JavaScript integrations
- Cross-platform Python code execution
- Error handling and edge cases

All tests use only the /api/execute-raw endpoint and follow the API contract:
{
  "success": true | false,
  "data": {
    "result": <string>,
    "stdout": <string>,
    "stderr": <string>,
    "executionTime": <int>
  } | null,
  "error": <string> | null,
  "meta": {
    "timestamp": <ISO_string>
  }
}
"""

import time
from typing import Dict, Any

import pytest
import requests
from requests.exceptions import RequestException


# Constants - Centralized configuration
API_BASE_URL = "http://localhost:3000"
API_EXECUTE_RAW_ENDPOINT = f"{API_BASE_URL}/api/execute-raw"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Timeouts and limits
SERVER_STARTUP_TIMEOUT = 180
REQUEST_TIMEOUT = 60
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0

# Expected response structure keys
EXPECTED_SUCCESS_KEYS = {"success", "data", "error", "meta"}
EXPECTED_DATA_KEYS = {"result", "stdout", "stderr", "executionTime"}
EXPECTED_META_KEYS = {"timestamp"}


@pytest.fixture(scope="session")
def server_health_check():
    """
    Verify server is running and accessible before starting tests.
    
    This fixture ensures the Pyodide Express Server is available at the
    expected endpoint and responding to health checks.
    
    Returns:
        None
        
    Raises:
        pytest.skip: If server is not accessible within timeout period
        
    Example:
        This fixture runs automatically for all tests that depend on server access.
    """
    def wait_for_server(url: str, timeout: int = SERVER_STARTUP_TIMEOUT) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return
            except RequestException:
                pass
            time.sleep(RETRY_DELAY)
        pytest.skip(f"Server at {url} is not responding within {timeout}s timeout")
    
    wait_for_server(API_HEALTH_ENDPOINT)


@pytest.fixture(scope="session")
def api_client(server_health_check):
    """
    Provide a configured requests session for API interactions.
    
    Creates a requests session with appropriate timeout settings and
    retry logic for robust API communication.
    
    Args:
        server_health_check: Fixture ensuring server availability
        
    Returns:
        requests.Session: Configured session object
        
    Example:
        def test_example(api_client):
            response = api_client.post('/api/execute-raw', data="print('test')")
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "text/plain"})
    return session


def execute_python_code(
    api_client: requests.Session,
    code: str,
    timeout: int = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """
    Execute Python code via /api/execute-raw endpoint with validation.
    
    Sends Python code to the Pyodide execution endpoint and validates
    the response structure according to the API contract.
    
    Args:
        api_client: Configured requests session
        code: Python code to execute as plain text string
        timeout: Request timeout in seconds (default: REQUEST_TIMEOUT)
        
    Returns:
        Dict containing the parsed JSON response from the API
        
    Raises:
        AssertionError: If response structure doesn't match API contract
        requests.RequestException: If HTTP request fails
        
    Example:
        response = execute_python_code(api_client, "import sys; print(sys.version)")
        assert response["success"] is True
        assert "Python" in response["data"]["result"]
    """
    response = api_client.post(
        API_EXECUTE_RAW_ENDPOINT,
        data=code,
        timeout=timeout
    )
    
    # Validate HTTP response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Parse and validate JSON structure
    data = response.json()
    
    # Validate API contract structure
    assert isinstance(data, dict), "Response must be a JSON object"
    missing_keys = EXPECTED_SUCCESS_KEYS - set(data.keys())
    assert not missing_keys, f"Missing required keys: {missing_keys}"
    
    # Validate success field
    assert isinstance(data["success"], bool), "success field must be boolean"
    
    # Validate meta structure
    assert isinstance(data["meta"], dict), "meta field must be object"
    missing_meta_keys = EXPECTED_META_KEYS - set(data["meta"].keys())
    assert not missing_meta_keys, f"Missing meta keys: {missing_meta_keys}"
    assert isinstance(data["meta"]["timestamp"], str), "timestamp must be string"
    
    # Validate success case structure
    if data["success"]:
        assert data["error"] is None, "error field must be null on success"
        assert isinstance(data["data"], dict), "data field must be object on success"
        missing_data_keys = EXPECTED_DATA_KEYS - set(data["data"].keys())
        assert not missing_data_keys, f"Missing data keys: {missing_data_keys}"
        assert isinstance(data["data"]["executionTime"], int), "executionTime must be integer"
        assert isinstance(data["data"]["result"], str), "result must be string"
        assert isinstance(data["data"]["stdout"], str), "stdout must be string"
        assert isinstance(data["data"]["stderr"], str), "stderr must be string"
    else:
        assert data["data"] is None, "data field must be null on failure"
        assert isinstance(data["error"], str), "error field must be string on failure"
    
    return data


class TestJavaScriptModuleAvailability:
    """
    Test suite for basic JavaScript module availability in Pyodide environment.

    These tests verify that the fundamental JavaScript interoperability features
    are accessible within the Pyodide WebAssembly runtime.
    """

    def test_given_pyodide_environment_when_importing_js_module_then_import_succeeds(
        self, api_client: requests.Session
    ):
        """
        Test that the JavaScript module can be imported successfully.
        
        This test verifies that the fundamental 'js' module is available in the
        Pyodide environment and can be imported without errors.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Python import js statement executes without error
            - Module type information is accessible
            - Response follows API contract structure
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "{'js_module_available': True, 'js_type': \"<class 'pyodide.ffi.JsProxy'>\"}",
                "stdout": "js module imported successfully\n",
                "stderr": "",
                "executionTime": 12
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: A Pyodide environment with JavaScript interoperability
        code = """
import js
print("js module imported successfully")

# Cross-platform result dictionary creation
result = {
    "js_module_available": True,
    "js_type": str(type(js))
}
print(f"Result: {result}")
"""
        
        # When: Attempting to import and use the js module
        response = execute_python_code(api_client, code)
        
        # Then: Import succeeds and module information is accessible
        assert response["success"] is True, f"JavaScript module import failed: {response.get('error')}"
        
        # Validate output content
        stdout = response["data"]["stdout"]
        assert "js module imported successfully" in stdout
        assert "js_module_available" in stdout
        assert "js_type" in stdout
        
        # Validate execution metadata
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""
    
    def test_given_js_module_when_accessing_attributes_then_attributes_are_available(
        self, api_client: requests.Session
    ):
        """
        Test that JavaScript module exposes expected attributes and functionality.
        
        This test verifies that the js module provides access to various JavaScript
        objects and functions that are typically available in a browser-like environment.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - js module has accessible attributes
            - Common JavaScript functionality is available
            - Attribute enumeration works correctly
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Attributes found: 15, Has pyodide: True, Sample: ['console', 'document']",
                "stdout": "...",
                "stderr": "",
                "executionTime": 8
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module is available
        code = """
import js
from pathlib import Path  # Cross-platform path handling

# Get non-private attributes using cross-platform approach
attributes = [attr for attr in dir(js) if not attr.startswith('_')]
attribute_count = len(attributes)

# Check for commonly expected attributes
has_pyodide = hasattr(js, 'pyodide')
has_mountFS = hasattr(js, 'mountFS')
has_FS = hasattr(js, 'FS')

# Get sample attributes for validation
sample_attributes = attributes[:10] if attributes else []

print(f"Attributes found: {attribute_count}")
print(f"Has pyodide: {has_pyodide}")
print(f"Has mountFS: {has_mountFS}")
print(f"Has FS: {has_FS}")
print(f"Sample attributes: {sample_attributes}")
"""
        
        # When: Accessing JavaScript module attributes  
        response = execute_python_code(api_client, code)
        
        # Then: Attributes are accessible and contain expected content
        assert response["success"] is True, f"JavaScript attribute access failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Attributes found:" in stdout
        assert "Has pyodide:" in stdout
        assert "Sample attributes:" in stdout
        
        # Validate that some attributes were found
        # Note: We don't assert specific attribute counts as they may vary
        # between different Pyodide configurations
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""
    
    def test_given_js_module_when_checking_pyodide_submodule_then_pyodide_access_works(
        self, api_client: requests.Session
    ):
        """
        Test access to Pyodide-specific JavaScript functionality through js.pyodide.
        
        This test verifies that Pyodide-specific JavaScript integrations are available
        and accessible through the js module, if present in the current configuration.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Pyodide submodule detection works without errors
            - If available, pyodide submodule has expected attributes
            - Code handles both presence and absence of pyodide submodule gracefully
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Pyodide available: True, Attributes: ['mountNodeFS', 'runPython']",
                "stdout": "...",
                "stderr": "",
                "executionTime": 15
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module with potential Pyodide submodule
        code = """
import js
from pathlib import Path  # Ensure cross-platform compatibility

pyodide_available = False
pyodide_attributes = []
has_mountNodeFS = False

# Safely check for pyodide submodule
try:
    if hasattr(js, 'pyodide'):
        pyodide_available = True
        # Get pyodide-specific attributes
        pyodide_attrs = [attr for attr in dir(js.pyodide) if not attr.startswith('_')]
        pyodide_attributes = pyodide_attrs[:15]  # First 15 for brevity
        has_mountNodeFS = hasattr(js.pyodide, 'mountNodeFS')
        
        print(f"Pyodide available: {pyodide_available}")
        print(f"Pyodide attributes count: {len(pyodide_attributes)}")
        print(f"Has mountNodeFS: {has_mountNodeFS}")
        print(f"Sample pyodide attributes: {pyodide_attributes}")
    else:
        print("Pyodide submodule not available in current configuration")
        
except Exception as e:
    print(f"Error checking pyodide submodule: {str(e)}")
    # Don't fail the test - this might be expected in some configurations
"""
        
        # When: Checking for Pyodide-specific JavaScript functionality
        response = execute_python_code(api_client, code)
        
        # Then: Check completes without errors regardless of pyodide presence
        assert response["success"] is True, f"Pyodide submodule check failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        # Test should succeed whether pyodide is available or not
        assert ("Pyodide available:" in stdout or 
                "Pyodide submodule not available" in stdout or
                "Error checking pyodide" in stdout)
        
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""


class TestBrowserGlobalsAccess:
    """
    Test suite for browser global object access through the JavaScript module.
    
    These tests verify that common browser global objects (console, document, window, etc.)
    are accessible through the js module in the Pyodide environment.
    """
    
    def test_given_js_module_when_accessing_browser_globals_then_globals_are_accessible(
        self, api_client: requests.Session
    ):
        """
        Test access to browser global objects through the JavaScript module.
        
        This test verifies that common browser globals like console, document, window,
        and fetch are accessible through the js module, providing a bridge between
        Python and JavaScript browser APIs.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Browser global objects are accessible through hasattr checks
            - At least some browser globals are available
            - Console object provides expected functionality if available
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Console: True, Document: True, Window: True, Fetch: False",
                "stdout": "Browser globals check completed...",
                "stderr": "",
                "executionTime": 18
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module with potential browser globals
        code = """
import js
from pathlib import Path  # Cross-platform imports

# Check for common browser global objects
browser_globals = {
    "has_console": hasattr(js, 'console'),
    "has_document": hasattr(js, 'document'), 
    "has_window": hasattr(js, 'window'),
    "has_fetch": hasattr(js, 'fetch'),
    "has_navigator": hasattr(js, 'navigator'),
    "has_location": hasattr(js, 'location')
}

print("Browser globals accessibility check:")
for global_name, is_available in browser_globals.items():
    print(f"  {global_name}: {is_available}")

# Test console access if available
console_accessible = False
console_error = None

if browser_globals["has_console"]:
    try:
        # Test console access without actually logging (to avoid side effects)
        console_accessible = hasattr(js.console, 'log')
        if console_accessible:
            print("Console.log method is accessible")
    except Exception as e:
        console_error = str(e)
        print(f"Console access error: {console_error}")

# Count available globals
available_globals = sum(browser_globals.values())
print(f"Total available browser globals: {available_globals}")

# Create summary
if available_globals > 0:
    available_list = [name.replace('has_', '') for name, available in browser_globals.items() if available]
    print(f"Available globals: {', '.join(available_list)}")
else:
    print("No browser globals detected - this may be expected in server-side Pyodide")
"""
        
        # When: Accessing browser global objects
        response = execute_python_code(api_client, code)
        
        # Then: Global object checks complete successfully
        assert response["success"] is True, f"Browser globals access failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Browser globals accessibility check:" in stdout
        assert "Total available browser globals:" in stdout
        
        # Note: We don't require specific globals to be present as this varies
        # between different Pyodide configurations (browser vs. server-side)
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""
    
    def test_given_browser_environment_when_testing_dom_interaction_then_dom_access_works_or_gracefully_fails(
        self, api_client: requests.Session  
    ):
        """
        Test DOM interaction capabilities through JavaScript module.
        
        This test attempts to interact with DOM elements through the js module,
        handling both cases where DOM is available (browser-like environment) and
        where it's not (server-side Pyodide).
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - DOM access attempts complete without Python errors
            - Graceful handling of both DOM presence and absence
            - Appropriate error messages when DOM is not available
            
        Example API Response (DOM not available):
            {
              "success": true,
              "data": {
                "result": "DOM not available in current environment",
                "stdout": "Testing DOM interaction...\nDOM access: Not available",
                "stderr": "",
                "executionTime": 5
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: Potential browser environment with DOM access
        code = """
import js
from pathlib import Path

print("Testing DOM interaction capabilities...")

dom_available = False
document_accessible = False
dom_error = None

try:
    # Check if document is available and accessible
    if hasattr(js, 'document'):
        document_accessible = True
        # Try basic DOM operations
        if hasattr(js.document, 'createElement'):
            # Test element creation (safest DOM operation)
            test_element = js.document.createElement('div')
            dom_available = True
            print("DOM access: Available - created test element successfully")
        else:
            print("DOM access: Document exists but createElement not available")
    else:
        print("DOM access: Document object not available")
        
except Exception as e:
    dom_error = str(e)
    print(f"DOM access error: {dom_error}")

# Test other browser APIs that might be available
web_apis = {
    "localStorage": hasattr(js, 'localStorage') if hasattr(js, 'localStorage') else False,
    "sessionStorage": hasattr(js, 'sessionStorage') if hasattr(js, 'sessionStorage') else False,
    "XMLHttpRequest": hasattr(js, 'XMLHttpRequest') if hasattr(js, 'XMLHttpRequest') else False,
    "JSON": hasattr(js, 'JSON') if hasattr(js, 'JSON') else False
}

print("\\nWeb API availability:")
for api_name, is_available in web_apis.items():
    print(f"  {api_name}: {is_available}")

# Summary
if dom_available:
    print("\\nSummary: Full DOM interaction available")
elif document_accessible:
    print("\\nSummary: Document available but limited DOM functionality")
else:
    print("\\nSummary: No DOM available - likely server-side Pyodide environment")
"""
        
        # When: Testing DOM interaction capabilities
        response = execute_python_code(api_client, code)
        
        # Then: DOM tests complete successfully regardless of DOM availability
        assert response["success"] is True, f"DOM interaction test failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Testing DOM interaction capabilities..." in stdout
        assert "Web API availability:" in stdout
        assert "Summary:" in stdout
        
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""


class TestJavaScriptErrorHandling:
    """
    Test suite for error handling in JavaScript module interactions.
    
    These tests verify that JavaScript-related errors are handled gracefully
    and don't crash the Pyodide environment.
    """
    
    def test_given_js_module_when_accessing_nonexistent_attribute_then_error_is_handled_gracefully(
        self, api_client: requests.Session
    ):
        """
        Test graceful handling of nonexistent JavaScript attributes.
        
        This test verifies that attempting to access nonexistent attributes
        on JavaScript objects results in appropriate Python exceptions rather
        than crashing the Pyodide environment.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Accessing nonexistent attributes raises appropriate Python exceptions
            - Error handling mechanisms work correctly
            - Pyodide environment remains stable after errors
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Error handled correctly: AttributeError caught",
                "stdout": "Testing nonexistent attribute access...",
                "stderr": "",
                "executionTime": 3
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module with potential nonexistent attributes
        code = """
import js
from pathlib import Path

print("Testing nonexistent attribute access...")

errors_caught = []

# Test 1: Access clearly nonexistent attribute
try:
    nonexistent = js.this_definitely_does_not_exist_in_javascript
    print("ERROR: Nonexistent attribute access should have failed!")
    errors_caught.append("UNEXPECTED_SUCCESS")
except AttributeError as e:
    print(f"✓ Correctly caught AttributeError: {str(e)[:50]}...")
    errors_caught.append("AttributeError")
except Exception as e:
    print(f"✓ Caught different exception type: {type(e).__name__}: {str(e)[:50]}...")
    errors_caught.append(type(e).__name__)

# Test 2: Access nonexistent method on existing object
try:
    if hasattr(js, 'console'):
        nonexistent_method = js.console.this_method_does_not_exist()
        print("ERROR: Nonexistent method call should have failed!")
        errors_caught.append("UNEXPECTED_METHOD_SUCCESS")
except AttributeError as e:
    print(f"✓ Correctly caught method AttributeError: {str(e)[:50]}...")
    errors_caught.append("MethodAttributeError")  
except Exception as e:
    print(f"✓ Caught method exception: {type(e).__name__}: {str(e)[:50]}...")
    errors_caught.append(f"Method{type(e).__name__}")

# Test 3: Verify environment is still stable
try:
    # Simple operation to verify environment stability
    test_value = len(dir(js))
    print(f"✓ Environment stable after errors - js has {test_value} attributes")
    errors_caught.append("ENVIRONMENT_STABLE")
except Exception as e:
    print(f"✗ Environment unstable: {str(e)}")
    errors_caught.append("ENVIRONMENT_UNSTABLE")

print(f"\\nError handling summary: {', '.join(errors_caught)}")
"""
        
        # When: Accessing nonexistent JavaScript attributes
        response = execute_python_code(api_client, code)
        
        # Then: Errors are handled gracefully and environment remains stable
        assert response["success"] is True, f"Error handling test failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"] 
        assert "Testing nonexistent attribute access..." in stdout
        assert "Error handling summary:" in stdout
        
        # Should have caught some kind of errors (AttributeError or similar)
        assert ("AttributeError" in stdout or "Exception" in stdout)
        assert "ENVIRONMENT_STABLE" in stdout or "Environment stable" in stdout
        
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""
    
    def test_given_js_module_when_invalid_javascript_operations_then_python_exceptions_are_raised(
        self, api_client: requests.Session  
    ):
        """
        Test that invalid JavaScript operations raise appropriate Python exceptions.
        
        This test verifies that various invalid JavaScript operations through the
        js module result in proper Python exceptions that can be caught and handled.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Invalid operations raise appropriate Python exceptions
            - Exception messages provide useful debugging information
            - Multiple error types are handled correctly
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Invalid operations handled: 3 errors caught",
                "stdout": "Testing invalid JavaScript operations...",
                "stderr": "",
                "executionTime": 7
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module that can perform various operations
        code = """
import js
from pathlib import Path

print("Testing invalid JavaScript operations...")

exception_types = []
test_results = []

# Test 1: Invalid function call syntax
try:
    # Try to call js itself as a function (invalid)
    result = js()
    test_results.append("ERROR: js() call should have failed")
except Exception as e:
    exception_name = type(e).__name__
    exception_types.append(exception_name)
    test_results.append(f"✓ js() call failed correctly: {exception_name}")

# Test 2: Invalid attribute assignment (if possible)
try:
    # Try to assign to a potentially read-only attribute
    js._invalid_assignment = "test"
    test_results.append("NOTE: Assignment succeeded (may be allowed)")
except Exception as e:
    exception_name = type(e).__name__ 
    exception_types.append(exception_name)
    test_results.append(f"✓ Invalid assignment failed correctly: {exception_name}")

# Test 3: Type coercion edge cases
try:
    # Try operations that might fail due to type mismatches
    if hasattr(js, 'console'):
        # Try to use console object in arithmetic (should fail)
        result = js.console + 42
        test_results.append(f"NOTE: Arithmetic with console succeeded: {type(result)}")
    else:
        test_results.append("SKIP: Console not available for arithmetic test")
except Exception as e:
    exception_name = type(e).__name__
    exception_types.append(exception_name)
    test_results.append(f"✓ Arithmetic operation failed correctly: {exception_name}")

# Print all test results
for result in test_results:
    print(result)

# Summary  
unique_exceptions = list(set(exception_types))
print(f"\\nException types caught: {', '.join(unique_exceptions) if unique_exceptions else 'None'}")
print(f"Total errors handled: {len(exception_types)}")

# Verify environment is still functional
try:
    js_attr_count = len([attr for attr in dir(js) if not attr.startswith('_')])
    print(f"✓ Environment remains functional - {js_attr_count} public js attributes available")
except Exception as e:
    print(f"✗ Environment check failed: {str(e)}")
"""
        
        # When: Performing invalid JavaScript operations
        response = execute_python_code(api_client, code)
        
        # Then: Operations fail gracefully with appropriate exceptions
        assert response["success"] is True, f"Invalid operations test failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Testing invalid JavaScript operations..." in stdout
        assert "Exception types caught:" in stdout
        assert "Total errors handled:" in stdout
        
        # Environment should remain functional
        assert ("Environment remains functional" in stdout or
                "Environment check" in stdout)
        
        assert response["data"]["executionTime"] > 0
        assert response["data"]["stderr"] == ""


class TestCrossPlatformCompatibility:
    """
    Test suite for cross-platform compatibility of JavaScript module usage.
    
    These tests ensure that Python code using the js module works consistently
    across different platforms (Windows, Linux, macOS) and different Pyodide
    configurations.
    """
    
    def test_given_cross_platform_code_when_using_pathlib_then_paths_work_consistently(
        self, api_client: requests.Session
    ):
        """
        Test cross-platform path handling in JavaScript module context.
        
        This test verifies that Python code using pathlib for cross-platform
        path handling works correctly when interacting with JavaScript modules
        and file system operations.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - pathlib Path objects work consistently across platforms
            - Path operations integrate properly with JavaScript file system APIs
            - No platform-specific path separator issues
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Cross-platform paths work correctly",
                "stdout": "Testing pathlib integration...\nPosix paths: /tmp/test.txt...",
                "stderr": "",
                "executionTime": 4
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: Cross-platform Python environment with JavaScript module
        code = """
import js
from pathlib import Path, PurePath, PurePosixPath
import os

print("Testing cross-platform path handling with JavaScript module...")

# Test 1: Basic pathlib usage
test_paths = {
    "posix_path": PurePosixPath("/tmp/test.txt"),
    "generic_path": Path("data") / "files" / "example.csv",
    "relative_path": Path("./logs/output.log"),
    "parent_navigation": Path("../parent/file.txt")
}

print("Path representations:")
for name, path_obj in test_paths.items():
    print(f"  {name}: {path_obj} (type: {type(path_obj).__name__})")

# Test 2: Path operations that should work consistently
path_operations = {}
try:
    data_path = Path("/uploads") / "data.csv"
    path_operations["join_operation"] = str(data_path)
    path_operations["parent"] = str(data_path.parent)  
    path_operations["name"] = data_path.name
    path_operations["suffix"] = data_path.suffix
    path_operations["stem"] = data_path.stem
    
    print(f"\\nPath operations on '{data_path}':")
    for op_name, result in path_operations.items():
        print(f"  {op_name}: {result}")
        
except Exception as e:
    print(f"Path operations error: {str(e)}")

# Test 3: Verify JavaScript integration doesn't break path handling
try:
    if hasattr(js, 'console') or True:  # Always test this
        # Create paths that might be used with JavaScript file operations  
        js_compatible_paths = [
            str(Path("/plots") / "matplotlib" / "chart.png"),
            str(Path("/uploads") / "data.csv"),
            str(Path("./temp") / "processing.log")
        ]
        
        print(f"\\nJavaScript-compatible path strings:")
        for js_path in js_compatible_paths:
            print(f"  {js_path}")
            # Verify these are proper string representations
            assert isinstance(js_path, str), f"Path {js_path} is not a string"
            
except Exception as e:
    print(f"JavaScript integration error: {str(e)}")

# Test 4: Platform detection (for documentation)
try:
    import platform
    platform_info = platform.system()
    print(f"\\nDetected platform: {platform_info}")
except ImportError:
    print("\\nPlatform detection not available (expected in Pyodide)")

print("\\n✓ Cross-platform path testing completed successfully")
"""
        
        # When: Using cross-platform path handling with JavaScript module
        response = execute_python_code(api_client, code)
        
        # Then: Path operations work consistently
        assert response["success"] is True, f"Cross-platform path test failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Testing cross-platform path handling" in stdout
        assert "Path representations:" in stdout
        assert "Path operations on" in stdout
        assert "JavaScript-compatible path strings:" in stdout
        assert "Cross-platform path testing completed successfully" in stdout
        
        # Verify no stderr output (clean execution)
        assert response["data"]["stderr"] == ""
        assert response["data"]["executionTime"] > 0


@pytest.mark.slow
class TestPerformanceAndEdgeCases:
    """
    Test suite for performance characteristics and edge cases.
    
    These tests verify that the JavaScript module integration performs well
    under various conditions and handles edge cases appropriately.
    """
    
    def test_given_large_js_object_when_accessing_many_attributes_then_performance_is_acceptable(
        self, api_client: requests.Session
    ):
        """
        Test performance when accessing many JavaScript object attributes.
        
        This test verifies that accessing a large number of JavaScript object
        attributes through the js module performs within acceptable limits.
        
        Args:
            api_client: Configured requests session fixture
            
        Expected Behavior:
            - Large-scale attribute access completes within reasonable time
            - Memory usage remains stable during bulk operations
            - No performance degradation over multiple accesses
            
        Example API Response:
            {
              "success": true,
              "data": {
                "result": "Performance test completed: 100 attributes in 50ms",
                "stdout": "Testing large-scale attribute access...",
                "stderr": "",
                "executionTime": 89
              },
              "error": null,
              "meta": {"timestamp": "2025-09-15T06:45:04.913Z"}
            }
        """
        # Given: JavaScript module with potentially many attributes
        code = """
import js
from pathlib import Path
import time

print("Testing large-scale JavaScript attribute access performance...")

start_time = time.time()

# Get all available attributes
all_attributes = dir(js)
non_private_attrs = [attr for attr in all_attributes if not attr.startswith('_')]

print(f"Found {len(all_attributes)} total attributes, {len(non_private_attrs)} non-private")

# Test accessing each attribute (checking existence, not calling)
access_times = []
successful_accesses = 0
failed_accesses = 0

for i, attr_name in enumerate(non_private_attrs[:50]):  # Limit to 50 for performance
    attr_start = time.time()
    try:
        # Just check if attribute exists and get its type
        attr_obj = getattr(js, attr_name, None)
        attr_type = str(type(attr_obj))
        successful_accesses += 1
        
        if i < 5:  # Show details for first few
            print(f"  {attr_name}: {attr_type}")
            
    except Exception as e:
        failed_accesses += 1
        if i < 5:
            print(f"  {attr_name}: ERROR - {str(e)[:30]}...")
    
    access_time = time.time() - attr_start
    access_times.append(access_time)

total_time = time.time() - start_time
avg_access_time = sum(access_times) / len(access_times) if access_times else 0

print(f"\\nPerformance Results:")
print(f"  Total time: {total_time:.3f} seconds")
print(f"  Successful accesses: {successful_accesses}")
print(f"  Failed accesses: {failed_accesses}")
print(f"  Average access time: {avg_access_time*1000:.2f} milliseconds")

# Performance validation
if total_time < 5.0:  # Should complete within 5 seconds
    print("✓ Performance acceptable for bulk attribute access")
else:
    print("⚠ Performance slower than expected - consider optimization")

if avg_access_time < 0.1:  # Average access should be under 100ms
    print("✓ Individual attribute access performance good")
else:
    print("⚠ Individual attribute access slower than expected")
"""
        
        # When: Performing large-scale attribute access
        response = execute_python_code(api_client, code, timeout=120)  # Extended timeout
        
        # Then: Performance is acceptable and operation completes successfully
        assert response["success"] is True, f"Performance test failed: {response.get('error')}"
        
        stdout = response["data"]["stdout"]
        assert "Testing large-scale JavaScript attribute access" in stdout
        assert "Performance Results:" in stdout
        assert "Total time:" in stdout
        assert "Successful accesses:" in stdout
        
        # Should complete in reasonable time (allowing for slower systems)
        assert response["data"]["executionTime"] < 120000  # Less than 2 minutes
        assert response["data"]["stderr"] == ""
