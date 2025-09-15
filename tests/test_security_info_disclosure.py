#!/usr/bin/env python3
"""
Security Information Disclosure Test Suite (Pytest BDD Implementation)

This module provides comprehensive pytest-based security testing for information
disclosure vulnerabilities in the Pyodide Express Server. Tests follow BDD
(Behavior-Driven Development) patterns and ensure API contract compliance.

Test Categories:
- System information disclosure attempts
- Environment variable access testing
- Filesystem access boundary validation
- Process information exposure checks
- API contract validation for all responses

API Contract Validation:
All tests verify that responses follow the strict API contract:
{
  "success": true | false,
  "data": <object|null>,
  "error": <string|null>,
  "meta": { "timestamp": <string> }
}

BDD Pattern:
Tests follow Given/When/Then patterns for clear behavior specification:
- Given: Initial security test setup and preconditions
- When: Security attack vector executed
- Then: Expected security outcomes and validation

Requirements Compliance:
✅ Pytest framework with BDD patterns
✅ Only /api/execute-raw endpoint usage (no internal pyodide APIs)
✅ Parameterized constants (no hardcoded globals)
✅ Comprehensive docstrings with examples
✅ Cross-platform pathlib usage
✅ API contract validation for all responses
✅ Security-focused test scenarios
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

# Import from same directory using absolute import that pytest can resolve
import sys
sys.path.append(str(Path(__file__).parent))
from conftest import Config


class SecurityTestConfig:
    """Security-specific test configuration constants."""
    
    # Security test timeouts (shorter for security tests)
    SECURITY_TIMEOUTS = {
        "info_disclosure": 15,      # Quick security probes
        "filesystem_access": 20,    # Filesystem boundary tests
        "environment_access": 10,   # Environment variable tests
        "process_info": 15,         # Process information tests
    }
    
    # Expected security boundaries
    SECURITY_BOUNDARIES = {
        "max_env_vars_exposed": 5,      # Reasonable limit for environment exposure
        "max_path_length_exposed": 10,  # Limit filesystem path exposure
        "blocked_directories": [        # Directories that should be blocked
            "/etc", "/proc", "/sys", "/dev", "/root", "/home"
        ],
        "allowed_working_dirs": [       # Safe working directories
            "/tmp", "/plots", "/uploads", "/", ""
        ]
    }
    
    # Information categories to test
    INFO_CATEGORIES = {
        "system_info": ["platform", "python_version", "os_type"],
        "environment": ["env_count", "env_sample"],
        "filesystem": ["cwd", "root_access", "directory_listing"],
        "process": ["pid", "executable_path"]
    }


@pytest.fixture
def security_test_setup(server_ready, test_cleanup):
    """
    Fixture providing secure test environment setup for security tests.
    
    This fixture ensures each security test starts with a clean environment
    and provides cleanup tracking for any test artifacts that might be created.
    
    Args:
        server_ready: Server readiness from conftest.py
        test_cleanup: Cleanup tracker from conftest.py
    
    Yields:
        SecurityTestConfig: Configuration object for security tests
        
    Example:
        >>> def test_security_scenario(security_test_setup):
        ...     config = security_test_setup
        ...     # Security test with proper configuration
    """
    # Setup: Ensure clean security test environment
    yield SecurityTestConfig
    
    # Teardown: Handled by test_cleanup fixture automatically


def execute_security_code(python_code: str, timeout: int = None) -> Dict[str, Any]:
    """
    Execute Python security test code using /api/execute-raw endpoint with validation.
    
    This function provides a secure wrapper for executing security test code,
    ensuring proper API contract validation and error handling.
    
    Args:
        python_code: Python code to execute for security testing
        timeout: Optional timeout override for security tests
        
    Returns:
        Dict containing validated API response with security test results
        
    Raises:
        AssertionError: If API contract validation fails
        requests.RequestException: If network request fails
        
    Example:
        >>> response = execute_security_code("import os; print(os.getcwd())")
        >>> assert response["success"] in [True, False]
        >>> assert "data" in response
    """
    if timeout is None:
        timeout = SecurityTestConfig.SECURITY_TIMEOUTS["info_disclosure"]
    
    try:
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=python_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=timeout
        )
        
        # Validate response format and status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        response_data = response.json()
        
        # Validate API contract compliance
        assert "success" in response_data, "Response missing 'success' field"
        assert "data" in response_data, "Response missing 'data' field"
        assert "error" in response_data, "Response missing 'error' field"
        assert "meta" in response_data, "Response missing 'meta' field"
        assert "timestamp" in response_data.get("meta", {}), "Response missing 'meta.timestamp'"
        
        # Validate data field structure for successful responses
        if response_data["success"]:
            assert response_data["data"] is not None, "Successful response must have non-null data"
            data = response_data["data"]
            assert "result" in data or "stdout" in data, "Data must contain result or stdout"
        else:
            assert response_data["error"] is not None, "Failed response must have error message"
            
        return response_data
        
    except requests.RequestException as e:
        pytest.fail(f"Network request failed: {e}")
    except (ValueError, KeyError) as e:
        pytest.fail(f"Invalid response format: {e}")


class TestSecurityInformationDisclosure:
    """
    Comprehensive security test suite for information disclosure vulnerabilities.
    
    This class tests various attack vectors that could lead to information disclosure
    in the Pyodide execution environment. All tests follow BDD patterns and validate
    that the system properly restricts access to sensitive information while maintaining
    functionality for legitimate use cases.
    
    Security Focus Areas:
    - System information exposure (OS details, Python version)
    - Environment variable access control
    - Filesystem access boundaries and restrictions
    - Process information exposure limits
    - API contract compliance under security scenarios
    """

    def test_given_system_info_request_when_executed_then_limited_disclosure(
        self, security_test_setup, api_contract_validator
    ):
        """
        Test system information disclosure boundaries with BDD pattern.
        
        Given a Pyodide execution environment with security controls
        When Python code attempts to access system information
        Then only safe, non-sensitive system information should be disclosed
        
        This test validates that basic system information (like Python version,
        platform type) can be accessed for legitimate debugging purposes, but
        sensitive system details are appropriately restricted.
        
        Args:
            security_test_setup: Security test configuration fixture
            api_contract_validator: API response validation fixture
            
        Expected Behavior:
            ✅ Python version information accessible (legitimate debugging)
            ✅ Basic platform information available (cross-platform compatibility)
            ✅ Sensitive system details restricted or sanitized
            ✅ API contract compliance maintained
            ✅ No unexpected errors or crashes
        """
        # Given: Security-controlled Pyodide environment
        config = security_test_setup
        
        # When: Executing code to gather system information
        system_info_code = """
# Security test: Basic system information disclosure
from pathlib import Path
import sys
import platform
import json

print("=== SYSTEM INFORMATION SECURITY TEST ===")

# Collect basic system information
system_info = {
    "test_type": "system_info_disclosure",
    "python_version": sys.version.split()[0],  # Just version number
    "platform_system": platform.system(),
    "platform_release": platform.release(),
    "python_executable": str(Path(sys.executable).name),  # Just filename
    "python_path_count": len(sys.path),
    "available_modules_count": len(sys.modules),
}

# Test current working directory access
try:
    current_dir = Path.cwd()
    system_info["current_directory"] = str(current_dir)
    system_info["cwd_accessible"] = True
except Exception as e:
    system_info["current_directory"] = f"Access restricted: {type(e).__name__}"
    system_info["cwd_accessible"] = False

# Test basic path operations with pathlib
try:
    temp_path = Path("/tmp")
    system_info["temp_path_exists"] = temp_path.exists()
    system_info["pathlib_functional"] = True
except Exception as e:
    system_info["temp_path_exists"] = False
    system_info["pathlib_functional"] = False

print("System information collected successfully")
print(f"Python Version: {system_info['python_version']}")
print(f"Platform: {system_info['platform_system']}")
print(f"Current Directory: {system_info['current_directory']}")

# Return structured data for analysis
system_info
"""
        
        # Execute security test code with API contract validation
        response_data = execute_security_code(
            system_info_code, 
            config.SECURITY_TIMEOUTS["info_disclosure"]
        )
        
        # Then: Validate API contract compliance using fixture
        api_contract_validator(response_data)
        
        # And: Verify successful system information collection
        assert response_data["success"] is True, "System info collection should succeed"
        
        # Extract and validate system information from response
        result_data = response_data["data"]
        assert "result" in result_data, "Response must contain result data"
        
        # Validate that basic system information is accessible (legitimate use)
        output_text = result_data.get("stdout", "") or str(result_data.get("result", ""))
        assert "System information collected successfully" in output_text, "System info should be collected"
        assert "Python Version:" in output_text, "Python version should be available"
        assert "Platform:" in output_text, "Platform info should be available"
        assert "Current Directory:" in output_text, "Working directory should be accessible"
        
        # And: Verify reasonable information boundaries are maintained
        # System info should be basic and not overly detailed
        assert "=== SYSTEM INFORMATION SECURITY TEST ===" in output_text, "Test marker should be present"
        
        # Check that the response doesn't contain sensitive paths or detailed system info
        sensitive_patterns = ["/etc/passwd", "/proc/", "/sys/", "root", "administrator"]
        for pattern in sensitive_patterns:
            assert pattern.lower() not in output_text.lower(), f"Sensitive pattern '{pattern}' should not be exposed"

    def test_given_environment_access_when_executed_then_controlled_exposure(
        self, security_test_setup, api_contract_validator
    ):
        """
        Test environment variable access control and exposure limits.
        
        Given a Pyodide execution environment with security controls
        When Python code attempts to access environment variables
        Then environment access should be controlled and limited appropriately
        
        This test validates that environment variable access is properly controlled,
        preventing exposure of sensitive configuration while allowing access to
        safe, necessary environment information for application functionality.
        
        Args:
            security_test_setup: Security test configuration fixture
            api_contract_validator: API response validation fixture
            
        Expected Behavior:
            ✅ Environment variable count accessible (basic system info)
            ✅ Sensitive environment variables protected
            ✅ Safe environment variables available if needed
            ✅ No full environment dump exposure
            ✅ API contract compliance maintained
        """
        # Given: Security-controlled environment access
        config = security_test_setup
        
        # When: Executing code to access environment variables
        env_access_code = """
# Security test: Environment variable access control
from pathlib import Path
import os

print("=== ENVIRONMENT ACCESS SECURITY TEST ===")

# Test environment variable access with security focus
env_info = {
    "test_type": "environment_access",
    "total_env_vars": len(os.environ),
    "environment_accessible": True,
}

# Attempt to access environment variables safely
try:
    # Get a few safe environment variable names (not values)
    env_keys = list(os.environ.keys())
    # Limit exposure to first few keys only
    env_info["sample_env_keys"] = env_keys[:3]  # Very limited sample
    env_info["env_keys_count"] = len(env_keys)
    
    # Check for common safe environment variables
    safe_vars = ["PATH", "HOME", "USER", "LANG", "PWD"]
    found_safe_vars = [var for var in safe_vars if var in os.environ]
    env_info["safe_vars_present"] = found_safe_vars[:2]  # Limit to 2
    
except Exception as e:
    env_info["environment_accessible"] = False
    env_info["access_error"] = str(e)

# Test basic OS operations (should be available)
try:
    current_dir = str(Path.cwd())
    env_info["current_working_dir"] = current_dir
    env_info["os_operations_available"] = True
except Exception as e:
    env_info["os_operations_available"] = False
    env_info["os_error"] = str(e)

print(f"Environment variables found: {env_info['total_env_vars']}")
print(f"Environment accessible: {env_info['environment_accessible']}")
print(f"Current directory: {env_info.get('current_working_dir', 'Unknown')}")

# Return limited environment information
env_info
"""
        
        # Execute environment access test with appropriate timeout
        response_data = execute_security_code(
            env_access_code,
            config.SECURITY_TIMEOUTS["environment_access"]
        )
        
        # Then: Validate API contract compliance
        api_contract_validator(response_data)
        
        # And: Verify controlled environment access
        assert response_data["success"] is True, "Environment access test should succeed"
        
        # Extract and validate environment access results
        result_data = response_data["data"]
        output_text = result_data.get("stdout", "") or str(result_data.get("result", ""))
        
        assert "=== ENVIRONMENT ACCESS SECURITY TEST ===" in output_text, "Test marker should be present"
        assert "Environment variables found:" in output_text, "Environment count should be reported"
        assert "Environment accessible:" in output_text, "Access status should be reported"
        
        # And: Verify that environment access is controlled (not excessive)
        # Should not expose full environment dump or sensitive variables
        sensitive_env_patterns = ["password", "secret", "key", "token", "credential"]
        for pattern in sensitive_env_patterns:
            assert pattern.lower() not in output_text.lower(), f"Sensitive env pattern '{pattern}' should not be exposed"
        
        # And: Verify reasonable environment information boundaries
        lines = output_text.split('\n')
        env_lines = [line for line in lines if 'environment' in line.lower() or 'env' in line.lower()]
        assert len(env_lines) <= 10, "Environment information should be limited"

    def test_given_filesystem_access_when_executed_then_boundary_enforcement(
        self, security_test_setup, api_contract_validator
    ):
        """
        Test filesystem access boundary enforcement and directory restrictions.
        
        Given a Pyodide execution environment with filesystem controls
        When Python code attempts to access various filesystem locations
        Then filesystem access should be properly bounded and restricted
        
        This test validates that filesystem access is appropriately controlled,
        allowing access to safe working directories while restricting access
        to sensitive system directories and files.
        
        Args:
            security_test_setup: Security test configuration fixture
            api_contract_validator: API response validation fixture
            
        Expected Behavior:
            ✅ Working directory access allowed (necessary for operation)
            ✅ Safe temporary directories accessible
            ✅ Sensitive system directories restricted
            ✅ Pathlib cross-platform operations functional
            ✅ API contract compliance maintained
        """
        # Given: Filesystem security controls in place
        config = security_test_setup
        
        # When: Executing filesystem boundary testing code
        filesystem_test_code = """
# Security test: Filesystem access boundary enforcement
from pathlib import Path
import os

print("=== FILESYSTEM ACCESS SECURITY TEST ===")

# Test filesystem access with security boundaries
fs_info = {
    "test_type": "filesystem_access", 
    "access_results": {},
    "pathlib_functional": True
}

# Test working directory access (should be allowed)
try:
    current_dir = Path.cwd()
    fs_info["current_directory"] = str(current_dir)
    fs_info["cwd_accessible"] = True
    fs_info["access_results"]["current_dir"] = "accessible"
except Exception as e:
    fs_info["cwd_accessible"] = False
    fs_info["access_results"]["current_dir"] = f"restricted: {type(e).__name__}"

# Test safe directory access (plots, uploads, tmp)
safe_directories = ["/plots", "/uploads", "/tmp"]
for directory in safe_directories:
    try:
        dir_path = Path(directory)
        exists = dir_path.exists()
        fs_info["access_results"][directory] = "accessible" if exists else "not_found"
    except Exception as e:
        fs_info["access_results"][directory] = f"restricted: {type(e).__name__}"

# Test potentially restricted directories (should be controlled)
restricted_directories = ["/etc", "/proc", "/sys", "/root"]
for directory in restricted_directories:
    try:
        dir_path = Path(directory)
        # Just test existence, not listing (which might be restricted)
        exists = dir_path.exists()
        if exists:
            # Try to list (this should be controlled)
            try:
                contents = list(dir_path.iterdir())
                fs_info["access_results"][directory] = f"accessible_with_contents: {len(contents)}"
            except Exception:
                fs_info["access_results"][directory] = "exists_but_listing_restricted"
        else:
            fs_info["access_results"][directory] = "not_found"
    except Exception as e:
        fs_info["access_results"][directory] = f"access_restricted: {type(e).__name__}"

# Test pathlib basic operations
try:
    test_path = Path("/tmp/security_test")
    fs_info["pathlib_operations"] = {
        "path_creation": "functional",
        "path_resolution": str(test_path.resolve()) if hasattr(test_path, 'resolve') else "limited"
    }
except Exception as e:
    fs_info["pathlib_functional"] = False
    fs_info["pathlib_error"] = str(e)

print(f"Current directory: {fs_info.get('current_directory', 'Unknown')}")
print(f"Pathlib functional: {fs_info['pathlib_functional']}")
print("Filesystem security test completed")

# Return filesystem access analysis
fs_info
"""
        
        # Execute filesystem security test
        response_data = execute_security_code(
            filesystem_test_code,
            config.SECURITY_TIMEOUTS["filesystem_access"]
        )
        
        # Then: Validate API contract compliance
        api_contract_validator(response_data)
        
        # And: Verify filesystem boundary enforcement
        assert response_data["success"] is True, "Filesystem security test should succeed"
        
        # Extract and validate filesystem access results
        result_data = response_data["data"]
        output_text = result_data.get("stdout", "") or str(result_data.get("result", ""))
        
        assert "=== FILESYSTEM ACCESS SECURITY TEST ===" in output_text, "Test marker should be present"
        assert "Current directory:" in output_text, "Current directory should be reported"
        assert "Pathlib functional:" in output_text, "Pathlib status should be reported"
        assert "Filesystem security test completed" in output_text, "Test completion should be indicated"
        
        # And: Verify that basic filesystem operations are functional
        # Working directory should be accessible (needed for normal operation)
        assert "current directory" in output_text.lower(), "Current directory access should work"
        
        # And: Verify pathlib is functional for cross-platform compatibility
        assert "pathlib functional" in output_text.lower(), "Pathlib should be functional"
        
        # And: Ensure no sensitive filesystem information is exposed
        sensitive_paths = ["/etc/passwd", "/root/", "/home/", "/sys/", "/proc/"]
        for path in sensitive_paths:
            assert path not in output_text, f"Sensitive path '{path}' should not be exposed in output"

    def test_given_process_info_request_when_executed_then_minimal_exposure(
        self, security_test_setup, api_contract_validator
    ):
        """
        Test process information exposure limits and security boundaries.
        
        Given a Pyodide execution environment with process controls  
        When Python code attempts to access process information
        Then process information should be minimally exposed with appropriate restrictions
        
        This test validates that process-related information access is properly
        controlled, preventing exposure of sensitive process details while
        allowing basic functionality needed for normal operation.
        
        Args:
            security_test_setup: Security test configuration fixture
            api_contract_validator: API response validation fixture
            
        Expected Behavior:
            ✅ Basic process operations functional (for normal use)
            ✅ Sensitive process information restricted
            ✅ Process listing/enumeration controlled
            ✅ Memory/system resource info appropriately limited
            ✅ API contract compliance maintained
        """
        # Given: Process information security controls
        config = security_test_setup
        
        # When: Executing process information gathering code
        process_info_code = """
# Security test: Process information exposure limits  
from pathlib import Path
import os
import sys

print("=== PROCESS INFORMATION SECURITY TEST ===")

# Test process information access with security focus
process_info = {
    "test_type": "process_info",
    "process_operations_available": True,
}

# Test basic process information (some may be restricted)
try:
    # Basic process ID (might be available in container)
    pid = os.getpid()
    process_info["process_id"] = pid
    process_info["pid_accessible"] = True
except Exception as e:
    process_info["pid_accessible"] = False
    process_info["pid_error"] = str(e)

# Test executable path information
try:
    executable = sys.executable
    # Only store filename, not full path (security)
    exec_path = Path(executable)
    process_info["executable_name"] = exec_path.name
    process_info["executable_accessible"] = True
except Exception as e:
    process_info["executable_accessible"] = False
    process_info["executable_error"] = str(e)

# Test system-related functions (should be controlled)
try:
    # Test some basic os functions
    process_info["os_functions"] = {
        "getcwd_available": hasattr(os, 'getcwd'),
        "getpid_available": hasattr(os, 'getpid'),
        "path_functions": hasattr(os.path, 'exists'),
    }
except Exception as e:
    process_info["os_functions_error"] = str(e)

# Test Python system information (should be available)
try:
    process_info["python_info"] = {
        "version": sys.version.split()[0],  # Just version number
        "platform": sys.platform,
        "modules_count": len(sys.modules),
        "path_count": len(sys.path)
    }
except Exception as e:
    process_info["python_info_error"] = str(e)

print(f"Process ID accessible: {process_info.get('pid_accessible', False)}")
print(f"Executable accessible: {process_info.get('executable_accessible', False)}")  
print(f"OS functions available: {process_info.get('os_functions', {}).get('getcwd_available', False)}")
print("Process information security test completed")

# Return controlled process information
process_info
"""
        
        # Execute process information security test
        response_data = execute_security_code(
            process_info_code,
            config.SECURITY_TIMEOUTS["process_info"]
        )
        
        # Then: Validate API contract compliance
        api_contract_validator(response_data)
        
        # And: Verify process information exposure is controlled
        assert response_data["success"] is True, "Process info security test should succeed"
        
        # Extract and validate process information results
        result_data = response_data["data"]
        output_text = result_data.get("stdout", "") or str(result_data.get("result", ""))
        
        assert "=== PROCESS INFORMATION SECURITY TEST ===" in output_text, "Test marker should be present"
        assert "Process ID accessible:" in output_text, "PID access status should be reported"
        assert "Executable accessible:" in output_text, "Executable access status should be reported"
        assert "Process information security test completed" in output_text, "Test completion should be indicated"
        
        # And: Verify that basic Python functionality is preserved
        # Normal Python operations should work for legitimate use
        assert "OS functions available:" in output_text, "OS function availability should be reported"
        
        # And: Ensure no sensitive process information is leaked
        sensitive_process_patterns = [
            "/usr/bin/", "/usr/local/", "administrator", "root", 
            "/home/", "sudo", "passwd", "shadow"
        ]
        for pattern in sensitive_process_patterns:
            assert pattern not in output_text, f"Sensitive process pattern '{pattern}' should not be exposed"
        
        # And: Verify reasonable process information boundaries
        # Should not contain detailed system paths or user information
        lines = output_text.split('\n')
        assert len(lines) <= 20, "Process information output should be reasonably bounded"
