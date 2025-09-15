"""
Security Logging Test Suite for Pyodide Express Server.

This module implements comprehensive testing for security logging functionality
using pytest and BDD (Behavior-Driven Development) patterns.

Test Coverage:
1. Security event logging (code execution, authentication, access control)
2. Statistics collection and accuracy validation
3. Dashboard endpoints functionality and security
4. Error tracking and IP address monitoring  
5. Security audit trail generation and integrity
6. Performance metrics tracking for security events
7. Cross-platform compatibility testing

Requirements Compliance:
✅ Pytest framework with BDD patterns
✅ Comprehensive docstrings with descriptions, inputs, outputs, examples
✅ No hardcoded globals - configurable constants and fixtures
✅ Only /api/execute-raw endpoint usage (no internal 'pyodide' APIs)
✅ Pathlib for cross-platform file operations
✅ API contract validation for all responses
✅ Proper error handling and edge case coverage

API Contract Format:
{
  "success": true | false,
  "data": <object|null>,
  "error": <string|null>,
  "meta": { "timestamp": <string> }
}

Authors: Security Testing Team
Version: 2.0.0 (Pytest/BDD Conversion)
Last Updated: 2025-01-XX
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

from tests.conftest import Config, validate_api_contract


# Test Configuration Constants
class SecurityTestConfig:
    """Centralized configuration for security logging tests."""
    
    # API endpoints for security testing
    ENDPOINTS = {
        "execute_raw": "/api/execute-raw",
        "dashboard_stats": "/api/dashboard/stats", 
        "dashboard_clear": "/api/dashboard/stats/clear",
        "dashboard_html": "/api/dashboard/stats/dashboard",
        "legacy_stats": "/api/stats",
        "upload": "/api/upload",
        "uploaded_files": "/api/uploaded-files",
    }
    
    # Security test timeouts (in seconds)
    TIMEOUTS = {
        "code_execution": 30,
        "api_request": 10,
        "dashboard_load": 15,
        "file_upload": 20,
    }
    
    # Test data constants
    TEST_USER_AGENT = "SecurityLoggingTestSuite/2.0.0"
    MAX_EXECUTION_COUNT = 10
    STATISTICS_RESET_WAIT_TIME = 1
    
    # Security validation patterns
    EXPECTED_ERROR_TYPES = [
        "SyntaxError",
        "NameError", 
        "ZeroDivisionError",
        "TypeError",
        "ValueError"
    ]
    
    # Dashboard validation patterns
    DASHBOARD_REQUIRED_ELEMENTS = [
        "Pyodide Express Server Dashboard",
        "chart.js",
        "canvas"
    ]


@pytest.fixture
def security_test_cleanup():
    """
    Provide cleanup functionality for security test artifacts.
    
    This fixture ensures all security test artifacts are properly cleaned up
    to prevent data contamination between tests and maintain test isolation.
    
    Yields:
        SecurityCleanupTracker: Object to track security test resources
        
    Example:
        >>> def test_security_feature(security_test_cleanup):
        ...     cleanup = security_test_cleanup
        ...     # Perform security test
        ...     cleanup.track_upload("test_file.csv")
        ...     # Resources automatically cleaned up after test
    """
    class SecurityCleanupTracker:
        """Tracks and cleans up security test artifacts."""
        
        def __init__(self):
            self.uploaded_files = []
            self.temp_files = []
            self.temp_dirs = []
            self.start_time = time.time()
            
        def track_upload(self, filename: str) -> None:
            """Track uploaded file for API cleanup."""
            self.uploaded_files.append(filename)
            
        def track_temp_file(self, filepath: Path) -> None:
            """Track temporary file for filesystem cleanup."""
            self.temp_files.append(filepath)
            
        def track_temp_dir(self, dirpath: Path) -> None:
            """Track temporary directory for cleanup."""
            self.temp_dirs.append(dirpath)
            
        def clear_statistics(self) -> bool:
            """Clear server statistics for clean test state."""
            try:
                response = requests.post(
                    f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_clear']}", 
                    timeout=SecurityTestConfig.TIMEOUTS["api_request"]
                )
                return response.status_code == 200
            except requests.RequestException:
                return False
                
        def cleanup(self) -> None:
            """Clean up all tracked security test artifacts."""
            # Clean up uploaded files via API
            for filename in self.uploaded_files:
                try:
                    requests.delete(
                        f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['uploaded_files']}/{filename}",
                        timeout=5
                    )
                except requests.RequestException:
                    pass  # Best effort cleanup
                    
            # Clean up temporary files
            for temp_file in self.temp_files:
                if temp_file.exists() and temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass  # Best effort cleanup
                        
            # Clean up directories (reverse order for nested)
            for temp_dir in reversed(self.temp_dirs):
                if temp_dir.exists() and temp_dir.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except OSError:
                        pass  # Best effort cleanup
    
    tracker = SecurityCleanupTracker()
    # Clear statistics before test for clean state
    tracker.clear_statistics()
    time.sleep(SecurityTestConfig.STATISTICS_RESET_WAIT_TIME)
    
    yield tracker
    tracker.cleanup()


@pytest.fixture
def sample_python_codes():
    """
    Provide sample Python code snippets for security testing.
    
    Returns a collection of Python code examples for testing various
    security scenarios including successful executions, errors, and
    edge cases with cross-platform pathlib usage.
    
    Returns:
        Dict[str, str]: Collection of categorized Python code samples
        
    Example:
        >>> def test_code_execution(sample_python_codes):
        ...     success_code = sample_python_codes["simple_success"]
        ...     # Execute and validate response
    """
    return {
        "simple_success": "print('Security logging test - simple execution')",
        "syntax_error": "invalid_syntax_here !@#$",
        "name_error": "print(undefined_variable_name)",
        "zero_division": "result = 1 / 0  # This will cause ZeroDivisionError",
        "medium_complexity": """
import time
from pathlib import Path

# Cross-platform path handling
temp_path = Path('/tmp')
result = sum(range(1000))
print(f'Medium complexity execution completed: {result}')
        """.strip(),
        "complex_execution": """
from pathlib import Path
import json

# Complex data processing with pathlib
data = [i**2 for i in range(100)]
stats = {
    'count': len(data),
    'sum': sum(data),
    'max': max(data),
    'min': min(data)
}

# Cross-platform path operations
output_dir = Path('/tmp/security_test')
output_dir.mkdir(parents=True, exist_ok=True)

print(f'Complex execution stats: {json.dumps(stats)}')
        """.strip(),
        "pathlib_demonstration": """
from pathlib import Path

# Demonstrate cross-platform pathlib usage
base_path = Path('/plots/matplotlib')
base_path.mkdir(parents=True, exist_ok=True)

test_file = base_path / 'security_test_output.txt'
test_file.write_text('Security test pathlib demonstration')

print(f'File created at: {test_file}')
print(f'File exists: {test_file.exists()}')
        """.strip()
    }


def execute_secure_python_code(code: str, timeout: int = None, user_agent: str = None) -> Dict[str, Any]:
    """
    Execute Python code securely via /api/execute-raw with validation.
    
    This function provides a secure wrapper for executing Python code
    through the API with comprehensive validation and error handling.
    
    Args:
        code: Python code to execute
        timeout: Request timeout in seconds (default from config)
        user_agent: Custom User-Agent string for tracking
        
    Returns:
        Dict[str, Any]: Validated API response following contract format
        
    Raises:
        requests.RequestException: If request fails
        AssertionError: If response doesn't match API contract
        
    Example:
        >>> result = execute_secure_python_code("print('Hello World')")
        >>> assert result["success"] is True
        >>> assert "Hello World" in result["data"]["stdout"]
    """
    if timeout is None:
        timeout = SecurityTestConfig.TIMEOUTS["code_execution"]
        
    headers = {"Content-Type": "text/plain"}
    if user_agent:
        headers["User-Agent"] = user_agent
        
    response = requests.post(
        f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['execute_raw']}",
        headers=headers,
        data=code,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    validate_api_contract(result)
    return result


def get_security_statistics() -> Dict[str, Any]:
    """
    Retrieve current security statistics from dashboard API.
    
    Fetches and validates security statistics from the dashboard
    endpoint with proper error handling and contract validation.
    
    Returns:
        Dict[str, Any]: Statistics data from dashboard API
        
    Raises:
        requests.RequestException: If API request fails
        AssertionError: If response doesn't match expected format
        
    Example:
        >>> stats = get_security_statistics()
        >>> assert "overview" in stats
        >>> assert "totalExecutions" in stats["overview"]
    """
    response = requests.get(
        f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_stats']}",
        timeout=SecurityTestConfig.TIMEOUTS["api_request"]
    )
    response.raise_for_status()
    
    data = response.json()
    assert "success" in data, "Dashboard response missing 'success' field"
    assert data["success"] is True, f"Dashboard request failed: {data.get('error', 'Unknown error')}"
    assert "stats" in data, "Dashboard response missing 'stats' field"
    
    return data["stats"]


class TestSecurityLoggingBehaviors:
    """BDD test scenarios for security logging functionality."""
    
    @pytest.mark.security
    @pytest.mark.api
    def test_given_valid_python_code_when_executed_then_security_event_is_logged(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Valid Python code execution generates security log entry.
        
        Given: A valid Python code snippet for execution
        When: The code is executed via /api/execute-raw endpoint
        Then: The execution is successfully logged with security metadata
        And: Statistics are updated with execution details
        And: IP address and User-Agent are tracked
        
        Description:
            This test validates that successful code execution events are properly
            logged in the security system with all required metadata including
            execution time, IP tracking, and user agent information.
            
        Input:
            - Python code: Simple print statement
            - Content-Type: text/plain
            - Custom User-Agent for tracking
            
        Output:
            - success: True
            - data.result: Execution output
            - data.stdout: Standard output stream
            - data.stderr: Standard error stream (empty)
            - data.executionTime: Positive integer
            - meta.timestamp: ISO timestamp
            
        Example:
            >>> # Input
            >>> code = "print('Security logging test')"
            >>> # Expected Output
            >>> {
            ...   "success": True,
            ...   "data": {
            ...     "result": "Security logging test\\n",
            ...     "stdout": "Security logging test\\n", 
            ...     "stderr": "",
            ...     "executionTime": 125
            ...   },
            ...   "error": None,
            ...   "meta": {"timestamp": "2025-01-XX..."}
            ... }
        """
        # Given: Valid Python code for security logging test
        test_code = sample_python_codes["simple_success"]
        custom_user_agent = SecurityTestConfig.TEST_USER_AGENT
        
        # When: Code is executed via secure endpoint
        result = execute_secure_python_code(test_code, user_agent=custom_user_agent)
        
        # Then: Execution is successful with proper contract compliance
        assert result["success"] is True, f"Code execution failed: {result.get('error')}"
        assert result["error"] is None, "Successful execution should have null error"
        assert result["data"] is not None, "Successful execution should have data"
        
        # And: Data structure contains all required fields
        data = result["data"]
        assert "result" in data, "Response data missing 'result' field"
        assert "stdout" in data, "Response data missing 'stdout' field"
        assert "stderr" in data, "Response data missing 'stderr' field"
        assert "executionTime" in data, "Response data missing 'executionTime' field"
        
        # And: Execution output is captured correctly
        assert "Security logging test" in data["stdout"], f"Expected output not found in stdout: {data['stdout']}"
        assert isinstance(data["executionTime"], (int, float)), f"executionTime must be numeric: {type(data['executionTime'])}"
        assert data["executionTime"] > 0, f"executionTime must be positive: {data['executionTime']}"
        
        # And: Security statistics are updated
        stats = get_security_statistics()
        overview = stats["overview"]
        assert overview["totalExecutions"] >= 1, f"Statistics not updated: {overview['totalExecutions']}"
        
        # And: IP tracking is functional
        top_ips = stats.get("topIPs", [])
        assert len(top_ips) > 0, "IP tracking not functioning"
        
        # And: User-Agent tracking is functional
        user_agents = stats.get("userAgents", [])
        assert len(user_agents) > 0, "User-Agent tracking not functioning"
        agent_strings = [ua.get("agent", "") for ua in user_agents]
        assert any(SecurityTestConfig.TEST_USER_AGENT in agent for agent in agent_strings), \
            f"Custom user agent not tracked: {agent_strings}"

    @pytest.mark.security
    @pytest.mark.api
    @pytest.mark.error_handling
    def test_given_invalid_python_code_when_executed_then_errors_are_properly_tracked(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Invalid Python code execution generates proper error logging.
        
        Given: Multiple types of invalid Python code (syntax errors, name errors)
        When: Each code snippet is executed via /api/execute-raw endpoint
        Then: Execution fails with proper error messages
        And: Error types are correctly categorized and tracked
        And: Statistics reflect the error counts and types
        
        Description:
            This test validates that Python execution errors are properly caught,
            categorized, and logged in the security system. It ensures that
            different error types are tracked separately for analysis.
            
        Input:
            - Various invalid Python code snippets
            - Content-Type: text/plain headers
            
        Output:
            - success: False for all invalid code
            - data: null for error responses
            - error: String containing error message
            - Statistics updated with error categorization
            
        Example:
            >>> # Input: Syntax Error
            >>> code = "invalid_syntax !@#"
            >>> # Expected Output
            >>> {
            ...   "success": False,
            ...   "data": None,
            ...   "error": "SyntaxError: invalid syntax",
            ...   "meta": {"timestamp": "2025-01-XX..."}
            ... }
        """
        # Given: Multiple types of invalid Python code
        error_test_cases = [
            ("syntax_error", "SyntaxError"),
            ("name_error", "NameError"), 
            ("zero_division", "ZeroDivisionError")
        ]
        
        executed_errors = []
        
        # When: Each invalid code snippet is executed
        for code_key, expected_error_type in error_test_cases:
            test_code = sample_python_codes[code_key]
            result = execute_secure_python_code(test_code)
            
            # Then: Execution fails with proper error structure
            assert result["success"] is False, f"Invalid code should fail: {code_key}"
            assert result["data"] is None, f"Error response should have null data: {code_key}"
            assert result["error"] is not None, f"Error response should have error message: {code_key}"
            assert isinstance(result["error"], str), f"Error must be string: {type(result['error'])}"
            
            executed_errors.append(result["error"])
        
        # And: Statistics reflect the error tracking
        stats = get_security_statistics()
        overview = stats["overview"]
        top_errors = stats.get("topErrors", [])
        
        # And: Total executions include all attempts
        assert overview["totalExecutions"] >= len(error_test_cases), \
            f"Statistics should reflect all executions: {overview['totalExecutions']} >= {len(error_test_cases)}"
        
        # And: Success rate reflects the failures
        success_rate = float(overview.get("successRate", "0"))
        assert success_rate == 0.0, f"Success rate should be 0% for all failures: {success_rate}"
        
        # And: Error types are properly categorized
        assert len(top_errors) > 0, "Error tracking should contain entries"
        tracked_error_types = [error.get("error", "") for error in top_errors]
        
        # Verify that our expected error types are present
        for expected_type in ["syntax", "name", "division"]:
            assert any(expected_type.lower() in tracked_error.lower() for tracked_error in tracked_error_types), \
                f"Error type '{expected_type}' not found in tracked errors: {tracked_error_types}"

    @pytest.mark.security
    @pytest.mark.integration
    def test_given_multiple_executions_when_performed_over_time_then_statistics_accuracy_is_maintained(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Multiple code executions maintain statistical accuracy over time.
        
        Given: A series of successful and failed Python code executions
        When: Multiple operations are performed sequentially
        Then: Statistics accurately reflect the execution history
        And: Success rates are calculated correctly
        And: Error categorization remains accurate
        And: Execution time tracking is maintained
        
        Description:
            This test validates the accuracy and consistency of security statistics
            over multiple operations, ensuring that the logging system maintains
            data integrity during extended usage periods.
            
        Input:
            - Multiple Python code snippets (3 successful, 1 failed)
            - Sequential execution pattern
            
        Output:
            - Accurate statistical tracking
            - Correct success rate calculation (75%)
            - Proper error categorization
            - Consistent execution time metrics
            
        Example:
            >>> # After 3 successful + 1 failed execution
            >>> stats = {
            ...   "totalExecutions": 4,
            ...   "successRate": "75.0",
            ...   "averageExecutionTime": 150
            ... }
        """
        # Given: Clear initial statistics state
        initial_stats = get_security_statistics()
        initial_count = initial_stats.get("overview", {}).get("totalExecutions", 0)
        
        # When: Multiple successful executions are performed
        successful_executions = 3
        for i in range(successful_executions):
            test_code = f"result = {i} * 2\nprint(f'Execution {{i}}: {{result}}')"
            result = execute_secure_python_code(test_code)
            assert result["success"] is True, f"Execution {i} should succeed"
        
        # And: One failed execution is performed
        failed_code = sample_python_codes["zero_division"]
        failed_result = execute_secure_python_code(failed_code)
        assert failed_result["success"] is False, "Zero division should fail"
        
        # Then: Statistics accurately reflect all operations
        final_stats = get_security_statistics()
        overview = final_stats["overview"]
        
        expected_total = initial_count + successful_executions + 1
        actual_total = overview["totalExecutions"]
        assert actual_total == expected_total, \
            f"Total executions mismatch: expected {expected_total}, got {actual_total}"
        
        # And: Success rate is calculated correctly
        expected_success_rate = (successful_executions / (successful_executions + 1)) * 100
        actual_success_rate = float(overview.get("successRate", "0"))
        assert abs(actual_success_rate - expected_success_rate) < 0.1, \
            f"Success rate mismatch: expected {expected_success_rate}%, got {actual_success_rate}%"
        
        # And: Error tracking includes the zero division error
        top_errors = final_stats.get("topErrors", [])
        assert len(top_errors) > 0, "Error tracking should contain entries"
        
        error_messages = [error.get("error", "") for error in top_errors]
        assert any("division by zero" in error.lower() for error in error_messages), \
            f"ZeroDivisionError not tracked: {error_messages}"
        
        # And: Average execution time is tracked
        avg_time = overview.get("averageExecutionTime")
        assert avg_time is not None, "Average execution time should be tracked"
        assert isinstance(avg_time, (int, float)), f"Average time must be numeric: {type(avg_time)}"
        assert avg_time > 0, f"Average execution time must be positive: {avg_time}"

    @pytest.mark.security
    @pytest.mark.performance
    def test_given_hourly_tracking_enabled_when_multiple_executions_occur_then_trends_are_recorded(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Hourly execution trends are properly tracked and maintained.
        
        Given: An active security logging system with hourly trend tracking
        When: Multiple code executions are performed within the current hour
        Then: Hourly trend data is updated correctly
        And: Trend data structure is valid (24-hour format)
        And: Recent executions are properly counted
        
        Description:
            This test validates the hourly trend tracking functionality of the
            security logging system, ensuring that execution patterns are
            properly recorded for analysis and monitoring.
            
        Input:
            - Series of Python code executions
            - Time-based execution pattern
            
        Output:
            - 24-hour trend data array
            - Accurate execution counts per hour
            - Proper data structure validation
            
        Example:
            >>> # Expected hourly trend structure
            >>> hourly_trend = [0, 0, 0, ..., 5, 0, ...]  # 24 integers
            >>> assert len(hourly_trend) == 24
            >>> assert sum(hourly_trend) == total_recent_executions
        """
        # Given: Number of test executions to perform
        execution_count = 5
        test_codes = [
            sample_python_codes["simple_success"],
            sample_python_codes["medium_complexity"],
            sample_python_codes["pathlib_demonstration"]
        ]
        
        # When: Multiple executions are performed
        for i in range(execution_count):
            code_index = i % len(test_codes)
            test_code = test_codes[code_index]
            result = execute_secure_python_code(test_code)
            assert result["success"] is True, f"Execution {i} should succeed"
        
        # Then: Hourly trend data is properly structured
        stats = get_security_statistics()
        hourly_trend = stats.get("hourlyTrend", [])
        
        # And: Trend data has correct 24-hour structure
        assert len(hourly_trend) == 24, f"Hourly trend should have 24 entries: {len(hourly_trend)}"
        assert all(isinstance(count, int) for count in hourly_trend), \
            f"All trend values must be integers: {[type(v) for v in hourly_trend]}"
        
        # And: Recent executions are tracked in trends
        total_recent = sum(hourly_trend)
        assert total_recent >= execution_count, \
            f"Trend should include recent executions: {total_recent} >= {execution_count}"
        
        # And: All values are non-negative
        assert all(count >= 0 for count in hourly_trend), \
            f"All trend values must be non-negative: {hourly_trend}"

    @pytest.mark.security
    @pytest.mark.api
    def test_given_dashboard_endpoints_when_accessed_then_all_endpoints_function_correctly(
        self, server_ready, security_test_cleanup
    ):
        """
        Test Scenario: All dashboard endpoints function correctly and securely.
        
        Given: Active security logging dashboard system
        When: Each dashboard endpoint is accessed
        Then: Stats endpoint returns proper JSON data
        And: HTML dashboard endpoint returns valid HTML content
        And: Stats clear endpoint functions correctly
        And: All responses follow API contract format
        
        Description:
            This test validates the complete dashboard API functionality,
            ensuring all endpoints are accessible, functional, and secure.
            It verifies both JSON API endpoints and HTML dashboard rendering.
            
        Input:
            - GET requests to dashboard endpoints
            - POST request to clear statistics
            
        Output:
            - Valid JSON responses for API endpoints
            - Valid HTML content for dashboard
            - Proper HTTP status codes
            - API contract compliance
            
        Example:
            >>> # Stats endpoint response
            >>> {
            ...   "success": True,
            ...   "stats": { "overview": {...}, "topIPs": [...] },
            ...   "timestamp": "2025-01-XX..."
            ... }
        """
        # Given: Execute some code to generate dashboard data
        test_code = "print('Dashboard functionality test')"
        execute_secure_python_code(test_code)
        
        # When: Main dashboard stats endpoint is accessed
        stats_response = requests.get(
            f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_stats']}",
            timeout=SecurityTestConfig.TIMEOUTS["dashboard_load"]
        )
        
        # Then: Stats endpoint returns valid data
        assert stats_response.status_code == 200, f"Stats endpoint failed: {stats_response.status_code}"
        stats_data = stats_response.json()
        
        # And: Response follows expected structure
        assert "success" in stats_data, "Stats response missing 'success' field"
        assert stats_data["success"] is True, f"Stats request failed: {stats_data.get('error')}"
        assert "stats" in stats_data, "Stats response missing 'stats' field"
        assert "timestamp" in stats_data, "Stats response missing timestamp"
        
        # And: Stats data contains required sections
        stats = stats_data["stats"]
        required_sections = ["overview", "topIPs", "userAgents", "hourlyTrend"]
        for section in required_sections:
            assert section in stats, f"Stats missing required section: {section}"
        
        # When: HTML dashboard endpoint is accessed
        dashboard_response = requests.get(
            f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_html']}",
            timeout=SecurityTestConfig.TIMEOUTS["dashboard_load"]
        )
        
        # Then: Dashboard returns valid HTML
        assert dashboard_response.status_code == 200, f"Dashboard HTML failed: {dashboard_response.status_code}"
        content_type = dashboard_response.headers.get("Content-Type", "")
        assert "text/html" in content_type, f"Expected HTML content type: {content_type}"
        
        # And: HTML contains required dashboard elements
        html_content = dashboard_response.text
        for element in SecurityTestConfig.DASHBOARD_REQUIRED_ELEMENTS:
            assert element.lower() in html_content.lower(), \
                f"Dashboard missing required element: {element}"
        
        # When: Stats clear endpoint is accessed
        clear_response = requests.post(
            f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_clear']}",
            timeout=SecurityTestConfig.TIMEOUTS["api_request"]
        )
        
        # Then: Clear operation succeeds
        assert clear_response.status_code == 200, f"Stats clear failed: {clear_response.status_code}"
        clear_data = clear_response.json()
        assert "success" in clear_data, "Clear response missing 'success' field"
        assert clear_data["success"] is True, f"Clear operation failed: {clear_data.get('error')}"
        assert "message" in clear_data, "Clear response missing confirmation message"

    @pytest.mark.security
    @pytest.mark.api
    def test_given_legacy_compatibility_requirements_when_stats_accessed_then_backward_compatibility_maintained(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Legacy stats endpoint maintains backward compatibility.
        
        Given: Requirements for backward compatibility with legacy systems
        When: Legacy /api/stats endpoint is accessed after code execution
        Then: Response contains all legacy fields at top level
        And: Enhanced statistics are available under executionStats
        And: Data structure matches expected legacy format
        
        Description:
            This test ensures that existing integrations continue to function
            while new enhanced features are available through the extended
            API structure.
            
        Input:
            - GET request to /api/stats endpoint
            - Prior code execution for data generation
            
        Output:
            - Legacy fields: uptime, memory, pyodide, timestamp
            - Enhanced stats under executionStats
            - Backward-compatible response structure
            
        Example:
            >>> # Legacy endpoint response structure
            >>> {
            ...   "uptime": 3600,
            ...   "memory": {...},
            ...   "pyodide": {...},
            ...   "timestamp": "2025-01-XX...",
            ...   "executionStats": {
            ...     "overview": {...},
            ...     "topIPs": [...],
            ...     ...
            ...   }
            ... }
        """
        # Given: Execute code to generate statistics data
        test_code = sample_python_codes["simple_success"]
        execute_secure_python_code(test_code)
        
        # When: Legacy stats endpoint is accessed
        legacy_response = requests.get(
            f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['legacy_stats']}",
            timeout=SecurityTestConfig.TIMEOUTS["api_request"]
        )
        
        # Then: Legacy endpoint returns valid response
        assert legacy_response.status_code == 200, f"Legacy stats failed: {legacy_response.status_code}"
        legacy_data = legacy_response.json()
        
        # And: Legacy top-level fields are present
        legacy_required_fields = ["uptime", "memory", "pyodide", "timestamp"]
        for field in legacy_required_fields:
            assert field in legacy_data, f"Legacy response missing required field: {field}"
        
        # And: Enhanced statistics are available under executionStats
        assert "executionStats" in legacy_data, "Legacy response missing executionStats"
        execution_stats = legacy_data["executionStats"]
        
        # And: Enhanced stats contain all required sections
        enhanced_sections = ["overview", "recent", "topIPs", "topErrors", "userAgents", "hourlyTrend"]
        for section in enhanced_sections:
            assert section in execution_stats, f"Enhanced stats missing section: {section}"
        
        # And: Overview data is properly structured
        overview = execution_stats["overview"]
        overview_fields = ["totalExecutions", "successRate", "averageExecutionTime"]
        for field in overview_fields:
            assert field in overview, f"Overview missing field: {field}"

    @pytest.mark.security
    @pytest.mark.tracking
    def test_given_custom_user_agent_when_code_executed_then_ip_and_user_agent_tracking_works(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: IP address and User-Agent tracking functions correctly.
        
        Given: A custom User-Agent string for tracking identification
        When: Python code is executed with the custom User-Agent
        Then: IP address is properly tracked in statistics
        And: Custom User-Agent is recorded and retrievable
        And: Tracking data is accurate and accessible
        
        Description:
            This test validates the security tracking functionality for
            client identification, ensuring that IP addresses and User-Agent
            strings are properly captured and stored for security analysis.
            
        Input:
            - Python code execution request
            - Custom User-Agent header: "SecurityLoggingTestSuite/2.0.0"
            - Standard IP address (provided by test environment)
            
        Output:
            - IP tracking in topIPs statistics
            - User-Agent tracking in userAgents statistics
            - Accurate count and identification data
            
        Example:
            >>> # Expected tracking in statistics
            >>> stats = {
            ...   "topIPs": [{"ip": "127.0.0.1", "count": 1}],
            ...   "userAgents": [{"agent": "SecurityLoggingTestSuite/2.0.0", "count": 1}]
            ... }
        """
        # Given: Custom User-Agent for tracking
        custom_user_agent = SecurityTestConfig.TEST_USER_AGENT
        test_code = sample_python_codes["pathlib_demonstration"]
        
        # When: Code is executed with custom User-Agent
        result = execute_secure_python_code(test_code, user_agent=custom_user_agent)
        
        # Then: Execution succeeds
        assert result["success"] is True, f"Code execution failed: {result.get('error')}"
        
        # And: Statistics are updated with tracking data
        stats = get_security_statistics()
        
        # And: IP tracking is functional
        top_ips = stats.get("topIPs", [])
        assert len(top_ips) > 0, "IP tracking should contain entries"
        
        # Verify IP tracking structure
        for ip_entry in top_ips:
            assert "ip" in ip_entry, f"IP entry missing 'ip' field: {ip_entry}"
            assert "count" in ip_entry, f"IP entry missing 'count' field: {ip_entry}"
            assert isinstance(ip_entry["count"], int), f"Count must be integer: {type(ip_entry['count'])}"
            assert ip_entry["count"] > 0, f"Count must be positive: {ip_entry['count']}"
        
        # And: User-Agent tracking captures custom agent
        user_agents = stats.get("userAgents", [])
        assert len(user_agents) > 0, "User-Agent tracking should contain entries"
        
        # Verify User-Agent tracking includes our custom agent
        agent_strings = [ua.get("agent", "") for ua in user_agents]
        assert any(custom_user_agent in agent for agent in agent_strings), \
            f"Custom User-Agent not found in tracking: {agent_strings}"
        
        # And: User-Agent tracking structure is correct
        for ua_entry in user_agents:
            assert "agent" in ua_entry, f"UA entry missing 'agent' field: {ua_entry}"
            assert "count" in ua_entry, f"UA entry missing 'count' field: {ua_entry}"
            assert isinstance(ua_entry["count"], int), f"Count must be integer: {type(ua_entry['count'])}"

    @pytest.mark.security
    @pytest.mark.performance
    def test_given_various_code_complexity_when_executed_then_execution_times_are_tracked(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Execution times are properly tracked across different code complexities.
        
        Given: Python code snippets of varying computational complexity
        When: Each code snippet is executed sequentially
        Then: Execution times are measured and recorded
        And: Average execution time is calculated correctly
        And: Individual execution times are positive and reasonable
        
        Description:
            This test validates the performance tracking capabilities of the
            security logging system, ensuring that execution times are
            accurately measured and aggregated for analysis.
            
        Input:
            - Simple code: Basic print statement
            - Medium code: Loop with calculations
            - Complex code: Data processing with file operations
            
        Output:
            - Individual execution times in response data
            - Average execution time in statistics
            - Performance metrics tracking
            
        Example:
            >>> # Expected execution time tracking
            >>> response = {
            ...   "data": {"executionTime": 150},  # milliseconds
            ...   "stats": {"averageExecutionTime": 175.5}
            ... }
        """
        # Given: Code snippets of varying complexity
        complexity_tests = [
            ("simple_success", "Simple execution"),
            ("medium_complexity", "Medium complexity execution"),
            ("complex_execution", "Complex data processing")
        ]
        
        execution_times = []
        
        # When: Each code snippet is executed
        for code_key, description in complexity_tests:
            test_code = sample_python_codes[code_key]
            result = execute_secure_python_code(test_code)
            
            # Then: Execution succeeds with timing data
            assert result["success"] is True, f"{description} should succeed"
            
            # And: Execution time is properly recorded
            execution_time = result["data"]["executionTime"]
            assert isinstance(execution_time, (int, float)), \
                f"Execution time must be numeric: {type(execution_time)}"
            assert execution_time > 0, f"Execution time must be positive: {execution_time}"
            
            execution_times.append(execution_time)
        
        # And: Statistics include average execution time tracking
        stats = get_security_statistics()
        overview = stats["overview"]
        
        avg_time = overview.get("averageExecutionTime")
        assert avg_time is not None, "Average execution time should be tracked"
        assert isinstance(avg_time, (int, float)), f"Average time must be numeric: {type(avg_time)}"
        assert avg_time > 0, f"Average execution time must be positive: {avg_time}"
        
        # And: Average is reasonable compared to individual times
        calculated_avg = sum(execution_times) / len(execution_times)
        # Allow some variance due to other concurrent executions
        assert abs(avg_time - calculated_avg) <= calculated_avg * 2, \
            f"Average time should be reasonable: {avg_time} vs calculated {calculated_avg}"

    @pytest.mark.security
    @pytest.mark.api
    def test_given_statistics_with_data_when_reset_requested_then_statistics_are_cleared(
        self, server_ready, security_test_cleanup, sample_python_codes
    ):
        """
        Test Scenario: Statistics reset functionality clears all tracked data.
        
        Given: An active system with accumulated statistics data
        When: Statistics reset is requested via clear endpoint
        Then: All execution counts are reset to zero
        And: Error tracking is cleared
        And: IP and User-Agent tracking is reset
        And: Hourly trends are cleared
        
        Description:
            This test validates the statistics reset functionality, ensuring
            that administrators can clear accumulated data for privacy,
            testing, or maintenance purposes while maintaining system integrity.
            
        Input:
            - Accumulated statistics from previous executions
            - POST request to /api/dashboard/stats/clear
            
        Output:
            - All counters reset to zero
            - Empty tracking arrays
            - Successful reset confirmation
            
        Example:
            >>> # Statistics after reset
            >>> stats = {
            ...   "overview": {"totalExecutions": 0, "successRate": 0},
            ...   "topIPs": [],
            ...   "topErrors": [],
            ...   "userAgents": []
            ... }
        """
        # Given: Generate some statistics data first
        test_code = sample_python_codes["simple_success"]
        execute_secure_python_code(test_code)
        
        # And: Verify statistics exist before reset
        stats_before = get_security_statistics()
        overview_before = stats_before["overview"]
        assert overview_before["totalExecutions"] > 0, "Should have executions before reset"
        
        # When: Statistics reset is requested
        reset_response = requests.post(
            f"{Config.BASE_URL}{SecurityTestConfig.ENDPOINTS['dashboard_clear']}",
            timeout=SecurityTestConfig.TIMEOUTS["api_request"]
        )
        
        # Then: Reset operation succeeds
        assert reset_response.status_code == 200, f"Reset request failed: {reset_response.status_code}"
        reset_data = reset_response.json()
        assert reset_data["success"] is True, f"Reset operation failed: {reset_data.get('error')}"
        
        # Allow time for reset to take effect
        time.sleep(SecurityTestConfig.STATISTICS_RESET_WAIT_TIME)
        
        # And: Statistics are properly cleared
        stats_after = get_security_statistics()
        overview_after = stats_after["overview"]
        
        # And: All counters are reset
        assert overview_after["totalExecutions"] == 0, \
            f"Total executions should be 0 after reset: {overview_after['totalExecutions']}"
        assert overview_after["successRate"] == 0, \
            f"Success rate should be 0 after reset: {overview_after['successRate']}"
        
        # And: Tracking arrays are cleared
        assert len(stats_after.get("topIPs", [])) == 0, "IP tracking should be cleared"
        assert len(stats_after.get("topErrors", [])) == 0, "Error tracking should be cleared"
        assert len(stats_after.get("userAgents", [])) == 0, "User-Agent tracking should be cleared"
        
        # And: Hourly trends are reset
        hourly_trend = stats_after.get("hourlyTrend", [])
        assert len(hourly_trend) == 24, "Hourly trend should maintain 24-hour structure"
        assert all(count == 0 for count in hourly_trend), "All hourly counts should be 0"

    @pytest.mark.security
    @pytest.mark.integration
    @pytest.mark.slow
    def test_given_pathlib_operations_when_executed_then_cross_platform_compatibility_is_maintained(
        self, server_ready, security_test_cleanup
    ):
        """
        Test Scenario: Cross-platform pathlib operations function correctly.
        
        Given: Python code using pathlib for file system operations
        When: Code is executed on the server platform
        Then: All path operations succeed across Windows/Linux
        And: File creation and manipulation work correctly
        And: Path separator handling is automatic
        And: Security logging captures pathlib usage
        
        Description:
            This test validates cross-platform compatibility of Python code
            execution, specifically focusing on pathlib usage for portable
            file system operations across different operating systems.
            
        Input:
            - Python code with pathlib operations
            - File creation and manipulation commands
            - Cross-platform path handling
            
        Output:
            - Successful code execution
            - Proper file system operations
            - Platform-independent behavior
            
        Example:
            >>> # Cross-platform pathlib code
            >>> code = '''
            ... from pathlib import Path
            ... test_dir = Path('/tmp/cross_platform_test')
            ... test_dir.mkdir(parents=True, exist_ok=True)
            ... test_file = test_dir / 'test.txt'
            ... test_file.write_text('Cross-platform test')
            ... print(f'Created: {test_file}')
            ... '''
        """
        # Given: Cross-platform pathlib code for testing
        pathlib_test_code = """
from pathlib import Path
import json

# Cross-platform directory operations
base_dir = Path('/tmp/security_pathlib_test')
base_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories
plots_dir = base_dir / 'plots'
data_dir = base_dir / 'data'
plots_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

# File operations
test_file = data_dir / 'cross_platform_test.txt'
test_file.write_text('Cross-platform pathlib test successful')

# JSON data handling
config_file = base_dir / 'config.json'
config_data = {
    'platform_test': True,
    'base_path': str(base_dir),
    'file_count': len(list(base_dir.rglob('*')))
}
config_file.write_text(json.dumps(config_data, indent=2))

# Verification
print(f'Base directory: {base_dir}')
print(f'Directory exists: {base_dir.exists()}')
print(f'Test file exists: {test_file.exists()}')
print(f'Config file size: {config_file.stat().st_size} bytes')
print(f'Total files created: {len(list(base_dir.rglob("*")))}')
        """.strip()
        
        # When: Pathlib code is executed
        result = execute_secure_python_code(pathlib_test_code)
        
        # Then: Execution succeeds
        assert result["success"] is True, f"Pathlib code execution failed: {result.get('error')}"
        
        # And: Output indicates successful operations
        stdout = result["data"]["stdout"]
        assert "Base directory:" in stdout, "Base directory creation not reported"
        assert "Directory exists: True" in stdout, "Directory existence check failed"
        assert "Test file exists: True" in stdout, "File creation failed"
        assert "Config file size:" in stdout, "JSON file creation failed"
        assert "Total files created:" in stdout, "File enumeration failed"
        
        # And: No errors occurred during execution
        stderr = result["data"]["stderr"]
        assert stderr.strip() == "", f"Unexpected errors during pathlib operations: {stderr}"
        
        # And: Execution time is reasonable for file operations
        execution_time = result["data"]["executionTime"]
        assert execution_time > 0, "Execution time should be positive"
        # File operations should complete within reasonable time
        assert execution_time < 5000, f"Pathlib operations took too long: {execution_time}ms"
        
        # And: Security logging captures the pathlib execution
        stats = get_security_statistics()
        overview = stats["overview"]
        assert overview["totalExecutions"] >= 1, "Pathlib execution should be logged"
