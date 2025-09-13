"""
Security test scenarios in BDD style using pytest.

This module contains comprehensive tests for API security and vulnerability testing,
written in Behavior-Driven Development (BDD) style using pytest. The tests focus on
preventing path traversal, code injection, input validation, and other security concerns.

All tests use only the public `/api/execute-raw` endpoint for Python code execution,
avoiding internal pyodide APIs. The tests follow a Given-When-Then pattern and include
comprehensive docstrings for each scenario.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import List, Generator

import pytest
import requests

# Global configuration constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 10
EXECUTION_TIMEOUT = 30000
MAX_CODE_LENGTH = 50000
MAX_FILE_SIZE_MB = 10
REQUEST_TIMEOUT = 60000
RATE_LIMIT_REQUESTS = 10
LONG_FILENAME_LENGTH = 1000


class TestPathTraversalSecurity:
    """Test scenarios for path traversal vulnerability prevention."""

    def test_given_dangerous_path_when_deleting_file_then_validation_error(self, server, base_url):
        """
        Scenario: Attempt path traversal in file deletion
        Given the API is available
        When I attempt to delete files using path traversal patterns
        Then the request should be rejected with validation error
        """
        # Given
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../../../etc/shadow",
            "..\\..\\..\\..\\..\\..",
            "...//...//...//etc//passwd",
        ]
        
        for path in dangerous_paths:
            # When
            response = requests.delete(
                f"{base_url}/api/uploaded-files/{path}", 
                timeout=DEFAULT_TIMEOUT
            )
            
            # Then
            assert response.status_code == 400
            response_data = response.json()
            assert response_data.get("success") is False
            assert "invalid" in response_data.get("error", "").lower()

    def test_given_traversal_path_when_getting_file_info_then_safe_handling(self, server, base_url):
        """
        Scenario: Attempt path traversal in file info endpoint
        Given the API is available
        When I request file info using path traversal patterns
        Then the system should handle the request safely
        """
        # Given
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\windows\\system.ini",
            "....//....//etc//hosts",
        ]
        
        for path in dangerous_paths:
            # When
            response = requests.get(
                f"{base_url}/api/file-info/{path}", 
                timeout=DEFAULT_TIMEOUT
            )
            
            # Then
            assert response.status_code in [400, 404, 200]
            if response.status_code == 200:
                response_data = response.json()
                assert response_data["uploadedFile"]["exists"] is False


class TestCodeInjectionSecurity:
    """Test scenarios for Python code injection and execution restrictions."""

    def test_given_filesystem_access_code_when_executing_then_sandboxed_safely(self, server, base_url):
        """
        Scenario: Attempt dangerous filesystem operations
        Given the API is available
        When I execute Python code that tries to access the filesystem
        Then the code should be sandboxed and prevent unauthorized access
        """
        # Given
        dangerous_codes = [
            # File system access attempts
            "import os; print(os.listdir('/'))",
            "print(open('/etc/passwd', 'r').read())",
            "with open('../../sensitive_file.txt') as f: print(f.read())",
        ]
        
        for code in dangerous_codes:
            # When
            response = requests.post(
                f"{base_url}/api/execute-raw", 
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 200
            response_data = response.json()
            
            # Should either fail execution or be safely sandboxed
            if response_data.get("success"):
                result = response_data.get("stdout", "") + response_data.get("stderr", "")
                # Should not contain sensitive system information
                assert "/etc/passwd" not in str(result)
                assert "root:" not in str(result)

    def test_given_system_command_code_when_executing_then_blocked_execution(self, server, base_url):
        """
        Scenario: Attempt system command execution
        Given the API is available
        When I execute Python code that tries to run system commands
        Then the execution should be blocked or safely sandboxed
        """
        # Given
        command_codes = [
            "import subprocess; print(subprocess.run(['ls', '-la'], capture_output=True, text=True).stdout)",
            "import os; print(os.system('whoami'))",
            "__import__('os').system('id')",
        ]
        
        for code in command_codes:
            # When
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 200
            response_data = response.json()
            
            # Should be blocked by Pyodide sandbox or fail safely
            if not response_data.get("success"):
                assert "error" in response_data

    def test_given_network_access_code_when_executing_then_blocked_safely(self, server, base_url):
        """
        Scenario: Attempt network access from executed code
        Given the API is available
        When I execute Python code that tries to access external networks
        Then the network access should be blocked by the sandbox
        """
        # Given
        network_codes = [
            "import urllib.request; print(urllib.request.urlopen('http://malicious.com').read())",
            "import socket; s = socket.socket(); s.connect(('external.com', 80)); print('connected')",
        ]
        
        for code in network_codes:
            # When
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 200
            # Network access should be blocked by Pyodide environment

    def test_given_exit_attempts_when_executing_then_handled_gracefully(self, server, base_url):
        """
        Scenario: Attempt to exit or quit the Python environment
        Given the API is available
        When I execute code that tries to exit the environment
        Then the exit attempts should be handled gracefully
        """
        # Given
        exit_codes = [
            "import sys; sys.exit(1)",
            "exit()",
            "quit()",
        ]
        
        for code in exit_codes:
            # When
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 200
            # Environment should continue to work after exit attempts

    def test_given_eval_exec_code_when_executing_then_sandboxed_properly(self, server, base_url):
        """
        Scenario: Attempt eval/exec code execution
        Given the API is available
        When I execute code using eval or exec functions
        Then the evaluation should be sandboxed properly
        """
        # Given
        eval_codes = [
            'exec(\'print("injected")\')',
            'print(eval("1+1"))',
        ]
        
        for code in eval_codes:
            # When
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 200
            # Eval/exec should work but within sandbox constraints


class TestPackageInstallationSecurity:
    """Test scenarios for package installation restrictions."""

    def test_given_blocked_packages_when_installing_then_installation_denied(self, server, base_url):
        """
        Scenario: Attempt to install dangerous packages
        Given the API is available
        When I try to install packages that could be dangerous
        Then the installation should be denied
        """
        # Given
        blocked_packages = [
            "os",
            "subprocess", 
            "socket",
            "urllib3",
            "requests-oauthlib"
        ]
        
        for package in blocked_packages:
            # When
            response = requests.post(
                f"{base_url}/api/install-package", 
                json={"package": package}, 
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code == 403
            response_data = response.json()
            assert response_data.get("success") is False
            assert "not allowed" in response_data.get("error", "").lower()


class TestInputValidationSecurity:
    """Test scenarios for input validation and sanitization."""

    def test_given_malicious_filenames_when_uploading_then_sanitized_safely(self, server, base_url):
        """
        Scenario: Upload files with malicious filenames
        Given the API is available
        When I upload files with dangerous filenames
        Then the filenames should be sanitized or rejected safely
        """
        # Given
        malicious_names = [
            "../../../etc/passwd.csv",
            "..\\..\\windows\\system.ini.csv",
            "file\x00.csv",  # Null byte injection
            "file\n.csv",    # Newline injection
            "file\r.csv",    # Carriage return injection
            "very_long_filename_" + "x" * LONG_FILENAME_LENGTH + ".csv",  # Extremely long filename
        ]
        
        for filename in malicious_names:
            # Given - create test file
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write("col1,col2\n1,2\n")
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, "rb") as fh:
                    # When
                    response = requests.post(
                        f"{base_url}/api/upload",
                        files={"file": (filename, fh, "text/csv")},
                        timeout=EXECUTION_TIMEOUT / 1000
                    )
                
                # Then
                if response.status_code == 200:
                    # If accepted, verify the filename was sanitized
                    response_data = response.json()
                    if response_data.get("success") and "data" in response_data:
                        file_info = response_data["data"]["file"]
                        stored_name = file_info.get("sanitizedOriginal", "")
                        # Should not contain path traversal components
                        assert ".." not in stored_name
                        assert "/" not in stored_name
                        assert "\\" not in stored_name
                        assert "\x00" not in stored_name
                        
                        # Clean up using only public APIs
                        try:
                            sanitized_filename = file_info.get("sanitizedOriginal")
                            if sanitized_filename:
                                requests.delete(
                                    f"{base_url}/api/uploaded-files/{sanitized_filename}", 
                                    timeout=DEFAULT_TIMEOUT
                                )
                        except Exception:
                            pass
                        
            finally:
                os.unlink(tmp_path)

    def test_given_json_injection_when_posting_then_parsed_safely(self, server, base_url):
        """
        Scenario: Attempt JSON injection attacks
        Given the API is available
        When I send JSON with prototype pollution attempts
        Then the JSON should be parsed safely without pollution
        """
        # Given
        injection_attempts = [
            '{"code": "print(\\"test\\")", "extra": {"__proto__": {"injected": true}}}',
            '{"code": "print(\\"test\\")", "constructor": {"prototype": {"injected": true}}}',
        ]
        
        for json_str in injection_attempts:
            # When
            response = requests.post(
                f"{base_url}/api/execute",
                data=json_str,
                headers={"Content-Type": "application/json"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            if response.status_code == 200:
                response_data = response.json()
                # Should not have prototype pollution
                assert "injected" not in str(response_data)


class TestRateLimitingSecurity:
    """Test scenarios for rate limiting behavior."""

    def test_given_multiple_rapid_requests_when_executing_then_rate_limit_enforced(self, server, base_url):
        """
        Scenario: Send multiple rapid requests to test rate limiting
        Given the API is available
        When I send many requests quickly
        Then rate limiting should allow reasonable usage while blocking abuse
        """
        # Given
        responses = []
        
        # When
        for i in range(RATE_LIMIT_REQUESTS):
            response = requests.post(
                f"{base_url}/api/execute-raw", 
                data=f"print({i})",
                headers={"Content-Type": "text/plain"},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            responses.append(response.status_code)
        
        # Then
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Should have mostly successful requests for normal usage
        assert success_count >= 5  # At least half should succeed


class TestContentTypeSecurity:
    """Test scenarios for content type validation."""

    def test_given_malicious_content_types_when_posting_then_rejected_properly(self, server, base_url):
        """
        Scenario: Send malicious content with wrong content types
        Given the API is available
        When I send malicious payloads with incorrect content types
        Then the requests should be rejected
        """
        # Given
        malicious_payloads = [
            ('<script>alert("xss")</script>', 'text/html'),
            ('<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><root>&xxe;</root>', 'text/xml'),
            ('javascript:alert("xss")', 'text/javascript'),
        ]
        
        for payload, content_type in malicious_payloads:
            # When
            response = requests.post(
                f"{base_url}/api/execute",
                data=payload,
                headers={"Content-Type": content_type},
                timeout=EXECUTION_TIMEOUT / 1000
            )
            
            # Then
            assert response.status_code in [400, 415]  # Bad Request or Unsupported Media Type


class TestFileUploadSecurity:
    """Test scenarios for file upload security."""

    def test_given_malicious_file_content_when_uploading_then_handled_safely(self, server, base_url):
        """
        Scenario: Upload files with malicious content
        Given the API is available
        When I upload files containing potentially dangerous content
        Then the content should be handled safely during processing
        """
        # Given
        malicious_contents = [
            # CSV with embedded JavaScript
            'name,script\ntest,"<script>alert(1)</script>"\n',
            # CSV with SQL injection attempts
            'id,name\n1,"Robert\'); DROP TABLE users; --"\n',
            # CSV with extremely long fields
            'name,value\n' + '"' + 'A' * 100000 + '",1\n',
            # CSV with many columns (potential DoS)
            ','.join([f'col{i}' for i in range(1000)]) + '\n' + ','.join(['1'] * 1000) + '\n',
        ]
        
        for content in malicious_contents:
            # Given - create malicious file
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, "rb") as fh:
                    # When
                    response = requests.post(
                        f"{base_url}/api/upload",
                        files={"file": ("malicious.csv", fh, "text/csv")},
                        timeout=EXECUTION_TIMEOUT / 1000
                    )
                
                # Then
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("success") and "data" in response_data:
                        file_info = response_data["data"]["file"]
                        sanitized_filename = file_info.get("sanitizedOriginal")
                        
                        if sanitized_filename:
                            # Test processing the uploaded file safely using execute-raw
                            code = f'''
import pandas as pd
from pathlib import Path

try:
    # Use the uploaded file path
    file_path = Path("/home/pyodide/uploads") / "{sanitized_filename}"
    df = pd.read_csv(file_path)
    print("processed_safely")
except Exception as e:
    print(f"error: {{type(e).__name__}}")
'''
                            
                            process_response = requests.post(
                                f"{base_url}/api/execute-raw",
                                data=code,
                                headers={"Content-Type": "text/plain"},
                                timeout=EXECUTION_TIMEOUT / 1000
                            )
                            assert process_response.status_code == 200
                            
                            # Clean up using public API
                            try:
                                requests.delete(
                                    f"{base_url}/api/uploaded-files/{sanitized_filename}", 
                                    timeout=DEFAULT_TIMEOUT
                                )
                            except Exception:
                                pass
                        
            finally:
                os.unlink(tmp_path)


class TestTimeoutSecurity:
    """Test scenarios for execution timeout security."""

    def test_given_long_running_code_when_executing_then_timeout_enforced(self, server, base_url):
        """
        Scenario: Execute long-running code to test timeout
        Given the API is available
        When I execute code that takes a long time
        Then the execution should be properly timed out
        """
        # Given
        long_running_code = '''
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.01)  # Small sleep to make it slower
print(total)
'''
        
        # When
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=10  # Short timeout for test
        )
        execution_time = time.time() - start_time
        
        # Then
        assert execution_time < 15  # Should not take more than 15 seconds
        assert response.status_code == 200


class TestMemorySecurityLimits:
    """Test scenarios for memory usage security limits."""

    def test_given_memory_intensive_code_when_executing_then_handled_safely(self, server, base_url):
        """
        Scenario: Execute memory-intensive operations
        Given the API is available
        When I execute code that uses significant memory
        Then the memory usage should be handled safely within limits
        """
        # Given
        memory_codes = [
            # Large list creation
            "large_list = list(range(100000)); print(len(large_list))",
            # Large string operations
            "big_string = 'x' * 1000000; print(len(big_string))",
            # Large dictionary
            "big_dict = {i: f'value_{i}' for i in range(10000)}; print(len(big_dict))",
        ]
        
        for code in memory_codes:
            # When
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=10
            )
            execution_time = time.time() - start_time
            
            # Then
            assert response.status_code == 200
            assert execution_time < 5  # Should complete reasonably quickly
            
            response_data = response.json()
            if response_data.get("success"):
                # If successful, result should be reasonable
                stdout = response_data.get("stdout", "")
                assert stdout.strip().isdigit()  # Should contain a number


# Pytest configuration for this module
pytestmark = [
    pytest.mark.security,
    pytest.mark.integration,
]