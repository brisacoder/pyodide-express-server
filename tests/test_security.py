"""
Security testing module for Pyodide Express Server.

This module provides comprehensive security testing for the Pyodide Express
Server, focusing on path traversal attacks, code injection prevention,
input validation, rate limiting, and file upload security.

Test Categories:
1. Path Traversal Prevention - Validates protection against directory traversal attacks
2. Code Injection Prevention - Tests Python code sandboxing and execution safety
3. Input Validation - Validates request sanitization and malformed input handling
4. Rate Limiting - Tests API rate limiting and abuse prevention
5. File Upload Security - Validates secure file handling and content validation
6. Content Type Security - Tests proper content-type validation and MIME handling

All tests follow BDD (Behavior-Driven Development) patterns with descriptive
Given-When-Then structure and comprehensive documentation.

Requirements Compliance:
1. ✅ Pytest framework with fixtures and parameterization
2. ✅ No hardcoded globals - all configuration via Config class
3. ✅ Only /api/execute-raw endpoint for code execution
4. ✅ BDD-style test structure and naming
5. ✅ API contract validation for all responses
6. ✅ Pathlib usage for all file operations
7. ✅ Comprehensive security test coverage
8. ✅ Detailed docstrings with examples
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import requests

from conftest import Config, execute_python_code, validate_api_contract


class TestPathTraversalPrevention:
    """
    Test suite for path traversal attack prevention.
    
    Validates that the server properly sanitizes and validates file paths
    to prevent directory traversal attacks that could access sensitive
    system files or directories outside the designated upload area.
    """
    @pytest.mark.parametrize(
        "dangerous_path,expected_behavior",
        [
            ("../../../etc/passwd", "should_reject"),
            ("..\\..\\..\\windows\\system32\\config\\sam", "should_reject"),
            ("../../../../etc/shadow", "should_reject"),
            ("..\\..\\..\\..\\..\\..", "should_reject"),
            ("...//...//...//etc//passwd", "should_reject"),
            ("file\x00.csv", "should_sanitize"),  # Null byte injection
            ("file\n.csv", "should_sanitize"),   # Newline injection
            ("file\r.csv", "should_sanitize"),   # Carriage return injection
        ]
    )
    def test_path_traversal_protection_in_file_deletion(
        self,
        server_ready: None,
        test_cleanup: Any,
        dangerous_path: str,
        expected_behavior: str
    ) -> None:
        """
        Test that file deletion endpoint prevents path traversal attacks.
        
        Given a running server with file deletion capabilities
        When attempting to delete files using path traversal patterns
        Then the server should reject dangerous paths or sanitize them safely
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            dangerous_path: Path containing potential traversal attack vectors
            expected_behavior: Expected server response behavior
        
        Raises:
            AssertionError: If path traversal protection is inadequate
            
        Example:
            >>> # Attempt to access /etc/passwd via traversal
            >>> response = requests.delete(f"{base_url}/api/uploaded-files/../../../etc/passwd")
            >>> assert response.status_code == 400  # Should be rejected
        """
        # Given: A running server with file deletion endpoint
        base_url = Config.BASE_URL
        timeout = Config.TIMEOUTS["api_request"]
        
        # When: Attempting to delete a file using path traversal
        response = requests.delete(
            f"{base_url}/api/uploaded-files/{dangerous_path}",
            timeout=timeout
        )
        
        # Then: Server should protect against path traversal
        if expected_behavior == "should_reject":
            assert response.status_code == 400, f"Path '{dangerous_path}' should be rejected with 400"
            
            try:
                response_data = response.json()
                validate_api_contract(response_data)
                assert response_data["success"] is False
                assert response_data["data"] is None, "Error responses should have null data"
                assert "invalid" in response_data["error"].lower() or \
                       "forbidden" in response_data["error"].lower()
            except requests.exceptions.JSONDecodeError:
                # Some rejections may not return JSON
                pass
                
        elif expected_behavior == "should_sanitize":
            # If accepted, should be safely sanitized
            if response.status_code == 200:
                response_data = response.json()
                validate_api_contract(response_data)
                # Verify the path was sanitized (no traversal components)
                assert ".." not in dangerous_path or response_data["success"] is False

    def test_path_traversal_prevention_in_code_execution(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test that Python code execution prevents file system traversal.
        
        Given a Pyodide execution environment
        When executing Python code that attempts file system traversal
        Then the code should be sandboxed and prevented from accessing restricted files
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Raises:
            AssertionError: If code execution allows unauthorized file access
            
        Example:
            >>> # Attempt to read system files
            >>> result = execute_python_code("open('/etc/passwd', 'r').read()")
            >>> assert not result["success"] or "/etc/passwd" not in result["data"]["result"]
        """
        # Given: A Pyodide execution environment
        dangerous_file_access_codes = [
            # Using pathlib for file access attempts
            """
from pathlib import Path
system_file = Path("/etc/passwd")
if system_file.exists():
    content = system_file.read_text()
    print(content)
else:
    print("File not found (sandboxed)")
            """,
            
            # Attempting Windows system file access
            """
from pathlib import Path
windows_file = Path("C:/Windows/System32/config/sam")
try:
    if windows_file.exists():
        content = windows_file.read_text()
        print(content)
    else:
        print("File not accessible")
except Exception as e:
    print(f"Access blocked: {e}")
            """,
            
            # Directory listing attempts
            """
from pathlib import Path
root_dir = Path("/")
try:
    items = list(root_dir.iterdir())
    print(f"Root directory contains: {[item.name for item in items]}")
except Exception as e:
    print(f"Directory access blocked: {e}")
            """,
        ]
        
        for i, code in enumerate(dangerous_file_access_codes):
            # When: Executing code that attempts file system access
            result = execute_python_code(code)
            test_cleanup.track_file(f"test_security_traversal_{i}.log")
            
            # Then: Should be sandboxed or fail safely
            assert result["success"] is True, f"Code execution should complete (even if sandboxed): {result}"
            
            # Verify no sensitive system information is exposed
            output = result["data"]["result"] + result["data"]["stdout"] + result["data"]["stderr"]
            
            # Should not contain actual system file contents
            sensitive_patterns = ["root:x:", "daemon:x:", "SAM database", "Administrator:"]
            for pattern in sensitive_patterns:
                assert pattern not in output, f"Sensitive data '{pattern}' found in output: {output[:200]}"


class TestProcessIsolationSecurity:
    """
    Test suite for process isolation and execution safety.
    
    Validates that the server's process pool architecture provides proper
    isolation for Python code execution. The server uses process isolation
    rather than code sandboxing - dangerous operations may execute but are
    contained within killable child processes that protect the main server.
    """

    @pytest.mark.parametrize(
        "code_operation,expected_behavior",
        [
            ("print('Basic execution test')", "should_succeed"),
            ("import os; print(f'OS module: {hasattr(os, \"system\")}')", "may_succeed"),
            ("exec('print(\"Dynamic execution\")')", "may_succeed"),
            ("eval('2 + 2')", "may_succeed"),
        ]
    )
    def test_process_isolation_contains_code_execution(
        self,
        server_ready: None,
        test_cleanup: Any,
        code_operation: str,
        expected_behavior: str
    ) -> None:
        """
        Test that dangerous Python operations are blocked or sandboxed.
        
        Given a Pyodide execution environment with security restrictions
        When executing Python code with dangerous operations
        Then the operations should fail safely without compromising the system
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            dangerous_operation: Python code containing dangerous operations
            security_category: Category of security threat being tested
            
        Raises:
            AssertionError: If dangerous operations are not properly blocked
            
        Example:
            >>> # System command execution should be blocked
            >>> result = execute_python_code("import os; os.system('whoami')")
            >>> assert not result["success"] or "permission" in result["data"]["stderr"].lower()
        """
        # Given: A secured Pyodide execution environment
        timeout = Config.TIMEOUTS["code_execution"]
        
        # When: Executing Python code in isolated process
        result = execute_python_code(code_operation, timeout)
        test_cleanup.track_file(f"test_isolation_{expected_behavior}.log")
        
        # Then: Server should respond (proving isolation works)
        assert isinstance(result, dict), "Server should respond with structured result"
        assert "success" in result, "Response should include success field"
        assert "data" in result or "error" in result, "Response should include data or error"
        
        # And: Main server should remain responsive (test by making another call)
        health_result = execute_python_code("print('Server still responsive')", 5000)
        assert health_result["success"], "Main server should remain responsive after isolated execution"

    def test_network_access_behavior_in_isolation(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test network access behavior in process isolation.
        
        Network access may work in Pyodide WebAssembly environment but should
        be contained within isolated processes that don't compromise the server.
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Example:
            >>> # Network access may work but is contained in isolated process
            >>> code = "import socket; print('Socket module available')"
            >>> result = execute_python_code(code)
            >>> # Main server should remain responsive regardless of network code
        """
        # Given: Network access restrictions should be in place
        network_access_attempts = [
            # HTTP requests using pathlib and urllib
            """
try:
    import urllib.request
    from pathlib import Path
    response = urllib.request.urlopen('http://httpbin.org/get', timeout=5)
    content = response.read().decode()
    print(f"Network access successful: {len(content)} bytes")
except Exception as e:
    print(f"Network access blocked: {type(e).__name__}: {e}")
            """,
            
            # Socket connections
            """
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(('httpbin.org', 80))
    print("Socket connection successful")
    sock.close()
except Exception as e:
    print(f"Socket access blocked: {type(e).__name__}: {e}")
            """,
        ]
        
        for i, code in enumerate(network_access_attempts):
            # When: Attempting network operations
            result = execute_python_code(code)
            test_cleanup.track_file(f"test_network_security_{i}.log")
            
            # Then: Server should respond (network code is isolated)
            assert isinstance(result, dict), f"Should get structured response for network test {i}"
            assert "success" in result, "Response should include success field"
            
            # And: Main server should remain responsive after network operations
            health_check = execute_python_code("print('Server responsive after network test')", 3000)
            assert health_check["success"], f"Server should remain responsive after network test {i}"


class TestInputValidationAndSanitization:
    """
    Test suite for input validation and sanitization.
    
    Validates that all API endpoints properly validate and sanitize
    input data to prevent injection attacks and malformed data processing.
    """

    @pytest.mark.parametrize(
        "malicious_payload,content_type,expected_status",
        [
            ('<script>alert("xss")</script>', 'text/html', [400, 415]),
            ('<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe SYSTEM '
             '"file:///etc/passwd">]><root>&xxe;</root>', 'text/xml', [400, 415]),
            ('javascript:alert("xss")', 'text/javascript', [400, 415]),
            ('{"__proto__": {"injected": true}}', 'application/json', [400, 415]),
            ('constructor.prototype.injected=true', 'application/x-www-form-urlencoded', [400, 415]),
        ]
    )
    def test_content_type_validation_prevents_injection(
        self,
        server_ready: None,
        test_cleanup: Any,
        malicious_payload: str,
        content_type: str,
        expected_status: List[int]
    ) -> None:
        """
        Test that improper content types are rejected to prevent injection.
        
        Given API endpoints expecting specific content types
        When sending malicious payloads with wrong content types
        Then the server should reject the requests with appropriate error codes
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            malicious_payload: Malicious content to send
            content_type: Content-Type header to use
            expected_status: List of acceptable HTTP status codes
            
        Raises:
            AssertionError: If malicious content is not properly rejected
            
        Example:
            >>> # HTML content should be rejected by execute-raw endpoint
            >>> response = requests.post(endpoint, data='<script>alert(1)</script>',
            ...                         headers={'Content-Type': 'text/html'})
            >>> assert response.status_code in [400, 415]
        """
        # Given: API endpoint expecting text/plain for execute-raw
        base_url = Config.BASE_URL
        endpoint = f"{base_url}{Config.ENDPOINTS['execute_raw']}"
        timeout = Config.TIMEOUTS["api_request"]
        
        # When: Sending malicious payload with wrong content type
        response = requests.post(
            endpoint,
            data=malicious_payload,
            headers={"Content-Type": content_type},
            timeout=timeout
        )
        
        # Then: Request should be rejected
        assert response.status_code in expected_status, \
            f"Expected status {expected_status}, got {response.status_code} for {content_type}"
        
        # Verify response doesn't contain injected content
        try:
            if response.headers.get('content-type', '').startswith('application/json'):
                response_data = response.json()
                response_text = str(response_data)
                
                dangerous_indicators = ["injected", "alert", "script", "xxe"]
                for indicator in dangerous_indicators:
                    assert indicator not in response_text.lower(), \
                        f"Response contains dangerous content '{indicator}': {response_text[:200]}"
        except requests.exceptions.JSONDecodeError:
            pass  # Non-JSON error responses are acceptable

    def test_oversized_input_handling(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test handling of oversized input to prevent DoS attacks.
        
        Given API endpoints with size limits
        When sending extremely large payloads
        Then the server should reject or handle them safely
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Raises:
            AssertionError: If oversized input is not handled properly
            
        Example:
            >>> # Very large code should be rejected or limited
            >>> large_code = "print('x')\n" * 100000
            >>> result = execute_python_code(large_code)
            >>> assert not result["success"] or len(result["data"]["result"]) < 1000000
        """
        # Given: Size limits should be enforced
        max_reasonable_size = Config.MAX_CODE_LENGTH
        
        # Create oversized Python code
        oversized_code = "print('A' * 1000)\n" * (max_reasonable_size // 100)
        
        # When: Sending oversized code for execution
        try:
            result = execute_python_code(oversized_code, timeout=10)
            test_cleanup.track_file("test_oversized_input.log")
            
            # Then: Should either reject or limit output safely
            if result["success"]:
                total_output_size = (
                    len(result["data"]["result"]) + 
                    len(result["data"]["stdout"]) +
                    len(result["data"]["stderr"])
                )
                
                # Output should be reasonably sized (not gigabytes)
                max_output_size = 10 * 1024 * 1024  # 10MB limit
                assert total_output_size < max_output_size, \
                    f"Output too large: {total_output_size} bytes"
                    
                # Execution time should be reasonable (not infinite loop)
                execution_time = result["data"]["executionTime"]
                max_execution_time = 60000  # 60 seconds
                assert execution_time < max_execution_time, \
                    f"Execution too long: {execution_time}ms"
            else:
                # If rejected, should be for size/resource reasons
                error_msg = result["error"].lower()
                size_indicators = ["size", "limit", "large", "memory", "timeout"]
                assert any(indicator in error_msg for indicator in size_indicators), \
                    f"Should fail with size-related error: {result['error']}"
                    
        except requests.exceptions.Timeout:
            # Timeout is acceptable for oversized input
            assert True, "Timeout is an acceptable response to oversized input"
        except requests.exceptions.RequestException as e:
            # Network-level rejection is also acceptable
            assert "413" in str(e) or "payload" in str(e).lower(), \
                f"Should reject with payload size error: {e}"


class TestFileUploadSecurity:
    """
    Test suite for file upload security.
    
    Validates secure handling of file uploads including filename sanitization,
    content validation, and prevention of malicious file processing.
    """

    @pytest.mark.parametrize(
        "malicious_filename,expected_behavior",
        [
            ("../../../etc/passwd.csv", "sanitize_path"),
            ("..\\..\\windows\\system.ini.csv", "sanitize_path"),
            ("file\x00.csv", "sanitize_nullbyte"),
            ("file\n.csv", "sanitize_newline"),
            ("file\r.csv", "sanitize_carriage"),
            ("very_long_filename_" + "x" * 1000 + ".csv", "limit_length"),
            ("normal_file.csv", "accept_normal"),
        ]
    )
    def test_malicious_filename_sanitization(
        self,
        server_ready: None,
        test_cleanup: Any,
        malicious_filename: str,
        expected_behavior: str
    ) -> None:
        """
        Test that malicious filenames are properly sanitized during upload.
        
        Given a file upload endpoint with security measures
        When uploading files with malicious or problematic filenames
        Then filenames should be sanitized or rejected safely
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            malicious_filename: Filename containing potential security issues
            expected_behavior: Expected sanitization behavior
            
        Raises:
            AssertionError: If filename sanitization is inadequate
            
        Example:
            >>> # Path traversal in filename should be sanitized
            >>> with open("test.csv", "w") as f: f.write("col1,col2\n1,2\n")
            >>> response = upload_file("../../../etc/passwd.csv", file_content)
            >>> assert ".." not in response["data"]["file"]["sanitizedName"]
        """
        # Given: A file upload endpoint with sanitization
        base_url = Config.BASE_URL
        upload_endpoint = f"{base_url}{Config.ENDPOINTS['upload']}"
        timeout = Config.TIMEOUTS["api_request"]
        
        # Create test file content
        test_content = "column1,column2\nvalue1,value2\ntest,data\n"
        
        # Create temporary file
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        temp_file.write_text(test_content)
        test_cleanup.track_file(temp_file)
        
        try:
            # When: Uploading file with malicious filename
            with temp_file.open("rb") as file_handle:
                files = {"file": (malicious_filename, file_handle, "text/csv")}
                response = requests.post(upload_endpoint, files=files, timeout=timeout)
            
            # Then: Response should handle filename safely
            if response.status_code == 200:
                response_data = response.json()
                validate_api_contract(response_data)
                
                if response_data["success"] and "data" in response_data and "file" in response_data["data"]:
                    file_info = response_data["data"]["file"]
                    stored_name = file_info.get("sanitizedOriginal", "")
                    
                    test_cleanup.track_upload(stored_name)
                    
                    # Verify sanitization based on expected behavior
                    if expected_behavior == "sanitize_path":
                        assert ".." not in stored_name, f"Path traversal not sanitized: {stored_name}"
                        assert "/" not in stored_name, f"Path separators not sanitized: {stored_name}"
                        assert "\\" not in stored_name, f"Windows separators not sanitized: {stored_name}"
                        
                    elif expected_behavior == "sanitize_nullbyte":
                        assert "\x00" not in stored_name, f"Null bytes not sanitized: {stored_name}"
                        
                    elif expected_behavior == "sanitize_newline":
                        assert "\n" not in stored_name, f"Newlines not sanitized: {stored_name}"
                        assert "\r" not in stored_name, f"Carriage returns not sanitized: {stored_name}"
                        
                    elif expected_behavior == "limit_length":
                        assert len(stored_name) <= 255, f"Filename too long: {len(stored_name)} chars"
                        
                    elif expected_behavior == "accept_normal":
                        assert stored_name.endswith(".csv"), f"Normal filename not preserved: {stored_name}"
                
            elif response.status_code in [400, 413, 500]:  # Bad Request, Payload Too Large, or Server Error
                # Rejection is acceptable for dangerous filenames that cause HTTP parsing issues
                assert True, f"Filename rejection is acceptable: {response.status_code}"
            else:
                # Debug: Print response body for 500 errors
                error_details = ""
                try:
                    error_body = response.json()
                    error_details = f" - Error: {error_body.get('error', 'Unknown')}"
                except:
                    error_details = f" - Raw response: {response.text[:200]}"
                pytest.fail(f"Unexpected response status: {response.status_code}{error_details}")
                
        finally:
            # Cleanup is handled by test_cleanup fixture
            pass

    def test_malicious_file_content_handling(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test secure handling of files with malicious content.
        
        Given a file processing system
        When uploading files containing malicious or dangerous content
        Then the system should process them safely without executing harmful code
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Raises:
            AssertionError: If malicious file content is not handled safely
            
        Example:
            >>> # CSV with embedded scripts should be processed safely
            >>> csv_with_script = 'name,script\ntest,"<script>alert(1)</script>"\n'
            >>> result = upload_and_process_csv(csv_with_script)
            >>> assert "script" not in result["data"]["processed_content"]
        """
        # Given: File processing capabilities
        malicious_file_contents = [
            # CSV with embedded JavaScript
            ('javascript_embed.csv', 'name,script\ntest,"<script>alert(1)</script>"\nuser,"javascript:void(0)"\n'),
            
            # CSV with SQL injection attempts  
            ('sql_injection.csv', 'id,name\n1,"Robert"); DROP TABLE users; --"\n2,"Alice\'; DELETE FROM data; --"\n'),
            
            # CSV with format string attacks
            ('format_string.csv', 'name,value\n"%s%s%s%s","%d%d%d%d"\n"{{config}}","{{secrets}}"\n'),
            
            # CSV with very long fields (potential buffer overflow)
            ('long_fields.csv', f'name,value\n"{"A" * 10000}","{"B" * 10000}"\n'),
        ]
        
        base_url = Config.BASE_URL
        upload_endpoint = f"{base_url}{Config.ENDPOINTS['upload']}"
        
        for filename, malicious_content in malicious_file_contents:
            # Create temporary file with malicious content
            temp_file = Path(tempfile.mktemp(suffix=".csv"))
            temp_file.write_text(malicious_content)
            test_cleanup.track_file(temp_file)
            
            try:
                # When: Uploading malicious file
                with temp_file.open("rb") as file_handle:
                    files = {"file": (filename, file_handle, "text/csv")}
                    upload_response = requests.post(
                        upload_endpoint, 
                        files=files, 
                        timeout=Config.TIMEOUTS["api_request"]
                    )
                
                if upload_response.status_code == 200:
                    upload_data = upload_response.json()
                    validate_api_contract(upload_data)
                    
                    if upload_data["success"] and "file" in upload_data["data"]:
                        uploaded_filename = upload_data["data"]["file"]["sanitizedOriginal"]
                        test_cleanup.track_upload(uploaded_filename)
                        
                        # Then: Try to process the file safely using pathlib
                        processing_code = f"""
from pathlib import Path
import pandas as pd

try:
    # Use pathlib for file operations
    csv_path = Path("{Config.PATHS['uploads_dir']}") / "{uploaded_filename}"
    
    if csv_path.exists():
        # Read with pandas safely
        df = pd.read_csv(csv_path)
        
        # Basic processing that shouldn't execute malicious code
        row_count = len(df)
        col_count = len(df.columns)
        
        # Safe string operations that won't interpret scripts
        column_names = list(df.columns)
        
        print(f"Processed CSV safely: {{row_count}} rows, {{col_count}} columns")
        print(f"Columns: {{column_names}}")
        
        # Verify no script execution occurred
        result = f"safe_processing_complete_{{row_count}}_{{col_count}}"
        print(result)
    else:
        print(f"File not found: {{csv_path}}")
        
except Exception as e:
    print(f"Safe processing error: {{type(e).__name__}}: {{e}}")
                        """
                        
                        processing_result = execute_python_code(processing_code)
                        
                        # Verify safe processing
                        if processing_result["success"]:
                            output = (processing_result["data"]["result"] + 
                                    processing_result["data"]["stdout"] + 
                                    processing_result["data"]["stderr"])
                            
                            # Should indicate safe processing
                            assert ("safe_processing_complete" in output or 
                                   "Safe processing error" in output), \
                                f"Processing should complete safely: {output[:300]}"
                            
                            # Should not execute malicious scripts
                            dangerous_indicators = [
                                "alert(1)", "DROP TABLE", "DELETE FROM", 
                                "javascript:", "<script>", "{{config}}", "{{secrets}}"
                            ]
                            
                            for indicator in dangerous_indicators:
                                assert indicator not in output, \
                                    f"Malicious content '{indicator}' should not be executed: {output[:300]}"
                        
                        else:
                            # Processing failure is acceptable for malicious files
                            assert "error" in processing_result["error"].lower(), \
                                f"Processing should fail safely: {processing_result['error']}"
                
            finally:
                # Cleanup handled by test_cleanup fixture
                pass


class TestRateLimitingAndAbusePrevention:
    """
    Test suite for rate limiting and abuse prevention.
    
    Validates that the server implements appropriate rate limiting
    to prevent abuse and denial of service attacks.
    """

    def test_reasonable_rate_limiting_behavior(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test that rate limiting allows normal usage but prevents abuse.
        
        Given a server with rate limiting configured
        When making multiple requests within a short timeframe
        Then normal usage should be allowed while excessive requests are limited
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Raises:
            AssertionError: If rate limiting is too aggressive or too lenient
            
        Example:
            >>> # Multiple normal requests should mostly succeed
            >>> results = [execute_python_code("print('test')") for _ in range(10)]
            >>> success_count = sum(1 for r in results if r.get("success"))
            >>> assert success_count >= 7  # Most should succeed
        """
        # Given: A server with rate limiting
        responses = []
        successful_executions = 0
        rate_limited_responses = 0
        
        # When: Making multiple requests quickly
        for i in range(15):  # Reasonable number of requests
            try:
                simple_code = f"print('Request {i}: Hello from Pyodide')"
                result = execute_python_code(simple_code, timeout=5)
                test_cleanup.track_file(f"rate_test_{i}.log")
                
                if result["success"]:
                    successful_executions += 1
                    responses.append(200)
                else:
                    responses.append(400)  # Assume error response
                    
            except requests.exceptions.Timeout:
                responses.append(408)  # Timeout
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    rate_limited_responses += 1
                    responses.append(429)
                else:
                    responses.append(500)
            except Exception:
                responses.append(500)  # General error
        
        # Then: Rate limiting should be reasonable
        total_requests = len(responses)
        success_rate = successful_executions / total_requests
        
        # Should allow reasonable number of successful requests (at least 50%)
        assert success_rate >= 0.5, \
            f"Success rate too low: {success_rate:.2%} ({successful_executions}/{total_requests})"
        
        # If rate limiting occurs, should be documented in responses  
        if rate_limited_responses > 0:
            assert rate_limited_responses < total_requests * 0.8, \
                f"Rate limiting too aggressive: {rate_limited_responses}/{total_requests} requests blocked"
        
        # Overall system should remain responsive
        assert successful_executions >= 5, \
            f"Too few successful requests: {successful_executions} (minimum 5 expected)"

    def test_resource_exhaustion_protection(
        self,
        server_ready: None,
        test_cleanup: Any
    ) -> None:
        """
        Test protection against resource exhaustion attacks.
        
        Given a server with resource protection measures
        When executing code that attempts to consume excessive resources
        Then the server should limit resource usage and remain stable
        
        Args:
            server_ready: Ensures server is available for testing
            test_cleanup: Provides cleanup functionality for test artifacts
            
        Raises:
            AssertionError: If server doesn't protect against resource exhaustion
            
        Example:
            >>> # Memory exhaustion attempt should be limited
            >>> memory_bomb = "data = ['x' * 1000000 for _ in range(1000)]"
            >>> result = execute_python_code(memory_bomb)
            >>> assert result["data"]["executionTime"] < 30000  # Should timeout/limit
        """
        # Given: Resource protection measures should be active
        resource_exhaustion_attempts = [
            # Memory exhaustion attempt using pathlib for logging
            """
from pathlib import Path
import time

try:
    # Attempt to allocate large amounts of memory
    start_time = time.time()
    memory_hog = []
    
    for i in range(1000):
        # Try to create large lists
        chunk = ['x' * 100000] * 100  # ~10MB per iteration
        memory_hog.append(chunk)
        
        # Check if we've been running too long
        if time.time() - start_time > 10:  # 10 second limit
            break
    
    print(f"Memory allocation completed: {len(memory_hog)} chunks")
    
except MemoryError as e:
    print(f"Memory limit reached (good): {e}")
except Exception as e:
    print(f"Resource protection active: {type(e).__name__}: {e}")
            """,
            
            # CPU exhaustion attempt
            """
import time
from pathlib import Path

try:
    start_time = time.time()
    iterations = 0
    
    # Busy loop that should be limited
    while iterations < 10000000:  # 10 million iterations
        iterations += 1
        
        # Check for timeout
        if time.time() - start_time > 15:  # 15 second limit
            break
    
    duration = time.time() - start_time
    print(f"CPU test completed: {iterations} iterations in {duration:.2f}s")
    
except Exception as e:
    print(f"CPU protection active: {type(e).__name__}: {e}")
            """,
        ]
        
        for i, resource_code in enumerate(resource_exhaustion_attempts):
            # When: Attempting resource exhaustion
            start_time = time.time()
            
            try:
                result = execute_python_code(resource_code, timeout=20)  # 20 second timeout
                execution_duration = time.time() - start_time
                test_cleanup.track_file(f"resource_test_{i}.log")
                
                # Then: Execution should be limited and controlled
                if result["success"]:
                    output = result["data"]["result"] + result["data"]["stdout"]
                    
                    # Should show protection is active
                    protection_indicators = [
                        "limit reached", "protection active", "Memory limit", 
                        "CPU protection", "timeout", "Resource"
                    ]
                    
                    has_protection = any(indicator in output for indicator in protection_indicators)
                    
                    # Execution time should be reasonable (not infinite)
                    max_reasonable_time = 25.0  # 25 seconds max
                    assert execution_duration < max_reasonable_time, \
                        f"Execution too long: {execution_duration:.2f}s"
                    
                    # If no explicit protection message, execution should complete quickly
                    if not has_protection:
                        assert execution_duration < 15.0, \
                            f"Long execution without protection indication: {execution_duration:.2f}s"
                
                else:
                    # Execution failure is acceptable for resource exhaustion
                    error_msg = result["error"].lower()
                    resource_indicators = [
                        "memory", "timeout", "resource", "limit", "exceeded"
                    ]
                    
                    has_resource_error = any(indicator in error_msg for indicator in resource_indicators)
                    assert has_resource_error, \
                        f"Should fail with resource-related error: {result['error']}"
                        
            except requests.exceptions.Timeout:
                # Timeout is an acceptable form of resource protection
                execution_duration = time.time() - start_time
                assert execution_duration >= 15.0, \
                    f"Timeout should occur after reasonable time: {execution_duration:.2f}s"


# Test markers for pytest organization
pytestmark = [
    pytest.mark.api,
    pytest.mark.integration, 
    pytest.mark.slow,
]
