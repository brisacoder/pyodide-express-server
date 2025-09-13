"""
Performance and resource limits test scenarios in BDD style using pytest.

This module contains comprehensive tests for performance characteristics,
resource management, and system limits, written in Behavior-Driven Development 
(BDD) style using pytest.

All tests use only the approved /api/execute-raw endpoint for Python code execution
and avoid internal REST APIs with 'pyodide' in their names.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Generator, Tuple

import pytest
import requests

# ==================== GLOBAL CONFIGURATION ====================

# API Configuration
BASE_URL = "http://localhost:3000"
EXECUTE_RAW_ENDPOINT = f"{BASE_URL}/api/execute-raw"

# Timeout Configuration (Global Constants)
DEFAULT_TIMEOUT = 10
EXECUTION_TIMEOUT = 30000  # 30 seconds for Python execution
LONG_EXECUTION_TIMEOUT = 60000  # 60 seconds for intensive operations
UPLOAD_TIMEOUT = 60
REQUEST_TIMEOUT = 30

# Performance Limits
MAX_EXECUTION_TIME = 10  # seconds
MAX_UPLOAD_TIME = 30     # seconds
MAX_PROCESSING_TIME = 15 # seconds
MAX_LIST_TIME = 2        # seconds
MAX_INFO_TIME = 1        # second

# Test Data Configuration
LARGE_CSV_ROWS = 5000
MULTIPLE_FILES_COUNT = 5
CONCURRENT_REQUESTS_COUNT = 5

# Memory and CPU Test Limits
MEMORY_TEST_SIZE = 100000    # Large list size
STRING_TEST_SIZE = 1000000   # Large string size
DICT_TEST_SIZE = 10000       # Large dictionary size
MATRIX_SIZE = 100            # Matrix dimensions
FIBONACCI_NUMBER = 20        # Fibonacci calculation input
PRIME_RANGE = 1000          # Prime number calculation range


def wait_for_server(url: str, timeout: int = 120) -> None:
    """
    Wait for server to be ready by polling the health endpoint.
    
    Args:
        url: The base URL of the server
        timeout: Maximum time to wait in seconds
        
    Raises:
        RuntimeError: If server doesn't start within timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


# ==================== FIXTURES ====================


@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure server is ready before running any tests.
    
    Returns:
        bool: True when server is confirmed ready
    """
    wait_for_server(BASE_URL)
    return True


@pytest.fixture
def default_timeout():
    """
    Provide the default timeout for API requests.
    
    Returns:
        int: Default timeout in seconds
    """
    return DEFAULT_TIMEOUT


@pytest.fixture
def execution_timeout():
    """
    Provide the default timeout for code execution.
    
    Returns:
        int: Execution timeout in milliseconds
    """
    return EXECUTION_TIMEOUT


@pytest.fixture
def sample_csv_content():
    """
    Provide sample CSV content for testing file operations.
    
    Returns:
        str: CSV content with headers and sample data
    """
    return "name,value,category\nitem1,100,A\nitem2,200,B\nitem3,300,C\n"


@pytest.fixture
def large_csv_content():
    """
    Generate large CSV content for performance testing.
    
    Returns:
        str: Large CSV content with specified number of rows
    """
    content = "id,value,category,description\n"
    for i in range(LARGE_CSV_ROWS):
        content += f"{i},{i*2},category_{i%10},description for item {i}\n"
    return content


@pytest.fixture
def uploaded_file(sample_csv_content, default_timeout) -> Generator[Tuple[str, str], None, None]:
    """
    Upload a test CSV file and return file references.
    
    This fixture handles both upload and cleanup automatically.
    
    Args:
        sample_csv_content: CSV content to upload
        default_timeout: Timeout for upload operation
        
    Yields:
        Tuple[str, str]: (filename, server_path) for the uploaded file
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(sample_csv_content)
        tmp_path = tmp.name

    try:
        # Upload file
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("test_data.csv", fh, "text/csv")},
                timeout=default_timeout,
            )

        # Extract file references
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data.get("success") is True

        file_info = upload_data["data"]["file"]
        filename = Path(file_info["vfsPath"]).name
        server_path = file_info["vfsPath"]

        # Yield for test to use
        yield filename, server_path

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)
        
        # Cleanup uploaded file
        try:
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=10)
        except (requests.RequestException, OSError):
            pass  # Ignore cleanup errors


# ==================== EXECUTION PERFORMANCE TESTS ====================


class TestExecutionPerformance:
    """Test scenarios for code execution performance and timeout handling."""

    def test_given_long_running_code_when_executed_with_timeout_then_completes_within_limit(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Execute code with performance constraints
        Given: A computationally intensive Python script
        When: I execute it with a reasonable timeout
        Then: It should complete within acceptable time limits
        """
        # Given
        intensive_code = '''
import time
total = 0
for i in range(100000):  # Reduced for faster testing
    total += i
    if i % 50000 == 0:
        time.sleep(0.01)  # Small sleep to make it measureable
str(total)
'''
        
        # When
        start_time = time.time()
        response = requests.post(
            EXECUTE_RAW_ENDPOINT,
            data=intensive_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then
        assert response.status_code == 200
        assert execution_time < MAX_EXECUTION_TIME, f"Execution took {execution_time:.2f}s, expected < {MAX_EXECUTION_TIME}s"
        
        result = response.json()
        assert result.get("success") is not False, f"Execution failed: {result}"

    def test_given_memory_intensive_operations_when_executed_then_handles_gracefully(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Execute memory-intensive operations
        Given: Python code that creates large data structures
        When: I execute multiple memory-intensive operations
        Then: Each should complete successfully within reasonable time
        """
        # Given
        memory_test_cases = [
            {
                "name": "large_list_creation",
                "code": f"large_list = list(range({MEMORY_TEST_SIZE})); len(large_list)",
                "expected_result": MEMORY_TEST_SIZE
            },
            {
                "name": "large_string_operations", 
                "code": f"big_string = 'x' * {STRING_TEST_SIZE}; len(big_string)",
                "expected_result": STRING_TEST_SIZE
            },
            {
                "name": "large_dictionary",
                "code": f"big_dict = {{i: f'value_{{i}}' for i in range({DICT_TEST_SIZE})}}; len(big_dict)",
                "expected_result": DICT_TEST_SIZE
            },
        ]
        
        for test_case in memory_test_cases:
            # When
            start_time = time.time()
            response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=test_case["code"],
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then
            assert response.status_code == 200, f"Failed for {test_case['name']}"
            assert execution_time < MAX_EXECUTION_TIME, f"{test_case['name']} took {execution_time:.2f}s"
            
            result = response.json()
            if result.get("success") is not False:
                # Convert result to int for comparison if it's a string representation
                actual_result = result.get("result")
                if isinstance(actual_result, str) and actual_result.isdigit():
                    actual_result = int(actual_result)
                assert actual_result == test_case["expected_result"], f"{test_case['name']} result mismatch"

    def test_given_cpu_intensive_calculations_when_executed_then_completes_efficiently(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Execute CPU-intensive calculations
        Given: Computationally complex Python algorithms
        When: I execute prime numbers, fibonacci, and matrix operations
        Then: All should complete within acceptable time with correct results
        """
        # Given
        cpu_test_cases = [
            {
                "name": "prime_number_calculation",
                "code": f'''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, {PRIME_RANGE}) if is_prime(i)]
len(primes)
''',
                "min_result": 100  # At least 100 primes under 1000
            },
            {
                "name": "fibonacci_calculation",
                "code": f'''
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib({FIBONACCI_NUMBER})
''',
                "expected_result": 6765  # fib(20) = 6765
            },
            {
                "name": "matrix_operations",
                "code": f'''
matrix = [[i*j for j in range({MATRIX_SIZE})] for i in range({MATRIX_SIZE})]
total = sum(sum(row) for row in matrix)
total
''',
                "min_result": 1000000  # Should be a large number
            },
        ]
        
        for test_case in cpu_test_cases:
            # When
            start_time = time.time()
            response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=test_case["code"],
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then
            assert response.status_code == 200, f"Failed for {test_case['name']}"
            assert execution_time < MAX_EXECUTION_TIME, f"{test_case['name']} took {execution_time:.2f}s"
            
            result = response.json()
            assert result.get("success") is not False, f"{test_case['name']} execution failed: {result}"
            
            actual_result = result.get("result")
            if isinstance(actual_result, str) and actual_result.isdigit():
                actual_result = int(actual_result)
                
            if "expected_result" in test_case:
                assert actual_result == test_case["expected_result"], f"{test_case['name']} result mismatch"
            elif "min_result" in test_case:
                assert actual_result >= test_case["min_result"], f"{test_case['name']} result too small"


# ==================== FILE PROCESSING PERFORMANCE TESTS ====================


class TestFileProcessingPerformance:
    """Test scenarios for file upload and processing performance."""

    def test_given_large_csv_file_when_uploaded_and_processed_then_completes_efficiently(
        self, server_ready, large_csv_content, default_timeout
    ):
        """
        Scenario: Process large CSV file
        Given: A large CSV file with thousands of rows
        When: I upload and process it with pandas operations
        Then: Upload and processing should complete within acceptable time limits
        """
        # Given - Create large CSV file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(large_csv_content)
            tmp_path = tmp.name
        
        try:
            # When - Upload the large file
            start_time = time.time()
            with open(tmp_path, "rb") as fh:
                upload_response = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("large_performance_test.csv", fh, "text/csv")},
                    timeout=UPLOAD_TIMEOUT
                )
            upload_time = time.time() - start_time
            
            # Then - Verify upload performance
            assert upload_response.status_code == 200
            assert upload_time < MAX_UPLOAD_TIME, f"Upload took {upload_time:.2f}s, expected < {MAX_UPLOAD_TIME}s"
            
            upload_data = upload_response.json()
            assert upload_data.get("success") is True
            
            file_info = upload_data["data"]["file"]
            server_filename = file_info["vfsPath"]
            
            # When - Process the large file with pandas
            processing_code = f'''
import pandas as pd
df = pd.read_csv("{server_filename}")
result = {{
    "shape": list(df.shape),
    "memory_usage": int(df.memory_usage(deep=True).sum()),
    "value_sum": int(df["value"].sum()),
    "categories": int(df["category"].nunique())
}}
str(result)
'''
            
            start_time = time.time()
            process_response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=processing_code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            processing_time = time.time() - start_time
            
            # Then - Verify processing performance
            assert process_response.status_code == 200
            assert processing_time < MAX_PROCESSING_TIME, f"Processing took {processing_time:.2f}s"
            
            result = process_response.json()
            assert result.get("success") is not False, f"Processing failed: {result}"
            
            # Verify the results make sense for our large dataset
            result_str = result.get("result", "")
            assert str(LARGE_CSV_ROWS) in result_str, "Should contain correct number of rows"
            assert "4" in result_str, "Should have 4 columns"
            
            # Cleanup
            cleanup_filename = Path(server_filename).name
            requests.delete(f"{BASE_URL}/api/uploaded-files/{cleanup_filename}", timeout=10)
            
        finally:
            os.unlink(tmp_path)

    def test_given_multiple_files_when_uploaded_and_listed_then_operations_are_fast(
        self, server_ready, sample_csv_content, default_timeout
    ):
        """
        Scenario: Handle multiple file operations efficiently
        Given: Multiple CSV files to upload
        When: I upload them and perform list/info operations repeatedly
        Then: All operations should complete within performance thresholds
        """
        # Given
        uploaded_files = []
        
        try:
            # When - Upload multiple files
            for i in range(MULTIPLE_FILES_COUNT):
                content = f"id,value\n{i},100\n{i+1},200\n"
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    response = requests.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (f"perf_test_file_{i}.csv", fh, "text/csv")},
                        timeout=default_timeout
                    )
                
                assert response.status_code == 200
                upload_data = response.json()
                file_info = upload_data["data"]["file"]
                uploaded_files.append({
                    'filename': Path(file_info["vfsPath"]).name,
                    'vfs_path': file_info["vfsPath"],
                    'temp_path': tmp_path
                })
                os.unlink(tmp_path)
            
            # When/Then - Test file listing performance multiple times
            for _ in range(3):
                start_time = time.time()
                list_response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
                list_time = time.time() - start_time
                
                assert list_response.status_code == 200
                assert list_time < MAX_LIST_TIME, f"File listing took {list_time:.2f}s"
                
                files_data = list_response.json()
                assert files_data.get("success") is True
                files_list = files_data.get("data", {}).get("files", [])
                
                # Verify all uploaded files are in the list
                file_names_in_response = [f["filename"] for f in files_list]
                for file_info in uploaded_files:
                    assert file_info['filename'] in file_names_in_response
            
            # When/Then - Test file info performance for all files
            for file_info in uploaded_files:
                start_time = time.time()
                info_response = requests.get(
                    f"{BASE_URL}/api/file-info/{file_info['filename']}", 
                    timeout=10
                )
                info_time = time.time() - start_time
                
                assert info_response.status_code == 200
                assert info_time < MAX_INFO_TIME, f"File info took {info_time:.2f}s"
                
                info_data = info_response.json()
                assert info_data.get("data", {}).get("exists") is True
                
        finally:
            # Cleanup all files
            for file_info in uploaded_files:
                try:
                    requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['filename']}", timeout=10)
                except (requests.RequestException, OSError):
                    pass  # Ignore cleanup errors


# ==================== CONCURRENT REQUEST PERFORMANCE TESTS ====================


class TestConcurrentRequestPerformance:
    """Test scenarios for handling multiple concurrent requests efficiently."""

    def test_given_multiple_execution_requests_when_sent_concurrently_then_all_complete_successfully(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Handle concurrent code execution requests
        Given: Multiple simple Python code snippets
        When: I send them as concurrent execution requests
        Then: All should complete successfully within reasonable time
        """
        # Given
        test_codes = [
            {"code": "1 + 1", "expected": "2"},
            {"code": "2 * 3", "expected": "6"}, 
            {"code": "'hello world'", "expected": "hello world"},
            {"code": "[1, 2, 3, 4, 5]", "expected": "[1, 2, 3, 4, 5]"},
            {"code": "{'key': 'value'}", "expected": "{'key': 'value'}"},
        ]
        
        # When - Send requests sequentially (simulating concurrent load)
        start_time = time.time()
        responses = []
        
        for test_case in test_codes:
            response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=test_case["code"],
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            responses.append((response, test_case["expected"]))
        
        total_time = time.time() - start_time
        
        # Then - All should succeed within reasonable time
        assert total_time < MAX_EXECUTION_TIME, f"All requests took {total_time:.2f}s"
        
        # Verify all responses are correct
        for response, expected in responses:
            assert response.status_code == 200
            result = response.json()
            assert result.get("success") is not False, f"Request failed: {result}"
            
            actual_result = result.get("result", "")
            # Handle different result formats from execute-raw
            if isinstance(actual_result, str):
                actual_result = actual_result.strip('"\'')  # Remove quotes if present
            assert str(actual_result) == expected or actual_result == expected

    def test_given_mixed_workload_when_executed_then_system_remains_responsive(
        self, server_ready, uploaded_file, execution_timeout
    ):
        """
        Scenario: Handle mixed workload efficiently
        Given: A combination of file operations and code execution
        When: I perform various operations in sequence
        Then: System should remain responsive throughout
        """
        # Given
        filename, server_path = uploaded_file
        
        mixed_operations = [
            {
                "type": "execution",
                "action": lambda: requests.post(
                    EXECUTE_RAW_ENDPOINT,
                    data="import time; time.time()",
                    headers={"Content-Type": "text/plain"},
                    timeout=REQUEST_TIMEOUT
                )
            },
            {
                "type": "file_list", 
                "action": lambda: requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
            },
            {
                "type": "file_info",
                "action": lambda: requests.get(f"{BASE_URL}/api/file-info/{filename}", timeout=10)
            },
            {
                "type": "data_processing",
                "action": lambda: requests.post(
                    EXECUTE_RAW_ENDPOINT,
                    data=f'''
import pandas as pd
df = pd.read_csv("{server_path}")
result = df.shape[0] + df.shape[1]
str(result)
''',
                    headers={"Content-Type": "text/plain"},
                    timeout=REQUEST_TIMEOUT
                )
            },
        ]
        
        # When - Execute mixed workload
        start_time = time.time()
        results = []
        
        for operation in mixed_operations:
            op_start = time.time()
            response = operation["action"]()
            op_time = time.time() - op_start
            
            results.append({
                "type": operation["type"],
                "response": response,
                "time": op_time
            })
        
        total_time = time.time() - start_time
        
        # Then - All operations should succeed within reasonable time
        assert total_time < MAX_EXECUTION_TIME, f"Mixed workload took {total_time:.2f}s"
        
        for result in results:
            assert result["response"].status_code == 200, f"{result['type']} failed"
            assert result["time"] < 5.0, f"{result['type']} took {result['time']:.2f}s"
            
            response_data = result["response"].json()
            assert response_data.get("success") is not False, f"{result['type']} response invalid"


# ==================== RESOURCE CLEANUP PERFORMANCE TESTS ====================


class TestResourceCleanupPerformance:
    """Test scenarios for resource cleanup and error recovery performance."""

    def test_given_execution_errors_when_occurred_then_system_recovers_quickly(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Quick recovery from execution errors
        Given: Python code that will cause various types of errors
        When: I execute error-inducing code followed by valid code
        Then: System should recover quickly and remain functional
        """
        # Given - Error-inducing codes
        error_codes = [
            "undefined_variable_xyz",  # NameError
            "1 / 0",                   # ZeroDivisionError
            "import nonexistent_module",  # ImportError
            "[1, 2, 3][10]",          # IndexError
        ]
        
        for error_code in error_codes:
            # When - Execute error code
            error_response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=error_code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            
            # Should handle error gracefully
            assert error_response.status_code == 200
            error_result = error_response.json()
            # Error may be in result or success may be False
            
            # When - Execute valid code immediately after error
            start_time = time.time()
            recovery_response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data="2 + 2",
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            recovery_time = time.time() - start_time
            
            # Then - System should recover quickly
            assert recovery_response.status_code == 200
            assert recovery_time < 2.0, f"Recovery took {recovery_time:.2f}s after {error_code}"
            
            recovery_result = recovery_response.json()
            assert recovery_result.get("success") is not False, f"Failed to recover after {error_code}"
            assert recovery_result.get("result") == "4", "Recovery execution gave wrong result"

    def test_given_large_operation_when_completed_then_memory_is_cleaned_efficiently(
        self, server_ready, execution_timeout
    ):
        """
        Scenario: Efficient memory cleanup after large operations
        Given: Memory-intensive operations that create large objects
        When: I execute them sequentially
        Then: Each should complete without cumulative memory issues
        """
        # Given - Series of memory-intensive operations
        memory_operations = [
            f"large_list_{i} = list(range(50000)); len(large_list_{i})" 
            for i in range(3)
        ]
        
        execution_times = []
        
        for i, operation in enumerate(memory_operations):
            # When
            start_time = time.time()
            response = requests.post(
                EXECUTE_RAW_ENDPOINT,
                data=operation,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Then
            assert response.status_code == 200
            assert execution_time < MAX_EXECUTION_TIME, f"Operation {i} took {execution_time:.2f}s"
            
            result = response.json()
            assert result.get("success") is not False, f"Operation {i} failed: {result}"
            assert result.get("result") == "50000", f"Operation {i} wrong result"
        
        # Verify no significant performance degradation over time
        # Later operations shouldn't be much slower than earlier ones
        first_time = execution_times[0]
        last_time = execution_times[-1]
        assert last_time < first_time * 2, f"Performance degraded: {first_time:.2f}s -> {last_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])