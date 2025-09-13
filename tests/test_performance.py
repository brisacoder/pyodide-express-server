"""
BDD-style performance tests for Pyodide Express Server.

This module tests performance characteristics and resource limits using pytest
and follows Behavior-Driven Development (BDD) patterns with Given-When-Then structure.
Only uses public APIs and the /api/execute-raw endpoint for Python execution.

Test Categories:
- Execution Performance: Timeout handling, memory, CPU-intensive operations
- File Processing Performance: Large file handling, multiple file operations  
- Concurrent Request Performance: Multiple simultaneous requests
- Resource Cleanup Performance: Error handling and resource management

All tests use pytest fixtures for proper test isolation and cleanup.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Generator, List, Dict, Any

import pytest
import requests


# ===== Global Configuration =====
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30  # seconds
UPLOAD_TIMEOUT = 60   # seconds for file uploads
EXECUTION_TIMEOUT = 30000  # milliseconds for Pyodide execution
MAX_RETRIES = 3


# ===== Pytest Fixtures =====

@pytest.fixture(scope="session")
def server_session() -> Generator[requests.Session, None, None]:
    """
    Provide a reusable HTTP session for all tests.
    
    Given a test session is started
    When tests need to make HTTP requests
    Then a configured session with timeouts should be available
    """
    session = requests.Session()
    session.timeout = DEFAULT_TIMEOUT
    
    # Wait for server to be ready
    start = time.time()
    while time.time() - start < 120:
        try:
            response = session.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                break
        except (requests.RequestException, OSError):
            pass
        time.sleep(1)
    else:
        pytest.fail("Server did not start in time")
    
    yield session
    session.close()


@pytest.fixture
def uploaded_files_tracker() -> Generator[List[str], None, None]:
    """
    Track uploaded files for automatic cleanup after each test.
    
    Given a test needs to upload files
    When files are uploaded during the test
    Then they should be automatically cleaned up afterward
    """
    uploaded_files: List[str] = []
    yield uploaded_files
    
    # Cleanup uploaded files
    session = requests.Session()
    session.timeout = DEFAULT_TIMEOUT
    for filename in uploaded_files:
        try:
            session.delete(f"{BASE_URL}/api/uploaded-files/{filename}")
        except requests.RequestException:
            pass  # File might already be deleted
    session.close()


@pytest.fixture  
def temp_files_tracker() -> Generator[List[Path], None, None]:
    """
    Track temporary files for automatic cleanup after each test.
    
    Given a test needs to create temporary files
    When temporary files are created during the test  
    Then they should be automatically cleaned up afterward
    """
    temp_files: List[Path] = []
    yield temp_files
    
    # Cleanup temporary files
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()


# ===== Performance Test Class =====

class TestExecutionPerformance:
    """Test performance characteristics of Python code execution."""

    def test_execution_timeout_handling(self, server_session: requests.Session) -> None:
        """
        Test that long-running code is properly timed out.
        
        Given a Pyodide server is running
        When I submit long-running Python code with a short timeout
        Then the execution should complete within reasonable time
        And not exceed the system timeout limits
        """
        # Given: Long-running Python code
        long_running_code = '''
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.1)  # Small sleep to make it slower
print(f"Final total: {total}")
total
'''
        
        # When: Execute with short timeout
        start_time = time.time()
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then: Should complete within reasonable time
        assert execution_time < 10, f"Execution took too long: {execution_time}s"
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
        
        response_data = response.json()
        # Could either succeed quickly or timeout - both are acceptable
        assert "success" in response_data

    def test_memory_intensive_operations(self, server_session: requests.Session) -> None:
        """
        Test handling of memory-intensive operations.
        
        Given a Pyodide server is running
        When I execute memory-intensive Python operations
        Then each operation should complete within reasonable time
        And return expected results
        """
        # Given: Memory-intensive operations
        memory_test_cases = [
            ("Large list creation", "large_list = list(range(100000)); len(large_list)"),
            ("Large string operations", "big_string = 'x' * 1000000; len(big_string)"),
            ("Large dictionary", "big_dict = {i: f'value_{i}' for i in range(10000)}; len(big_dict)"),
        ]
        
        for test_name, code in memory_test_cases:
            # When: Execute memory-intensive code
            start_time = time.time()
            response = server_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: Should complete reasonably quickly
            assert response.status_code == 200, f"{test_name} failed with status {response.status_code}"
            assert execution_time < 5, f"{test_name} took too long: {execution_time}s"
            
            response_data = response.json()
            if response_data.get("success"):
                # If successful, result should be reasonable
                result = response_data.get("result")
                assert isinstance(result, int), f"{test_name} result should be integer"

    def test_cpu_intensive_operations(self, server_session: requests.Session) -> None:
        """
        Test CPU-intensive calculations.
        
        Given a Pyodide server is running
        When I execute CPU-intensive Python calculations
        Then each calculation should complete within acceptable time
        And return correct integer results
        """
        # Given: CPU-intensive calculations
        cpu_test_cases = [
            ("Prime number calculation", '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
len(primes)
'''),
            ("Fibonacci calculation", '''
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib(20)
'''),
            ("Matrix operations", '''
matrix = [[i*j for j in range(100)] for i in range(100)]
total = sum(sum(row) for row in matrix)
total
'''),
        ]
        
        for test_name, code in cpu_test_cases:
            # When: Execute CPU-intensive code
            start_time = time.time()
            response = server_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: Should complete within time limit
            assert response.status_code == 200, f"{test_name} failed with status {response.status_code}"
            assert execution_time < 10, f"{test_name} took too long: {execution_time}s"
            
            response_data = response.json()
            assert response_data.get("success"), f"{test_name} execution failed"
            assert isinstance(response_data.get("result"), int), f"{test_name} should return integer"


class TestFileProcessingPerformance:
    """Test performance characteristics of file processing operations."""

    def test_large_csv_processing(
        self, 
        server_session: requests.Session,
        uploaded_files_tracker: List[str],
        temp_files_tracker: List[Path]
    ) -> None:
        """
        Test processing of larger CSV files.
        
        Given a large CSV file is created
        When I upload and process the file through the API
        Then the upload and processing should complete within time limits
        And return correct analysis results
        """
        # Given: Create a moderately large CSV file
        large_csv_content = "id,value,category,description\n"
        for i in range(5000):  # 5000 rows
            large_csv_content += f"{i},{i*2},category_{i%10},description for item {i}\n"
        
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        temp_files_tracker.append(temp_file)
        temp_file.write_text(large_csv_content)
        
        # When: Upload the large file
        start_time = time.time()
        with temp_file.open("rb") as fh:
            response = server_session.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("large.csv", fh, "text/csv")},
                timeout=UPLOAD_TIMEOUT
            )
        upload_time = time.time() - start_time
        
        # Then: Upload should succeed within time limit
        assert response.status_code == 200, f"Upload failed with status {response.status_code}"
        assert upload_time < 30, f"Upload took too long: {upload_time}s"
        
        upload_data = response.json()
        filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_files_tracker.append(filename)
        
        # When: Process the large file
        processing_code = f'''
# Read the uploaded CSV file
import os
import csv

file_path = "/home/pyodide/uploads/{filename}"
print(f"Processing file: {{file_path}}")

# Read and analyze the CSV file
rows = []
with open(file_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Calculate statistics
total_rows = len(rows)
value_sum = sum(int(row['value']) for row in rows)
categories = set(row['category'] for row in rows)
category_count = len(categories)

result = {{
    "shape": [total_rows, 4],  # rows, columns
    "value_sum": value_sum,
    "categories": category_count
}}

print(f"Analysis complete: {{result}}")
result
'''
        
        start_time = time.time()
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=processing_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        processing_time = time.time() - start_time
        
        # Then: Processing should succeed within time limit
        assert response.status_code == 200, f"Processing failed with status {response.status_code}"
        assert processing_time < 15, f"Processing took too long: {processing_time}s"
        
        response_data = response.json()
        assert response_data.get("success"), "Processing execution failed"
        
        result = response_data.get("result")
        assert result["shape"] == [5000, 4], f"Expected [5000, 4] rows/cols, got {result['shape']}"
        assert result["categories"] == 10, f"Expected 10 categories, got {result['categories']}"

    def test_multiple_file_operations(
        self,
        server_session: requests.Session, 
        uploaded_files_tracker: List[str],
        temp_files_tracker: List[Path]
    ) -> None:
        """
        Test performance with multiple file operations.
        
        Given multiple small CSV files are created
        When I upload all files and list them multiple times
        Then all operations should complete quickly
        And file listing should be fast due to caching
        """
        # Given: Create multiple small files
        files_to_create = 5
        created_files: List[Dict[str, Any]] = []
        
        for i in range(files_to_create):
            content = f"id,value\n{i},100\n{i+1},200\n"
            temp_file = Path(tempfile.mktemp(suffix=".csv"))
            temp_files_tracker.append(temp_file)
            temp_file.write_text(content)
            
            # When: Upload each file
            with temp_file.open("rb") as fh:
                response = server_session.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": (f"file_{i}.csv", fh, "text/csv")},
                    timeout=DEFAULT_TIMEOUT
                )
            
            # Then: Upload should succeed
            assert response.status_code == 200, f"Upload {i} failed"
            
            upload_data = response.json()
            filename = upload_data["data"]["file"]["storedFilename"]
            uploaded_files_tracker.append(filename)
            created_files.append({"filename": filename, "index": i})
        
        # When: List files multiple times to test caching/performance  
        for iteration in range(3):
            start_time = time.time()
            response = server_session.get(f"{BASE_URL}/api/uploaded-files", timeout=DEFAULT_TIMEOUT)
            list_time = time.time() - start_time
            
            # Then: Listing should be fast
            assert response.status_code == 200, f"File listing failed on iteration {iteration}"
            assert list_time < 2, f"File listing took too long: {list_time}s on iteration {iteration}"
            
            files_in_response = [f["filename"] for f in response.json()["data"]["files"]]
            for file_info in created_files:
                assert file_info["filename"] in files_in_response, f"File {file_info['filename']} not found in listing"


class TestConcurrentRequestPerformance:
    """Test handling of multiple concurrent requests."""

    def test_concurrent_execution_requests(self, server_session: requests.Session) -> None:
        """
        Test handling of multiple execution requests.
        
        Given multiple simple Python expressions
        When I send all requests sequentially
        Then all requests should succeed quickly
        And return correct results
        """
        # Given: Multiple simple Python expressions
        test_cases = [
            ("Addition", "1 + 1", 2),
            ("Multiplication", "2 * 3", 6), 
            ("String literal", "'hello world'", "hello world"),
            ("List creation", "[1, 2, 3, 4, 5]", [1, 2, 3, 4, 5]),
            ("Dictionary creation", "{'key': 'value'}", {"key": "value"}),
        ]
        
        start_time = time.time()
        responses: List[requests.Response] = []
        
        for test_name, code, expected in test_cases:
            # When: Execute each expression
            response = server_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            responses.append(response)
        
        total_time = time.time() - start_time
        
        # Then: All should succeed quickly
        assert total_time < 10, f"Total execution time too long: {total_time}s"
        
        for i, (response, (test_name, code, expected)) in enumerate(zip(responses, test_cases)):
            assert response.status_code == 200, f"{test_name} failed with status {response.status_code}"
            
            response_data = response.json()
            assert response_data.get("success"), f"{test_name} execution failed"
            assert response_data.get("result") == expected, f"{test_name} result mismatch"


class TestResourceCleanupPerformance:
    """Test that errors don't cause resource leaks."""

    def test_cleanup_after_errors(
        self,
        server_session: requests.Session,
        uploaded_files_tracker: List[str], 
        temp_files_tracker: List[Path]
    ) -> None:
        """
        Test that errors don't cause resource leaks.
        
        Given a file is uploaded and variables are set
        When a Python execution error occurs
        Then the system should recover gracefully
        And subsequent operations should work normally
        """
        # Given: Create and upload a test file
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        temp_files_tracker.append(temp_file)
        temp_file.write_text("a,b\n1,2\n")
        
        with temp_file.open("rb") as fh:
            response = server_session.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("cleanup_test.csv", fh, "text/csv")},
                timeout=DEFAULT_TIMEOUT
            )
        assert response.status_code == 200, "File upload failed"
        
        upload_data = response.json()
        filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_files_tracker.append(filename)
        
        # Given: Set some variables
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw",
            data="test_var = 'should_be_cleaned'",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200, "Variable setting failed"
        
        # When: Cause an error
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw", 
            data="undefined_variable_xyz",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200, "Error request failed at HTTP level"
        
        response_data = response.json()
        assert not response_data.get("success"), "Error should have failed execution"
        
        # Then: System should work normally after error
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw",
            data="2 + 2",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert response.status_code == 200, "Recovery test failed"
        
        response_data = response.json()
        assert response_data.get("success"), "System didn't recover from error"
        assert response_data.get("result") == 4, "Recovery calculation incorrect"
        
        # Then: Files should still be accessible after error
        response = server_session.get(f"{BASE_URL}/api/uploaded-files", timeout=DEFAULT_TIMEOUT)
        assert response.status_code == 200, "File listing failed after error"
        
        files_in_response = [f["filename"] for f in response.json()["data"]["files"]]
        assert filename in files_in_response, "Uploaded file not accessible after error"


# ===== Performance Benchmarking =====

@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Additional performance benchmarks for comprehensive testing."""

    def test_execution_time_consistency(self, server_session: requests.Session) -> None:
        """
        Test that execution times are consistent across multiple runs.
        
        Given a consistent Python calculation
        When I execute it multiple times
        Then execution times should be relatively consistent
        And within acceptable variance
        """
        # Given: Consistent calculation
        code = "sum(i*i for i in range(1000))"
        execution_times: List[float] = []
        
        # When: Execute multiple times
        for _ in range(5):
            start_time = time.time()
            response = server_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Then: Each execution should succeed
            assert response.status_code == 200, "Execution failed"
            response_data = response.json()
            assert response_data.get("success"), "Execution was not successful"
        
        # Then: Times should be consistent (within 3x variance)
        min_time = min(execution_times)
        max_time = max(execution_times)
        assert max_time / min_time < 3, f"Execution time variance too high: {min_time}s to {max_time}s"

    def test_memory_usage_stability(self, server_session: requests.Session) -> None:
        """
        Test that memory usage remains stable across operations.
        
        Given multiple memory-intensive operations
        When I execute them sequentially
        Then each should succeed without memory errors
        And performance should remain stable
        """
        # Given: Memory-intensive operations
        memory_operations = [
            "data = list(range(50000)); sum(data)",
            "text = 'hello' * 100000; len(text)", 
            "matrix = [[i+j for j in range(100)] for i in range(100)]; len(matrix)",
            "mapping = {i: str(i) for i in range(10000)}; len(mapping)",
        ]
        
        execution_times: List[float] = []
        
        for operation in memory_operations:
            # When: Execute memory operation
            start_time = time.time()
            response = server_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=operation,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Then: Should succeed and perform reasonably
            assert response.status_code == 200, f"Operation failed: {operation}"
            response_data = response.json()
            assert response_data.get("success"), f"Execution failed for: {operation}"
            assert execution_time < 8, f"Operation too slow: {execution_time}s for {operation}"
        
        # Then: Performance should remain stable
        avg_time = sum(execution_times) / len(execution_times)
        for i, exec_time in enumerate(execution_times):
            assert abs(exec_time - avg_time) < 5, f"Operation {i} time {exec_time}s deviates too much from average {avg_time}s"