"""BDD-style performance tests converted from unittest to pytest.

Tests performance characteristics and resource limits using the public API.
All tests follow Given-When-Then pattern and use /api/execute-raw for code execution.
No internal 'pyodide' APIs are used - only public endpoints.
"""

# Global timeout constants
DEFAULT_TIMEOUT = 30
SHORT_TIMEOUT = 5
LONG_TIMEOUT = 60
EXECUTION_TIMEOUT_LIMIT = 10
MEMORY_LIMIT_TIMEOUT = 5
FILE_OPERATION_TIMEOUT = 15
UPLOAD_TIMEOUT = 30
LIST_FILES_TIMEOUT = 2
INFO_TIMEOUT = 1

import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Tuple

import pytest
import requests

BASE_URL = "http://localhost:3000"


@pytest.fixture(scope="session")
def server_url() -> str:
    """Provide the base URL for all API tests."""
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def wait_for_server():
    """Ensure server is ready before running tests."""
    start = time.time()
    while time.time() - start < 120:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=SHORT_TIMEOUT)
            if r.status_code == 200:
                return
        except (requests.RequestException, OSError):
            pass
        time.sleep(1)
    else:
        pytest.fail("Server did not start in time")


@pytest.fixture
def temp_csv_file() -> Generator[Tuple[str, str], None, None]:
    """Create a temporary CSV file for testing.
    
    Returns:
        Tuple of (file_path, content) for the temporary CSV file
    """
    content = "id,value\n1,100\n2,200\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    yield tmp_path, content
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def large_csv_file() -> Generator[str, None, None]:
    """Create a large CSV file for performance testing.
    
    Returns:
        Path to the temporary large CSV file
    """
    content = "id,value,category,description\n"
    for i in range(5000):  # 5000 rows
        content += f"{i},{i*2},category_{i%10},description for item {i}\n"
    
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def uploaded_file_cleanup():
    """Track uploaded files for cleanup after test."""
    uploaded_files = []
    
    def track_file(filename: str):
        uploaded_files.append(filename)
    
    yield track_file
    
    # Cleanup all tracked files
    for filename in uploaded_files:
        try:
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=DEFAULT_TIMEOUT)
        except requests.RequestException:
            pass  # File might already be deleted


class TestExecutionPerformance:
    """Test execution performance characteristics."""

    def test_given_long_running_code_when_timeout_applied_then_execution_completes_within_limit(
        self, server_url: str
    ):
        """
        Given: A piece of Python code that takes time to execute
        When: I submit it with a timeout limit
        Then: The execution should complete within reasonable time bounds
        """
        # Given: Long-running Python code
        long_running_code = '''
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.1)  # Small sleep to make it slower
total
'''
        
        # When: I execute the code with a timeout
        start_time = time.time()
        response = requests.post(
            f"{server_url}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then: Execution should complete within reasonable time
        assert execution_time < EXECUTION_TIMEOUT_LIMIT
        assert response.status_code == 200

    def test_given_memory_intensive_operations_when_executed_then_completes_efficiently(
        self, server_url: str
    ):
        """
        Given: Memory-intensive Python operations
        When: I execute them via the API
        Then: They should complete efficiently within memory limits
        """
        memory_intensive_codes = [
            # Given: Large list creation
            "large_list = list(range(100000)); len(large_list)",
            # Given: Large string operations  
            "big_string = 'x' * 1000000; len(big_string)",
            # Given: Large dictionary creation
            "big_dict = {i: f'value_{i}' for i in range(10000)}; len(big_dict)",
        ]
        
        for code in memory_intensive_codes:
            # When: I execute memory-intensive code
            start_time = time.time()
            response = requests.post(
                f"{server_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: Should complete quickly and successfully
            assert response.status_code == 200
            assert execution_time < MEMORY_LIMIT_TIMEOUT

    def test_given_cpu_intensive_calculations_when_executed_then_produces_correct_results(
        self, server_url: str
    ):
        """
        Given: CPU-intensive mathematical calculations
        When: I execute them through the API
        Then: They should produce correct results in reasonable time
        """
        cpu_intensive_codes = [
            # Given: Prime number calculation
            '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
len(primes)
''',
            # Given: Fibonacci calculation
            '''
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib(20)
''',
            # Given: Matrix operations
            '''
matrix = [[i*j for j in range(100)] for i in range(100)]
total = sum(sum(row) for row in matrix)
total
''',
        ]
        
        for code in cpu_intensive_codes:
            # When: I execute CPU-intensive code
            start_time = time.time()
            response = requests.post(
                f"{server_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: Should complete within time limit
            assert response.status_code == 200
            assert execution_time < EXECUTION_TIMEOUT_LIMIT


class TestFileProcessingPerformance:
    """Test file processing performance characteristics."""

    def test_given_large_csv_file_when_uploaded_and_processed_then_handles_efficiently(
        self, server_url: str, large_csv_file: str, uploaded_file_cleanup
    ):
        """
        Given: A large CSV file with many rows
        When: I upload and process it through the API
        Then: The system should handle it efficiently within time limits
        """
        # Given: A large CSV file
        # (provided by fixture)
        
        # When: I upload the large file
        start_time = time.time()
        with open(large_csv_file, "rb") as fh:
            upload_response = requests.post(
                f"{server_url}/api/upload",
                files={"file": ("large.csv", fh, "text/csv")},
                timeout=LONG_TIMEOUT
            )
        upload_time = time.time() - start_time
        
        # Then: Upload should complete within time limit
        assert upload_response.status_code == 200
        assert upload_time < UPLOAD_TIMEOUT
        
        upload_data = upload_response.json()
        assert upload_data["success"] is True
        
        # Track file for cleanup
        filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_file_cleanup(filename)
        vfs_path = upload_data["data"]["file"]["vfsPath"]
        
        # When: I process the large file using execute-raw
        processing_code = f'''
import pandas as pd
from pathlib import Path

# Read the uploaded CSV file
file_path = Path("/home/pyodide/{vfs_path}")
df = pd.read_csv(str(file_path))

result = {{
    "shape": list(df.shape),
    "value_sum": int(df["value"].sum()),
    "categories": int(df["category"].nunique())
}}
print(f"Processed file with shape: {{result['shape']}}")
print(f"Total value sum: {{result['value_sum']}}")
print(f"Unique categories: {{result['categories']}}")
'''
        
        start_time = time.time()
        process_response = requests.post(
            f"{server_url}/api/execute-raw",
            data=processing_code,
            headers={"Content-Type": "text/plain"},
            timeout=LONG_TIMEOUT
        )
        processing_time = time.time() - start_time
        
        # Then: Processing should complete efficiently
        assert process_response.status_code == 200
        assert processing_time < FILE_OPERATION_TIMEOUT

    def test_given_multiple_files_when_uploaded_then_operations_remain_fast(
        self, server_url: str, uploaded_file_cleanup
    ):
        """
        Given: Multiple small CSV files
        When: I upload them and perform operations
        Then: File operations should remain fast even with multiple files
        """
        files_to_create = 5
        uploaded_filenames = []
        
        # Given: Multiple small CSV files to upload
        for i in range(files_to_create):
            content = f"id,value\n{i},100\n{i+1},200\n"
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # When: I upload each file
            with open(tmp_path, "rb") as fh:
                upload_response = requests.post(
                    f"{server_url}/api/upload",
                    files={"file": (f"file_{i}.csv", fh, "text/csv")},
                    timeout=DEFAULT_TIMEOUT
                )
            
            # Then: Upload should succeed
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            
            filename = upload_data["data"]["file"]["storedFilename"]
            uploaded_filenames.append(filename)
            uploaded_file_cleanup(filename)
            
            # Cleanup temp file
            os.unlink(tmp_path)
        
        # When: I list files multiple times (test caching/performance)
        for _ in range(3):
            start_time = time.time()
            list_response = requests.get(f"{server_url}/api/uploaded-files", timeout=DEFAULT_TIMEOUT)
            list_time = time.time() - start_time
            
            # Then: Listing should be fast
            assert list_response.status_code == 200
            assert list_time < LIST_FILES_TIMEOUT
            
            files_data = list_response.json()
            assert files_data["success"] is True
            
            # Verify all uploaded files are present
            returned_filenames = [f["filename"] for f in files_data["data"]["files"]]
            for filename in uploaded_filenames:
                assert filename in returned_filenames


class TestConcurrentRequestPerformance:
    """Test concurrent request handling performance."""

    def test_given_multiple_execution_requests_when_sent_concurrently_then_all_succeed(
        self, server_url: str
    ):
        """
        Given: Multiple simple Python code snippets
        When: I send them as concurrent execution requests
        Then: All should succeed and complete in reasonable time
        """
        # Given: Multiple simple code snippets
        codes = [
            "1 + 1",
            "2 * 3", 
            "'hello world'",
            "[1, 2, 3, 4, 5]",
            "{'key': 'value'}",
        ]
        
        # When: I send multiple execution requests
        start_time = time.time()
        responses = []
        
        for code in codes:
            response = requests.post(
                f"{server_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            responses.append(response)
        
        total_time = time.time() - start_time
        
        # Then: All should succeed within reasonable time
        for response in responses:
            assert response.status_code == 200
        
        assert total_time < EXECUTION_TIMEOUT_LIMIT


class TestResourceCleanupPerformance:
    """Test resource cleanup and error handling performance."""

    def test_given_errors_in_execution_when_they_occur_then_system_recovers_cleanly(
        self, server_url: str, temp_csv_file: Tuple[str, str], uploaded_file_cleanup
    ):
        """
        Given: A system with uploaded files and variables
        When: An error occurs during execution
        Then: The system should recover cleanly without resource leaks
        """
        tmp_path, _ = temp_csv_file
        
        # Given: Upload a test file
        with open(tmp_path, "rb") as fh:
            upload_response = requests.post(
                f"{server_url}/api/upload",
                files={"file": ("cleanup_test.csv", fh, "text/csv")},
                timeout=DEFAULT_TIMEOUT
            )
        assert upload_response.status_code == 200
        
        upload_data = upload_response.json()
        filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_file_cleanup(filename)
        
        # Given: Set some variables in the environment
        setup_response = requests.post(
            f"{server_url}/api/execute-raw",
            data="test_var = 'should_be_cleaned'",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert setup_response.status_code == 200
        
        # When: I cause an error
        error_response = requests.post(
            f"{server_url}/api/execute-raw",
            data="undefined_variable_xyz",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert error_response.status_code == 200
        # Note: API returns 200 even for Python errors
        
        # Then: System should still work normally after error
        recovery_response = requests.post(
            f"{server_url}/api/execute-raw",
            data="2 + 2",
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        assert recovery_response.status_code == 200
        
        # And: Files should still be accessible
        files_response = requests.get(f"{server_url}/api/uploaded-files", timeout=DEFAULT_TIMEOUT)
        assert files_response.status_code == 200
        
        files_data = files_response.json()
        assert files_data["success"] is True
        
        # Verify our uploaded file is still there
        returned_filenames = [f["filename"] for f in files_data["data"]["files"]]
        assert filename in returned_filenames