"""Performance tests for Pyodide Express Server.

BDD-style tests focusing on performance characteristics and resource limits.
Converted from unittest to pytest with proper fixtures and globals.

All tests use the /api/execute-raw endpoint exclusively and avoid internal pyodide APIs.
"""

import os
import tempfile
import time
from typing import Generator

import pytest
import requests

# ===== Configuration Constants =====


class TestConfig:
    """Configuration constants for performance testing."""

    # Server Configuration
    BASE_URL = "http://localhost:3000"

    # Timeout Constants (milliseconds for API, seconds for requests)
    PYODIDE_DEFAULT_TIMEOUT = 30000  # Default Pyodide execution timeout
    REQUEST_TIMEOUT = 60  # HTTP request timeout
    UPLOAD_TIMEOUT = 90  # File upload timeout
    SERVER_STARTUP_TIMEOUT = 120  # Server readiness timeout

    # Performance Thresholds (seconds)
    MAX_SIMPLE_EXECUTION_TIME = 5  # Simple code execution
    MAX_MEMORY_OPERATION_TIME = 8  # Memory-intensive operations
    MAX_CPU_OPERATION_TIME = 15  # CPU-intensive calculations
    MAX_FILE_UPLOAD_TIME = 30  # Large file upload
    MAX_FILE_PROCESSING_TIME = 20  # File data processing
    MAX_FILE_LISTING_TIME = 3  # File listing operations
    MAX_LIST_TIME = 5.0  # Maximum time for file listing operations
    MAX_SEQUENTIAL_REQUESTS_TIME = 20  # Multiple sequential requests
    MAX_ERROR_RECOVERY_TIME = 5  # Recovery after errors

    # Test Data Configuration
    SMALL_CSV_ROWS = 100  # Small CSV file size
    LARGE_CSV_ROWS = 5000  # Large CSV file size
    CONCURRENT_REQUESTS = 5  # Number of concurrent test requests
    MEMORY_TEST_SIZE = 100000  # Size for memory tests
    STRING_TEST_SIZE = 1000000  # Size for string tests
    DICT_TEST_SIZE = 10000  # Size for dictionary tests
    FIBONACCI_N = 20  # Fibonacci test parameter
    PRIME_LIMIT = 1000  # Prime number test limit
    MATRIX_SIZE = 100  # Matrix operation size


# ===== Fixtures =====


@pytest.fixture(scope="session")
def api_client() -> Generator[requests.Session, None, None]:
    """Provide a configured requests session for API calls."""
    session = requests.Session()

    # Wait for server to be ready
    start = time.time()
    while time.time() - start < TestConfig.SERVER_STARTUP_TIMEOUT:
        try:
            response = session.get(f"{TestConfig.BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                break
        except (requests.RequestException, requests.ConnectionError):
            pass
        time.sleep(1)
    else:
        pytest.fail("Server did not start within timeout")

    yield session
    session.close()


@pytest.fixture
def temp_csv_file() -> Generator[str, None, None]:
    """Create a temporary CSV file for testing."""
    content = "id,value,category,description\n"
    for i in range(TestConfig.SMALL_CSV_ROWS):
        content += f"{i},{i*2},category_{i % 5},description for item {i}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def large_csv_file() -> Generator[str, None, None]:
    """Create a large CSV file for performance testing."""
    content = "id,value,category,description\n"
    for i in range(TestConfig.LARGE_CSV_ROWS):
        content += f"{i},{i*2},category_{i % 10},description for item {i}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def uploaded_files_cleanup():
    """Track and cleanup uploaded files."""
    uploaded_files = []

    def track_file(filename: str):
        uploaded_files.append(filename)

    yield track_file

    # Cleanup all tracked files
    session = requests.Session()
    for filename in uploaded_files:
        try:
            session.delete(f"{TestConfig.BASE_URL}/api/uploaded-files/{filename}", timeout=10)
        except requests.RequestException:
            pass  # File might already be deleted
    session.close()


# ===== Test Classes =====


class TestExecutionPerformance:
    """Test performance characteristics of Python code execution."""

    def test_given_long_running_code_when_executed_with_timeout_then_completes_within_limits(
        self, api_client: requests.Session
    ):
        """
        Given: A long-running Python code with timeout
        When: Code is executed via /api/execute-raw
        Then: Execution completes within reasonable time limits
        """
        # Given
        long_running_code = """
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.1)  # Small sleep to make it slower
print(f"Total: {total}")
"""

        # When
        start_time = time.time()
        response = api_client.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        execution_time = time.time() - start_time

        # Then
        assert (
            execution_time < TestConfig.MAX_SIMPLE_EXECUTION_TIME
        ), f"Execution took {execution_time:.2f}s, expected < {TestConfig.MAX_SIMPLE_EXECUTION_TIME}s"
        assert response.status_code == 200

        result = response.json()
        # Should either succeed or handle timeout gracefully
        assert "success" in result

    def test_given_memory_intensive_code_when_executed_then_handles_large_data_structures(
        self, api_client: requests.Session
    ):
        """
        Given: Memory-intensive Python operations
        When: Code creates large data structures
        Then: Operations complete within memory limits
        """
        memory_test_cases = [
            (
                "large_list = list(range(100000))\nprint(f'List length: {len(large_list)}')",
                "large list creation",
            ),
            (
                "big_string = 'x' * 1000000\nprint(f'String length: {len(big_string)}')",
                "large string operations",
            ),
            (
                "big_dict = {i: f'value_{i}' for i in range("
                + str(TestConfig.DICT_TEST_SIZE)
                + ")}\n"
                "print(f'Dict size: {len(big_dict)}')",
                "large dictionary",
            ),
        ]

        for code, description in memory_test_cases:
            # Given
            start_time = time.time()

            # When
            response = api_client.post(
                f"{TestConfig.BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            execution_time = time.time() - start_time

            # Then
            assert (
                execution_time < TestConfig.MAX_MEMORY_OPERATION_TIME
            ), f"{description} took {execution_time:.2f}s"
            assert response.status_code == 200

            result = response.json()
            assert (
                result.get("success") is True
            ), f"{description} failed: {result.get('error', 'Unknown error')}"
            stdout_data = result.get("data", {}).get("stdout", "")
            assert "length" in stdout_data.lower() or "size" in stdout_data.lower()

    def test_given_cpu_intensive_code_when_executed_then_completes_efficiently(
        self, api_client: requests.Session
    ):
        """
        Given: CPU-intensive calculations
        When: Code performs complex computations
        Then: Calculations complete within performance limits
        """
        cpu_test_cases = [
            (
                """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
print(f"Prime count: {len(primes)}")
""",
                "prime number calculation",
            ),
            (
                """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

result = fib(20)
print(f"Fibonacci result: {result}")
""",
                "fibonacci calculation",
            ),
            (
                """
matrix = [[i*j for j in range(100)] for i in range(100)]
total = sum(sum(row) for row in matrix)
print(f"Matrix total: {total}")
""",
                "matrix operations",
            ),
        ]

        for code, description in cpu_test_cases:
            # Given
            start_time = time.time()

            # When
            response = api_client.post(
                f"{TestConfig.BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            execution_time = time.time() - start_time

            # Then
            assert (
                execution_time < TestConfig.MAX_CPU_OPERATION_TIME
            ), f"{description} took {execution_time:.2f}s"
            assert response.status_code == 200

            result = response.json()
            assert (
                result.get("success") is True
            ), f"{description} failed: {result.get('error', 'Unknown error')}"
            stdout_data = result.get("data", {}).get("stdout", "")
            assert (
                "result" in stdout_data.lower()
                or "count" in stdout_data.lower()
                or "total" in stdout_data.lower()
            )


class TestFileProcessingPerformance:
    """Test performance of file upload and processing operations."""

    def test_given_large_csv_file_when_uploaded_and_processed_then_completes_within_limits(
        self, api_client: requests.Session, large_csv_file: str, uploaded_files_cleanup
    ):
        """
        Given: A large CSV file
        When: File is uploaded and processed with pandas
        Then: Operations complete within performance limits
        """
        # Given - Upload the large file
        start_time = time.time()

        with open(large_csv_file, "rb") as fh:
            response = api_client.post(
                f"{TestConfig.BASE_URL}/api/upload",
                files={"file": ("large.csv", fh, "text/csv")},
                timeout=TestConfig.UPLOAD_TIMEOUT,
            )
        upload_time = time.time() - start_time

        # Then - Upload should succeed within time limits
        assert (
            upload_time < TestConfig.MAX_FILE_UPLOAD_TIME
        ), f"Upload took {upload_time:.2f}s"
        assert response.status_code == 200

        upload_data = response.json()
        assert upload_data.get("success") is True

        # Track for cleanup
        uploaded_filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_files_cleanup(uploaded_filename)

        # When - Process the large file (using basic Python since pandas may not be available)
        processing_code = f"""
from pathlib import Path

# Load the uploaded file
upload_path = Path("/uploads/{uploaded_filename}")
if upload_path.exists():
    with open(upload_path, 'r') as f:
        lines = f.readlines()

    # Basic processing without pandas
    header = lines[0].strip().split(',')
    data_lines = lines[1:]

    result = {{
        "rows": len(data_lines),
        "columns": len(header),
        "file_size": len(''.join(lines)),
        "first_line": header,
        "sample_data": data_lines[0].strip().split(',') if data_lines else []
    }}
    print(f"Processing result: {{result}}")
else:
    print("File not found in uploads directory")
"""

        start_time = time.time()
        response = api_client.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            data=processing_code,
            headers={"Content-Type": "text/plain"},
            timeout=TestConfig.REQUEST_TIMEOUT,
        )
        processing_time = time.time() - start_time

        # Then - Processing should succeed within time limits
        assert (
            processing_time < TestConfig.MAX_FILE_PROCESSING_TIME
        ), f"Processing took {processing_time:.2f}s"
        assert response.status_code == 200

        result = response.json()
        assert (
            result.get("success") is True
        ), f"Processing failed: {result.get('error', 'Unknown error')}"
        assert "Processing result:" in result["data"]["stdout"]

    def test_given_multiple_files_when_uploaded_then_listing_performs_efficiently(
        self, api_client: requests.Session, uploaded_files_cleanup
    ):
        """
        Given: Multiple uploaded files
        When: File listing is requested multiple times
        Then: Listing operations complete quickly
        """
        files_to_create = 5
        uploaded_filenames = []

        # Given - Upload multiple files
        for i in range(files_to_create):
            content = f"id,value\n{i},100\n{i+1},200\n"
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                with open(tmp_path, "rb") as fh:
                    response = api_client.post(
                        f"{TestConfig.BASE_URL}/api/upload",
                        files={"file": (f"file_{i}.csv", fh, "text/csv")},
                        timeout=30,
                    )

                assert response.status_code == 200
                upload_data = response.json()
                uploaded_filename = upload_data["data"]["file"]["storedFilename"]
                uploaded_filenames.append(uploaded_filename)
                uploaded_files_cleanup(uploaded_filename)

            finally:
                os.unlink(tmp_path)

        # When - List files multiple times to test performance
        for iteration in range(3):
            start_time = time.time()
            response = api_client.get(
                f"{TestConfig.BASE_URL}/api/uploaded-files", timeout=10
            )
            list_time = time.time() - start_time

            # Then - Listing should be fast
            assert (
                list_time < TestConfig.MAX_LIST_TIME
            ), f"File listing took {list_time:.2f}s in iteration {iteration}"
            assert response.status_code == 200

            files_data = response.json()
            assert files_data.get("success") is True

            # Verify all uploaded files are listed
            listed_filenames = [f["filename"] for f in files_data["data"]["files"]]
            for uploaded_filename in uploaded_filenames:
                assert (
                    uploaded_filename in listed_filenames
                ), f"File {uploaded_filename} not found in listing"


class TestConcurrentRequestPerformance:
    """Test performance under concurrent request load."""

    def test_given_multiple_execution_requests_when_sent_sequentially_then_all_complete_efficiently(
        self, api_client: requests.Session
    ):
        """
        Given: Multiple simple Python execution requests
        When: Requests are sent sequentially
        Then: All requests complete within reasonable total time
        """
        # Given
        test_cases = [
            ("print(1 + 1)", "2"),
            ("print(2 * 3)", "6"),
            ("print('hello world')", "hello world"),
            ("print(len([1, 2, 3, 4, 5]))", "5"),
            ("print(type({'key': 'value'}).__name__)", "dict"),
        ]

        start_time = time.time()
        responses = []

        # When
        for code, expected_output in test_cases:
            response = api_client.post(
                f"{TestConfig.BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            responses.append((response, expected_output))

        total_time = time.time() - start_time

        # Then - All should succeed within reasonable time
        assert (
            total_time < TestConfig.MAX_SEQUENTIAL_REQUESTS_TIME
        ), f"All requests took {total_time:.2f}s"

        for response, expected_output in responses:
            assert response.status_code == 200
            result = response.json()
            assert (
                result.get("success") is True
            ), f"Request failed: {result.get('error', 'Unknown error')}"
            assert (
                expected_output in result["data"]["stdout"]
            ), f"Expected '{expected_output}' in output"


class TestResourceCleanupPerformance:
    """Test that errors don't cause resource leaks and cleanup is efficient."""

    def test_given_error_in_execution_when_system_recovers_then_performance_unaffected(
        self, api_client: requests.Session, temp_csv_file: str, uploaded_files_cleanup
    ):
        """
        Given: An uploaded file and some variables
        When: An error occurs during execution
        Then: System recovers and maintains performance
        """
        # Given - Upload a file and set variables
        with open(temp_csv_file, "rb") as fh:
            response = api_client.post(
                f"{TestConfig.BASE_URL}/api/upload",
                files={"file": ("cleanup_test.csv", fh, "text/csv")},
                timeout=30,
            )
        assert response.status_code == 200
        upload_data = response.json()
        uploaded_filename = upload_data["data"]["file"]["storedFilename"]
        uploaded_files_cleanup(uploaded_filename)

        # Set some variables
        response = api_client.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            data="test_var = 'should_be_cleaned'",
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        assert response.status_code == 200

        # When - Cause an error
        response = api_client.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            data="print(undefined_variable_xyz)",
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is False  # Should fail gracefully

        # Then - Verify system still works normally after error
        response = api_client.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            data="print(2 + 2)",
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        assert "4" in result["data"]["stdout"]

        # Verify file still accessible after error by checking upload listing
        response = api_client.get(
            f"{TestConfig.BASE_URL}/api/uploaded-files", timeout=10
        )
        assert response.status_code == 200
        files_data = response.json()
        assert files_data.get("success") is True
        listed_filenames = [f["filename"] for f in files_data["data"]["files"]]
        assert uploaded_filename in listed_filenames


# ===== Markers for Test Organization =====

# Mark slow tests
pytestmark = pytest.mark.slow
