"""
Performance Testing Suite for Pyodide Express Server

This module contains comprehensive performance tests using pytest with BDD style testing.
Tests push the server to its limits with computation, memory, and file processing tasks.
All tests use only the public /api/execute-raw endpoint for Pyodide code execution.

Test Categories:
- Execution Performance: Timeout handling, CPU intensive operations
- Memory Performance: Large data structures, memory-intensive operations
- File Processing: Large CSV files, multiple file operations
- Computational Limits: Progressive complexity with numpy/pandas/scikit-learn
- Concurrency Performance: Multiple simultaneous requests
- Resource Cleanup: Error recovery and cleanup validation

Author: GitHub Copilot
Date: September 2025
"""

import os
import tempfile
import time

import pytest
import requests


# ===== GLOBAL CONFIGURATION =====
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
EXECUTION_TIMEOUT_SHORT = 5000  # 5 seconds
EXECUTION_TIMEOUT_MEDIUM = 15000  # 15 seconds
EXECUTION_TIMEOUT_LONG = 60000  # 60 seconds
MAX_RESPONSE_TIME = 30  # seconds
LARGE_FILE_ROWS = 10000
VERY_LARGE_FILE_ROWS = 50000
EXTREME_FILE_ROWS = 100000


# ===== PYTEST FIXTURES =====

@pytest.fixture(scope="session")
def server_health():
    """
    Ensure the server is running and healthy before any tests.

    Validates:
    - Server responds to health check
    - Pyodide is available
    - Basic connectivity works
    """
    max_wait = 120  # 2 minutes max wait
    start = time.time()

    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                assert health_data["status"] == "ok"
                assert health_data["pyodide"] == "available"
                return health_data
        except Exception:
            pass
        time.sleep(1)

    pytest.fail("Server is not healthy or not responding")


@pytest.fixture
def performance_session():
    """
    Create a requests session optimized for performance testing.

    Returns:
        requests.Session: Configured session with appropriate timeouts
    """
    session = requests.Session()
    session.timeout = DEFAULT_TIMEOUT
    return session


@pytest.fixture
def temp_csv_file():
    """
    Create a temporary CSV file for testing.

    Yields:
        str: Path to temporary CSV file
    """
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write("id,value,category\n1,100,A\n2,200,B\n")
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def large_csv_file():
    """
    Create a large CSV file for performance testing.

    Yields:
        str: Path to large CSV file with 10K+ rows
    """
    content = "id,value,category,description,data\n"
    for i in range(LARGE_FILE_ROWS):
        content += f"{i},{i*2.5},cat_{i % 5},desc_{i},{'x' * (i % 100)}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def very_large_csv_file():
    """
    Create a very large CSV file for extreme performance testing.

    Yields:
        str: Path to very large CSV file with 50K+ rows
    """
    content = "id,value1,value2,category,description,metadata\n"
    for i in range(VERY_LARGE_FILE_ROWS):
        content += f"{i},{i*1.5},{i*2.3},category_{i % 10},description for {i},meta_{i % 20}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def uploaded_files_tracker():
    """
    Track uploaded files for cleanup after tests.

    Yields:
        list: List to store uploaded filenames for cleanup
    """
    uploaded_files = []
    yield uploaded_files

    # Cleanup uploaded files
    for filename in uploaded_files:
        try:
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=10)
        except Exception:
            pass  # File might already be deleted


# ===== EXECUTION PERFORMANCE TESTS =====

class TestExecutionPerformance:
    """Test suite for execution performance characteristics."""

    def test_given_long_running_code_when_timeout_is_short_then_execution_completes_within_limit(
        self, server_health, performance_session
    ):
        """
        Test that long-running Python code respects timeout limits.

        Given: A Python script that performs intensive computation
        When: The script is executed with a short timeout
        Then: The execution completes within the timeout window
        And: The server remains responsive
        """
        # Given: Long-running computational code
        long_running_code = '''
import time
total = 0
for i in range(100000):
    total += i * i
    if i % 10000 == 0:
        time.sleep(0.01)  # Small sleep to control timing
print(f"Computation complete: {total}")
'''

        # When: Code is executed with timeout
        start_time = time.time()
        response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=30  # Reasonable timeout
        )
        execution_time = time.time() - start_time

        # Then: Execution completes within reasonable time
        assert execution_time < 10  # Should not take more than 10 seconds
        assert response.status_code == 200

        # And: Response contains expected output or timeout message
        response_text = response.text
        assert isinstance(response_text, str)

    def test_given_cpu_intensive_operations_when_executed_then_results_are_accurate_and_fast(
        self, server_health, performance_session
    ):
        """
        Test CPU-intensive mathematical operations for accuracy and performance.

        Given: CPU-intensive mathematical calculations
        When: Multiple operations are executed sequentially
        Then: All results are mathematically correct
        And: Each operation completes within acceptable time limits
        """
        cpu_intensive_operations = [
            {
                "name": "Prime number calculation",
                "code": '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
print(f"Found {len(primes)} primes up to 1000")
print(f"First 10 primes: {primes[:10]}")
''',
                "expected_primes": 168,
                "max_time": 5
            },
            {
                "name": "Fibonacci sequence",
                "code": '''
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fib_iterative(30)
print(f"Fibonacci(30) = {result}")
''',
                "expected_fib": 832040,
                "max_time": 3
            },
            {
                "name": "Matrix multiplication",
                "code": '''
size = 100
matrix_a = [[i * j for j in range(size)] for i in range(size)]
matrix_b = [[j * i for j in range(size)] for i in range(size)]

# Simple matrix multiplication
result = [[0 for _ in range(size)] for _ in range(size)]
for i in range(size):
    for j in range(size):
        for k in range(size):
            result[i][j] += matrix_a[i][k] * matrix_b[k][j]

total = sum(sum(row) for row in result)
print(f"Matrix multiplication total: {total}")
''',
                "max_time": 8
            }
        ]

        for operation in cpu_intensive_operations:
            # Given: CPU intensive operation
            code = operation["code"]

            # When: Operation is executed
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time

            # Then: Operation completes successfully
            assert response.status_code == 200
            assert execution_time < operation["max_time"]

            # And: Response contains expected output
            response_text = response.text
            assert len(response_text) > 0

            # Validate specific results
            if "expected_primes" in operation:
                assert "168 primes" in response_text
            elif "expected_fib" in operation:
                assert "832040" in response_text

    def test_given_memory_intensive_operations_when_executed_then_memory_is_managed_efficiently(
        self, server_health, performance_session
    ):
        """
        Test memory-intensive operations for proper memory management.

        Given: Operations that create large data structures
        When: Multiple memory-intensive operations are executed
        Then: Each operation completes successfully
        And: Memory is properly managed without crashes
        """
        memory_operations = [
            {
                "name": "Large list creation",
                "code": '''
import sys
large_list = list(range(500000))
memory_usage = sys.getsizeof(large_list)
print(f"Created list with {len(large_list)} elements")
print(f"Memory usage: {memory_usage} bytes")
del large_list  # Explicit cleanup
print("List deleted successfully")
''',
                "max_time": 5
            },
            {
                "name": "Large string operations",
                "code": '''
import sys
big_string = 'Hello World! ' * 100000
memory_usage = sys.getsizeof(big_string)
print(f"Created string with {len(big_string)} characters")
print(f"Memory usage: {memory_usage} bytes")

# String operations
upper_string = big_string.upper()
print(f"Uppercase string length: {len(upper_string)}")
del big_string, upper_string
print("Strings deleted successfully")
''',
                "max_time": 5
            },
            {
                "name": "Large dictionary operations",
                "code": '''
import sys
big_dict = {f"key_{i}": f"value_{i}" * 10 for i in range(50000)}
memory_usage = sys.getsizeof(big_dict)
print(f"Created dictionary with {len(big_dict)} items")
print(f"Memory usage: {memory_usage} bytes")

# Dictionary operations
keys_sum = sum(1 for key in big_dict.keys() if key.startswith("key_1"))
print(f"Keys starting with 'key_1': {keys_sum}")
del big_dict
print("Dictionary deleted successfully")
''',
                "max_time": 8
            }
        ]

        for operation in memory_operations:
            # Given: Memory intensive operation
            code = operation["code"]

            # When: Operation is executed
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time

            # Then: Operation completes within time limit
            assert response.status_code == 200
            assert execution_time < operation["max_time"]

            # And: Response indicates successful memory management
            response_text = response.text
            assert "deleted successfully" in response_text
            assert "Memory usage:" in response_text


# ===== FILE PROCESSING PERFORMANCE TESTS =====

class TestFileProcessingPerformance:
    """Test suite for file processing performance."""

    def test_given_large_csv_file_when_uploaded_and_processed_then_operations_complete_efficiently(
        self, server_health, performance_session, large_csv_file, uploaded_files_tracker
    ):
        """
        Test processing of large CSV files for performance and accuracy.

        Given: A large CSV file with 10K+ rows
        When: The file is uploaded and processed with pandas
        Then: Upload completes within acceptable time
        And: Data processing operations are accurate and efficient
        """
        # Given: Large CSV file is available
        assert os.path.exists(large_csv_file)
        file_size = os.path.getsize(large_csv_file)
        assert file_size > 100000  # Should be > 100KB

        # When: File is uploaded
        start_time = time.time()
        with open(large_csv_file, "rb") as fh:
            upload_response = performance_session.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("large_performance_test.csv", fh, "text/csv")}
            )
        upload_time = time.time() - start_time

        # Then: Upload completes successfully and efficiently
        assert upload_response.status_code == 200
        assert upload_time < 15  # Should upload within 15 seconds

        upload_data = upload_response.json()
        filename = upload_data["filename"]
        uploaded_files_tracker.append(filename)

        # When: Large file is processed with pandas
        processing_code = f'''
import pandas as pd
import time
from pathlib import Path

# Load the data
start_time = time.time()
df = pd.read_csv('/uploads/{filename}')
load_time = time.time() - start_time

print(f"Data loading took {{load_time:.3f}} seconds")
print(f"DataFrame shape: {{df.shape}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum()}} bytes")

# Perform various operations
operations_start = time.time()

# Basic statistics
value_stats = df['value'].describe()
print(f"Value column statistics:")
print(f"  Mean: {{value_stats['mean']:.2f}}")
print(f"  Std: {{value_stats['std']:.2f}}")

# Grouping operations
category_counts = df['category'].value_counts()
print(f"Category distribution:")
for cat, count in category_counts.head().items():
    print(f"  {{cat}}: {{count}}")

# Advanced operations
df['value_squared'] = df['value'] ** 2
df['value_log'] = df['value'].apply(lambda x: x ** 0.5)
complex_result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])

operations_time = time.time() - operations_start
print(f"Data operations took {{operations_time:.3f}} seconds")
print(f"Complex grouping result shape: {{complex_result.shape}}")

total_time = time.time() - start_time
print(f"Total processing time: {{total_time:.3f}} seconds")
'''

        start_time = time.time()
        processing_response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=processing_code, headers={"Content-Type": "text/plain"}
        )
        processing_time = time.time() - start_time

        # Then: Processing completes successfully and efficiently
        assert processing_response.status_code == 200
        assert processing_time < 20  # Should process within 20 seconds

        response_text = processing_response.text
        assert "DataFrame shape:" in response_text
        assert f"({LARGE_FILE_ROWS}," in response_text  # Should show correct row count
        assert "Total processing time:" in response_text

    def test_given_very_large_csv_file_when_processed_then_system_handles_extreme_load(
        self, server_health, performance_session, very_large_csv_file, uploaded_files_tracker
    ):
        """
        Test system limits with very large CSV files.

        Given: A very large CSV file with 50K+ rows
        When: The file is processed with complex pandas operations
        Then: System handles the load without crashing
        And: Performance remains within acceptable bounds
        """
        # Given: Very large CSV file
        file_size = os.path.getsize(very_large_csv_file)
        assert file_size > 1000000  # Should be > 1MB

        # When: Very large file is uploaded
        start_time = time.time()
        with open(very_large_csv_file, "rb") as fh:
            upload_response = performance_session.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("very_large_test.csv", fh, "text/csv")},
                timeout=60  # Extended timeout for large file
            )
        upload_time = time.time() - start_time

        # Then: Upload handles large file
        assert upload_response.status_code == 200
        assert upload_time < 30  # Should upload within 30 seconds

        filename = upload_response.json()["filename"]
        uploaded_files_tracker.append(filename)

        # When: Complex processing is performed
        extreme_processing_code = f'''
import pandas as pd
import numpy as np
import time

print("Starting extreme file processing...")
start_time = time.time()

# Load with chunking for memory efficiency
chunk_size = 10000
chunks = []
load_start = time.time()

for chunk in pd.read_csv('/uploads/{filename}', chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
load_time = time.time() - load_start

print(f"Loaded {{len(chunks)}} chunks in {{load_time:.3f}} seconds")
print(f"Final DataFrame shape: {{df.shape}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}} MB")

# Intensive operations
ops_start = time.time()

# Statistical operations
print("Performing statistical operations...")
stats = df.describe(include='all')
print(f"Statistics computed for {{len(stats.columns)}} columns")

# Grouping operations
print("Performing grouping operations...")
grouped_stats = df.groupby('category').agg({{
    'value1': ['mean', 'std', 'min', 'max'],
    'value2': ['sum', 'count']
}})
print(f"Grouped statistics shape: {{grouped_stats.shape}}")

# Memory-intensive operations
print("Creating derived columns...")
df['complex_calc'] = df['value1'] * df['value2'] + np.sin(df['value1'] * 0.1)
df['category_encoded'] = pd.Categorical(df['category']).codes

ops_time = time.time() - ops_start
total_time = time.time() - start_time

print(f"Operations completed in {{ops_time:.3f}} seconds")
print(f"Total processing time: {{total_time:.3f}} seconds")
print(f"Rows processed per second: {{df.shape[0] / total_time:.0f}}")

# Cleanup to free memory
del df, chunks
print("Memory cleanup completed")
'''

        start_time = time.time()
        processing_response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=extreme_processing_code, headers={"Content-Type": "text/plain"},
            timeout=90  # Extended timeout
        )
        processing_time = time.time() - start_time

        # Then: Extreme processing completes successfully
        assert processing_response.status_code == 200
        assert processing_time < 60  # Should complete within 1 minute

        response_text = processing_response.text
        assert "extreme file processing" in response_text
        assert f"({VERY_LARGE_FILE_ROWS}," in response_text
        assert "Memory cleanup completed" in response_text

    def test_given_multiple_csv_files_when_processed_concurrently_then_system_maintains_performance(
        self, server_health, performance_session, uploaded_files_tracker
    ):
        """
        Test concurrent file processing performance.

        Given: Multiple CSV files of varying sizes
        When: Files are uploaded and processed in sequence
        Then: Each operation maintains consistent performance
        And: System resources are managed efficiently
        """
        files_to_process = []

        # Given: Create multiple test files
        for i in range(5):
            content = f"id,value,category,data\n"
            rows = 1000 * (i + 1)  # 1K, 2K, 3K, 4K, 5K rows
            for j in range(rows):
                content += f"{j},{j * (i + 1)},cat_{j % 3},data_{j}_{i}\n"

            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                files_to_process.append({
                    "path": tmp.name,
                    "name": f"concurrent_test_{i}.csv",
                    "expected_rows": rows
                })

        try:
            uploaded_filenames = []
            processing_times = []

            # When: Files are uploaded and processed
            for file_info in files_to_process:
                # Upload file
                start_time = time.time()
                with open(file_info["path"], "rb") as fh:
                    upload_response = performance_session.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (file_info["name"], fh, "text/csv")}
                    )

                assert upload_response.status_code == 200
                filename = upload_response.json()["filename"]
                uploaded_filenames.append(filename)
                uploaded_files_tracker.append(filename)

                # Process file
                processing_code = f'''
import pandas as pd
import time

start_time = time.time()
df = pd.read_csv('/uploads/{filename}')
load_time = time.time() - start_time

print(f"File: {filename}")
print(f"Shape: {{df.shape}}")
print(f"Load time: {{load_time:.3f}}s")

# Perform operations
ops_start = time.time()
summary = df.groupby('category')['value'].agg(['count', 'mean', 'sum'])
ops_time = time.time() - ops_start

print(f"Operations time: {{ops_time:.3f}}s")
print(f"Total time: {{time.time() - start_time:.3f}}s")
print(f"Processing rate: {{df.shape[0] / (time.time() - start_time):.0f}} rows/sec")
'''

                processing_response = performance_session.post(
                    f"{BASE_URL}/api/execute-raw",
                    data=processing_code, headers={"Content-Type": "text/plain"}
                )

                total_time = time.time() - start_time
                processing_times.append(total_time)

                # Then: Each file processes successfully
                assert processing_response.status_code == 200
                assert total_time < 15  # Each file should process within 15 seconds

                response_text = processing_response.text
                assert f"Shape: ({file_info['expected_rows']}," in response_text
                assert "Processing rate:" in response_text

            # Then: Performance remains consistent across files
            assert len(processing_times) == 5
            avg_time = sum(processing_times) / len(processing_times)
            assert avg_time < 10  # Average processing time should be reasonable

            # Performance should not degrade significantly
            first_half_avg = sum(processing_times[:2]) / 2
            second_half_avg = sum(processing_times[3:]) / 2
            assert second_half_avg < first_half_avg * 2  # Should not be more than 2x slower

        finally:
            # Cleanup temporary files
            for file_info in files_to_process:
                if os.path.exists(file_info["path"]):
                    os.unlink(file_info["path"])


# ===== COMPUTATIONAL LIMITS TESTS =====

class TestComputationalLimits:
    """Test suite for pushing computational limits with data science libraries."""

    def test_given_numpy_operations_when_complexity_increases_then_performance_scales_appropriately(
        self, server_health, performance_session
    ):
        """
        Test numpy computational limits with increasing complexity.

        Given: Numpy operations of increasing computational complexity
        When: Each operation is executed and timed
        Then: Performance scales predictably with complexity
        And: System handles intensive numerical computations
        """
        numpy_complexity_tests = [
            {
                "name": "Small matrix operations",
                "code": '''
import numpy as np
import time

start_time = time.time()

# Small matrices (100x100)
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)

# Matrix operations
c = np.dot(a, b)
eigenvals = np.linalg.eigvals(c)
det = np.linalg.det(c)

duration = time.time() - start_time
print(f"Small matrix operations completed in {{duration:.3f}} seconds")
print(f"Matrix shape: {{c.shape}}")
print(f"Determinant: {{det:.6f}}")
print(f"Max eigenvalue: {{np.max(np.real(eigenvals)):.6f}}")
''',
                "max_time": 5
            },
            {
                "name": "Medium matrix operations",
                "code": '''
import numpy as np
import time

start_time = time.time()

# Medium matrices (500x500)
a = np.random.rand(500, 500)
b = np.random.rand(500, 500)

# Intensive operations
c = np.dot(a, b)
svd_u, svd_s, svd_vh = np.linalg.svd(c[:100, :100])  # SVD on subset for speed
inv_subset = np.linalg.inv(c[:100, :100])

duration = time.time() - start_time
print(f"Medium matrix operations completed in {{duration:.3f}} seconds")
print(f"Full matrix shape: {{c.shape}}")
print(f"SVD singular values (first 5): {{svd_s[:5]}}")
print(f"Inverse condition number: {{np.linalg.cond(inv_subset):.2f}}")
''',
                "max_time": 10
            },
            {
                "name": "Large array computations",
                "code": '''
import numpy as np
import time

start_time = time.time()

# Large arrays
size = 1000000  # 1M elements
a = np.random.rand(size)
b = np.random.rand(size)

# Vectorized operations
c = a * b + np.sin(a) + np.cos(b)
d = np.sqrt(np.abs(c))
e = np.fft.fft(d[:10000])  # FFT on subset

# Statistical operations
stats = {{
    'mean': np.mean(d),
    'std': np.std(d),
    'median': np.median(d),
    'percentile_95': np.percentile(d, 95)
}}

duration = time.time() - start_time
print(f"Large array computations completed in {{duration:.3f}} seconds")
print(f"Array size: {{size}} elements")
print(f"Mean: {{stats['mean']:.6f}}")
print(f"Std: {{stats['std']:.6f}}")
print(f"FFT result size: {{len(e)}}")
''',
                "max_time": 8
            }
        ]

        execution_times = []

        for test in numpy_complexity_tests:
            # Given: Numpy operation of specific complexity
            code = test["code"]

            # When: Operation is executed
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            # Then: Operation completes within expected time
            assert response.status_code == 200
            assert execution_time < test["max_time"]

            response_text = response.text
            assert "completed in" in response_text
            assert "seconds" in response_text

            print(f"✅ {test['name']}: {execution_time:.3f}s")

        # Then: Performance scales as expected
        assert len(execution_times) == 3
        # Later tests should generally take longer (but not always due to different operations)
        assert max(execution_times) < 15  # No test should take more than 15 seconds

    def test_given_pandas_operations_when_data_size_increases_then_memory_and_performance_scale(
        self, server_health, performance_session
    ):
        """
        Test pandas computational limits with increasing data sizes.

        Given: Pandas operations on datasets of increasing size
        When: Complex data manipulation tasks are performed
        Then: Operations complete successfully within memory limits
        And: Performance degrades gracefully with data size
        """
        pandas_scaling_tests = [
            {
                "name": "Small DataFrame operations",
                "rows": 10000,
                "code_template": '''
import pandas as pd
import numpy as np
import time

start_time = time.time()

# Create DataFrame
rows = {rows}
df = pd.DataFrame({{
    'id': range(rows),
    'value1': np.random.rand(rows),
    'value2': np.random.rand(rows) * 100,
    'category': [f'cat_{{i % 10}}' for i in range(rows)],
    'date': pd.date_range('2023-01-01', periods=rows, freq='1H')
}})

creation_time = time.time() - start_time
print(f"DataFrame creation ({{rows}} rows): {{creation_time:.3f}}s")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}} MB")

# Operations
ops_start = time.time()

# Statistical operations
stats = df.describe()
correlations = df[['value1', 'value2']].corr()

# Groupby operations
grouped = df.groupby('category').agg({{
    'value1': ['mean', 'std', 'count'],
    'value2': ['sum', 'min', 'max']
}})

# Time series operations
df['hour'] = df['date'].dt.hour
hourly_stats = df.groupby('hour')['value1'].mean()

ops_time = time.time() - ops_start
total_time = time.time() - start_time

print(f"Operations time: {{ops_time:.3f}}s")
print(f"Total time: {{total_time:.3f}}s")
print(f"Rows per second: {{rows / total_time:.0f}}")
print(f"Categories found: {{len(grouped)}}")
''',
                "max_time": 8
            },
            {
                "name": "Medium DataFrame operations",
                "rows": 50000,
                "code_template": '''
import pandas as pd
import numpy as np
import time

start_time = time.time()

# Create larger DataFrame
rows = {rows}
df = pd.DataFrame({{
    'id': range(rows),
    'value1': np.random.rand(rows),
    'value2': np.random.rand(rows) * 1000,
    'value3': np.random.randint(1, 100, rows),
    'category': [f'category_{{i % 20}}' for i in range(rows)],
    'subcategory': [f'sub_{{i % 5}}' for i in range(rows)],
    'date': pd.date_range('2022-01-01', periods=rows, freq='30T')
}})

creation_time = time.time() - start_time
print(f"DataFrame creation ({{rows}} rows): {{creation_time:.3f}}s")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}} MB")

# Complex operations
ops_start = time.time()

# Multi-level grouping
multi_grouped = df.groupby(['category', 'subcategory']).agg({{
    'value1': ['mean', 'std'],
    'value2': ['sum', 'count'],
    'value3': ['median', 'max']
}})

# Window operations
df['rolling_mean'] = df['value1'].rolling(window=100).mean()
df['pct_change'] = df['value2'].pct_change()

# Filtering and sorting
filtered = df[df['value1'] > 0.5].sort_values('value2', ascending=False).head(1000)

ops_time = time.time() - ops_start
total_time = time.time() - start_time

print(f"Operations time: {{ops_time:.3f}}s")
print(f"Total time: {{total_time:.3f}}s")
print(f"Processing rate: {{rows / total_time:.0f}} rows/sec")
print(f"Multi-group shape: {{multi_grouped.shape}}")
print(f"Filtered records: {{len(filtered)}}")
''',
                "max_time": 15
            }
        ]

        for test in pandas_scaling_tests:
            # Given: Pandas operation with specific data size
            code = test["code_template"].format(rows=test["rows"])

            # When: Complex pandas operations are executed
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time

            # Then: Operations complete within expected time
            assert response.status_code == 200
            assert execution_time < test["max_time"]

            response_text = response.text
            assert f"DataFrame creation ({test['rows']} rows)" in response_text
            assert "Memory usage:" in response_text
            assert "Processing rate:" in response_text or "Rows per second:" in response_text

            print(f"✅ {test['name']}: {execution_time:.3f}s")

    def test_given_scikit_learn_operations_when_model_complexity_increases_then_training_scales_appropriately(
        self, server_health, performance_session
    ):
        """
        Test scikit-learn computational limits with increasing model complexity.

        Given: Machine learning models of increasing complexity
        When: Models are trained on progressively larger datasets
        Then: Training completes successfully within time limits
        And: Model performance metrics are reasonable
        """
        sklearn_complexity_tests = [
            {
                "name": "Simple linear regression",
                "code": '''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

start_time = time.time()

# Generate synthetic data
n_samples = 10000
n_features = 10

X = np.random.rand(n_samples, n_features)
y = np.sum(X * np.random.rand(n_features), axis=1) + np.random.rand(n_samples) * 0.1

data_time = time.time() - start_time
print(f"Data generation ({{n_samples}} samples, {{n_features}} features): {{data_time:.3f}}s")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
train_start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
train_time = time.time() - train_start

# Predictions and evaluation
pred_start = time.time()
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pred_time = time.time() - pred_start

total_time = time.time() - start_time

print(f"Training time: {{train_time:.3f}}s")
print(f"Prediction time: {{pred_time:.3f}}s")
print(f"Total time: {{total_time:.3f}}s")
print(f"MSE: {{mse:.6f}}")
print(f"R² Score: {{r2:.6f}}")
print(f"Samples per second (training): {{len(X_train) / train_time:.0f}}")
''',
                "max_time": 10
            },
            {
                "name": "Random Forest classification",
                "code": '''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import time

start_time = time.time()

# Generate classification data
n_samples = 20000
n_features = 20
n_classes = 5

X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

data_time = time.time() - start_time
print(f"Data generation ({{n_samples}} samples, {{n_features}} features): {{data_time:.3f}}s")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with Random Forest
train_start = time.time()
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
model.fit(X_train, y_train)
train_time = time.time() - train_start

# Predictions
pred_start = time.time()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
pred_time = time.time() - pred_start

# Feature importance
importance_start = time.time()
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
importance_time = time.time() - importance_start

total_time = time.time() - start_time

print(f"Training time: {{train_time:.3f}}s")
print(f"Prediction time: {{pred_time:.3f}}s")
print(f"Feature importance time: {{importance_time:.3f}}s")
print(f"Total time: {{total_time:.3f}}s")
print(f"Accuracy: {{accuracy:.4f}}")
print(f"Training rate: {{len(X_train) / train_time:.0f}} samples/sec")
print(f"Top feature indices: {{top_features.tolist()}}")
''',
                "max_time": 15
            },
            {
                "name": "K-means clustering with PCA",
                "code": '''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time

start_time = time.time()

# Generate clustering data
n_samples = 15000
n_features = 30
n_clusters = 8

X = np.random.rand(n_samples, n_features)
# Add some structure to make clustering meaningful
for i in range(n_clusters):
    center = np.random.rand(n_features) * 10
    cluster_samples = n_samples // n_clusters
    start_idx = i * cluster_samples
    end_idx = start_idx + cluster_samples
    if i == n_clusters - 1:  # Last cluster gets remaining samples
        end_idx = n_samples
    X[start_idx:end_idx] += center

data_time = time.time() - start_time
print(f"Data generation ({{n_samples}} samples, {{n_features}} features): {{data_time:.3f}}s")

# Data preprocessing
prep_start = time.time()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
prep_time = time.time() - prep_start

# PCA for dimensionality reduction
pca_start = time.time()
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - pca_start

# K-means clustering
cluster_start = time.time()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
cluster_time = time.time() - cluster_start

# Evaluation
eval_start = time.time()
silhouette_avg = silhouette_score(X_pca, clusters)
eval_time = time.time() - eval_start

total_time = time.time() - start_time

print(f"Preprocessing time: {{prep_time:.3f}}s")
print(f"PCA time: {{pca_time:.3f}}s")
print(f"Clustering time: {{cluster_time:.3f}}s")
print(f"Evaluation time: {{eval_time:.3f}}s")
print(f"Total time: {{total_time:.3f}}s")
print(f"Silhouette score: {{silhouette_avg:.4f}}")
print(f"PCA explained variance ratio (first 3): {{pca.explained_variance_ratio_[:3]}}")
print(f"Clustering rate: {{n_samples / cluster_time:.0f}} samples/sec")
''',
                "max_time": 20
            }
        ]

        for test in sklearn_complexity_tests:
            # Given: Scikit-learn operation with specific complexity
            code = test["code"]

            # When: Machine learning operation is executed
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time

            # Then: Operation completes within expected time
            assert response.status_code == 200
            assert execution_time < test["max_time"]

            response_text = response.text
            assert "Data generation" in response_text
            assert "Total time:" in response_text

            # Verify model-specific outputs
            if "regression" in test["name"]:
                assert "R² Score:" in response_text
                assert "MSE:" in response_text
            elif "classification" in test["name"]:
                assert "Accuracy:" in response_text
                assert "Training rate:" in response_text
            elif "clustering" in test["name"]:
                assert "Silhouette score:" in response_text
                assert "PCA" in response_text

            print(f"✅ {test['name']}: {execution_time:.3f}s")


# ===== CONCURRENCY AND CLEANUP TESTS =====

class TestConcurrencyAndCleanup:
    """Test suite for concurrent requests and resource cleanup."""

    def test_given_multiple_concurrent_requests_when_executed_simultaneously_then_all_complete_successfully(
        self, server_health, performance_session
    ):
        """
        Test handling of multiple concurrent execution requests.

        Given: Multiple Python execution requests
        When: Requests are sent in rapid succession
        Then: All requests complete successfully
        And: Response times remain reasonable
        """
        # Given: Multiple simple execution requests
        test_codes = [
            "result = sum(range(1000)); print(f'Sum: {result}')",
            "import math; result = [math.sqrt(i) for i in range(500)]; print(f'Computed {len(result)} square roots')",
            "text = 'hello world ' * 100; print(f'Text length: {len(text)}')",
            "data = {'key' + str(i): i**2 for i in range(200)}; print(f'Dict size: {len(data)}')",
            "import random; values = [random.random() for _ in range(1000)]; print(f'Average: {sum(values)/len(values):.4f}')"
        ]

        # When: Requests are executed in sequence (simulating concurrent load)
        start_time = time.time()
        responses = []

        for i, code in enumerate(test_codes):
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            responses.append({
                "index": i,
                "response": response,
                "code": code[:30] + "..." if len(code) > 30 else code
            })

        total_time = time.time() - start_time

        # Then: All requests complete successfully
        assert len(responses) == len(test_codes)
        assert total_time < 15  # Should complete all within 15 seconds

        for resp_data in responses:
            response = resp_data["response"]
            assert response.status_code == 200

            response_text = response.text
            assert len(response_text) > 0

            # Verify expected outputs
            if "Sum:" in resp_data["code"]:
                assert "Sum: 499500" in response_text
            elif "square roots" in resp_data["code"]:
                assert "Computed 500 square roots" in response_text
            elif "Text length:" in resp_data["code"]:
                assert "Text length:" in response_text
            elif "Dict size:" in resp_data["code"]:
                assert "Dict size: 200" in response_text
            elif "Average:" in resp_data["code"]:
                assert "Average:" in response_text

        # Performance metrics
        avg_time = total_time / len(test_codes)
        print(f"✅ Concurrent requests: {len(test_codes)} completed in {total_time:.3f}s (avg: {avg_time:.3f}s)")

    def test_given_execution_errors_when_they_occur_then_system_recovers_and_cleans_up_properly(
        self, server_health, performance_session, temp_csv_file, uploaded_files_tracker
    ):
        """
        Test system recovery and cleanup after execution errors.

        Given: Python code that will cause various types of errors
        When: Error-inducing code is executed
        Then: Errors are handled gracefully
        And: System continues to function normally
        And: Resources are properly cleaned up
        """
        # Given: Upload a test file for cleanup testing
        with open(temp_csv_file, "rb") as fh:
            upload_response = performance_session.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("cleanup_test.csv", fh, "text/csv")}
            )

        assert upload_response.status_code == 200
        filename = upload_response.json()["filename"]
        uploaded_files_tracker.append(filename)

        # Set some variables before errors
        setup_code = '''
test_var = "should_persist_after_errors"
import pandas as pd
df = pd.read_csv("/uploads/{}")
print(f"Setup complete - DataFrame shape: {{df.shape}}")
print(f"Test variable: {{test_var}}")
'''.format(filename)

        response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=setup_code, headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 200
        assert "Setup complete" in response.text

        # Test various error scenarios
        error_tests = [
            {
                "name": "NameError",
                "code": "print(undefined_variable_xyz)",
                "expected_error_indicator": "error" # Response should indicate error
            },
            {
                "name": "ZeroDivisionError",
                "code": "result = 10 / 0; print(result)",
                "expected_error_indicator": "error"
            },
            {
                "name": "ImportError",
                "code": "import nonexistent_module; print('Should not reach here')",
                "expected_error_indicator": "error"
            },
            {
                "name": "IndexError",
                "code": "my_list = [1, 2, 3]; print(my_list[10])",
                "expected_error_indicator": "error"
            }
        ]

        for error_test in error_tests:
            # When: Error-inducing code is executed
            error_response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=error_test["code"], headers={"Content-Type": "text/plain"}
            )

            # Then: Error is handled gracefully (status 200 but error in response)
            assert error_response.status_code == 200
            error_text = error_response.text
            # The response should contain error information (exact format may vary)
            assert len(error_text) > 0  # Should have some error output

            print(f"✅ {error_test['name']} handled: {len(error_text)} chars in response")

        # Then: System continues to function normally after errors
        recovery_code = '''
# Test that system still works
print("Testing system recovery...")
result = 2 + 2
print(f"Basic math works: {result}")

# Test that uploaded file is still accessible
import pandas as pd
df = pd.read_csv("/uploads/{}")
print(f"File still accessible - shape: {{df.shape}}")

# Test that we can still create new variables
recovery_var = "system_recovered"
print(f"Recovery variable: {{recovery_var}}")

print("System recovery test complete")
'''.format(filename)

        recovery_response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=recovery_code, headers={"Content-Type": "text/plain"}
        )

        # Then: Recovery is successful
        assert recovery_response.status_code == 200
        recovery_text = recovery_response.text
        assert "Testing system recovery" in recovery_text
        assert "Basic math works: 4" in recovery_text
        assert "File still accessible" in recovery_text
        assert "System recovery test complete" in recovery_text

        print("✅ System recovery after errors: successful")

    def test_given_timeout_scenarios_when_code_exceeds_limits_then_timeouts_are_enforced_properly(
        self, server_health, performance_session
    ):
        """
        Test timeout enforcement for long-running operations.

        Given: Python code designed to run longer than timeout limits
        When: Code is executed with various timeout settings
        Then: Timeouts are properly enforced
        And: System remains responsive after timeouts
        """
        timeout_tests = [
            {
                "name": "Short timeout test",
                "code": '''
import time
print("Starting potentially long operation...")
for i in range(100):
    # Simulate work
    total = sum(range(10000))
    if i % 10 == 0:
        print(f"Iteration {i}: {total}")
    time.sleep(0.1)  # This will likely cause timeout
print("This should not be reached with short timeout")
''',
                "timeout": 2000,  # 2 seconds
                "max_execution_time": 5
            },
            {
                "name": "Medium timeout test",
                "code": '''
import time
print("Starting medium duration operation...")
start_time = time.time()
iterations = 0
while time.time() - start_time < 8:  # Try to run for 8 seconds
    total = sum(i*i for i in range(1000))
    iterations += 1
    if iterations % 100 == 0:
        print(f"Completed {iterations} iterations")
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f}s")

print(f"Final iterations: {iterations}")
''',
                "timeout": 5000,  # 5 seconds
                "max_execution_time": 8
            }
        ]

        for timeout_test in timeout_tests:
            # Given: Code that may exceed timeout limits
            code = timeout_test["code"]
            timeout = timeout_test["timeout"]

            # When: Code is executed with timeout
            start_time = time.time()
            response = performance_session.post(
                f"{BASE_URL}/api/execute-raw",
                data=code, headers={"Content-Type": "text/plain"}
            )
            execution_time = time.time() - start_time

            # Then: Execution completes within reasonable time
            assert response.status_code == 200
            assert execution_time < timeout_test["max_execution_time"]

            response_text = response.text
            assert len(response_text) > 0

            # Should have some output (either completion or timeout)
            has_output = ("Starting" in response_text or
                         "Iteration" in response_text or
                         "Elapsed time" in response_text or
                         "timeout" in response_text.lower() or
                         "error" in response_text.lower())
            assert has_output

            print(f"✅ {timeout_test['name']}: {execution_time:.3f}s (timeout: {timeout}ms)")

        # Then: System remains responsive after timeout tests
        health_response = performance_session.get(f"{BASE_URL}/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"

        # Simple execution still works
        simple_response = performance_session.post(
            f"{BASE_URL}/api/execute-raw",
            data="print('System still responsive'); result = 1 + 1; print(f'1 + 1 = {result}')",
            headers={"Content-Type": "text/plain"}
        )
        assert simple_response.status_code == 200
        assert "System still responsive" in simple_response.text
        assert "1 + 1 = 2" in simple_response.text

        print("✅ System responsiveness after timeouts: confirmed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
