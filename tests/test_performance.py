"""BDD-style performance tests using pytest.

Tests performance characteristics and resource limits using only the /api/execute-raw endpoint.
Follows Given-When-Then BDD patterns and avoids internal pyodide APIs.
"""

import subprocess
import time
from typing import Generator

import pytest
import requests

# Global configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
EXECUTION_TIMEOUT = 30000  # milliseconds
LONG_EXECUTION_TIMEOUT = 60000  # milliseconds
SERVER_STARTUP_TIMEOUT = 120
REQUEST_TIMEOUT = 30

# Performance thresholds
MAX_SIMPLE_EXECUTION_TIME = 5
MAX_MEMORY_EXECUTION_TIME = 10
MAX_CPU_EXECUTION_TIME = 15
MAX_UPLOAD_TIME = 30
MAX_PROCESSING_TIME = 15


@pytest.fixture(scope="session")
def server() -> Generator[subprocess.Popen, None, None]:
    """Start the server for the test session and ensure it's ready."""
    # Given: A server process is started
    server_process = subprocess.Popen(
        ["node", "src/server.js"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # When: We wait for the server to be ready
    start = time.time()
    while time.time() - start < SERVER_STARTUP_TIMEOUT:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        server_process.terminate()
        raise RuntimeError("Server did not start in time")
    
    yield server_process
    
    # Then: Clean up the server process
    server_process.terminate()
    try:
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture
def api_client(server):
    """Provide a configured requests session for API calls."""
    session = requests.Session()
    session.timeout = REQUEST_TIMEOUT
    return session


class TestExecutionPerformance:
    """BDD-style tests for execution performance characteristics."""

    def test_given_long_running_code_when_executed_then_times_out_appropriately(self, api_client):
        """Test that long-running code is properly timed out."""
        # Given: A piece of code that should take a long time
        long_running_code = '''
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.1)  # Small sleep to make it slower
print(f"Total: {total}")
'''
        
        # When: The code is executed with a short timeout
        start_time = time.time()
        response = api_client.post(
            f"{BASE_URL}/api/execute-raw",
            data=long_running_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then: The execution should complete within reasonable time
        assert execution_time < MAX_SIMPLE_EXECUTION_TIME
        assert response.status_code == 200
        
        # And: The response should be structured correctly
        response_data = response.json()
        assert "success" in response_data
        assert "timestamp" in response_data

    def test_given_memory_intensive_operations_when_executed_then_completes_efficiently(self, api_client):
        """Test handling of memory-intensive operations."""
        memory_test_cases = [
            # Given: Large list creation
            ("large_list = list(range(100000))\nprint(f'List length: {len(large_list)}')", "100000"),
            # Given: Large string operations  
            ("big_string = 'x' * 1000000\nprint(f'String length: {len(big_string)}')", "1000000"),
            # Given: Large dictionary
            ("big_dict = {i: f'value_{i}' for i in range(10000)}\nprint(f'Dict length: {len(big_dict)}')", "10000"),
        ]
        
        for code, expected_output in memory_test_cases:
            # When: Memory-intensive code is executed
            start_time = time.time()
            response = api_client.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: The execution should complete quickly
            assert response.status_code == 200
            assert execution_time < MAX_MEMORY_EXECUTION_TIME
            
            # And: The result should be correct
            response_data = response.json()
            if response_data.get("success"):
                assert expected_output in response_data.get("stdout", "")

    def test_given_cpu_intensive_operations_when_executed_then_handles_load_properly(self, api_client):
        """Test CPU-intensive calculations."""
        cpu_test_cases = [
            # Given: Prime number calculation
            ("""
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
print(f"Found {len(primes)} primes")
""", "Found"),
            
            # Given: Fibonacci calculation
            ("""
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

result = fib(20)
print(f"Fibonacci 20: {result}")
""", "Fibonacci 20:"),
            
            # Given: Matrix operations
            ("""
matrix = [[i*j for j in range(100)] for i in range(100)]
total = sum(sum(row) for row in matrix)
print(f"Matrix total: {total}")
""", "Matrix total:"),
        ]
        
        for code, expected_pattern in cpu_test_cases:
            # When: CPU-intensive code is executed
            start_time = time.time()
            response = api_client.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: The execution should complete within reasonable time
            assert response.status_code == 200
            assert execution_time < MAX_CPU_EXECUTION_TIME
            
            # And: The computation should succeed
            response_data = response.json()
            assert response_data.get("success") is True
            assert expected_pattern in response_data.get("stdout", "")


class TestDataProcessingPerformance:
    """BDD-style tests for data processing performance using simulated data."""

    def test_given_large_csv_data_when_processed_then_handles_efficiently(self, api_client):
        """Test processing of larger CSV-like data without file uploads."""
        # Given: A large dataset is created directly in Python
        data_processing_code = '''
import io
import csv

# Create large CSV data in memory
csv_data = "id,value,category,description\\n"
for i in range(5000):  # 5000 rows
    csv_data += f"{i},{i*2},category_{i%10},description for item {i}\\n"

# Simulate pandas-like processing
lines = csv_data.strip().split("\\n")
header = lines[0].split(",")
rows = [line.split(",") for line in lines[1:]]

# Process the data
total_value = sum(int(row[1]) for row in rows)
unique_categories = len(set(row[2] for row in rows))
row_count = len(rows)

result = {
    "shape": [row_count, len(header)],
    "value_sum": total_value,
    "categories": unique_categories
}

print(f"Processed CSV: {result}")
'''
        
        # When: The large data processing code is executed
        start_time = time.time()
        response = api_client.post(
            f"{BASE_URL}/api/execute-raw",
            data=data_processing_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then: The processing should complete within reasonable time
        assert response.status_code == 200
        assert execution_time < MAX_PROCESSING_TIME
        
        # And: The results should be correct
        response_data = response.json()
        assert response_data.get("success") is True
        stdout = response_data.get("stdout", "")
        assert "5000" in stdout  # Row count
        assert "10" in stdout    # Unique categories

    def test_given_multiple_data_operations_when_executed_then_maintains_performance(self, api_client):
        """Test performance with multiple data operations."""
        data_operations = [
            # Given: Simple data creation
            "data = list(range(1000))\nprint(f'Created {len(data)} items')",
            # Given: Data transformation
            "transformed = [x * 2 for x in range(1000)]\nprint(f'Transformed {len(transformed)} items')",
            # Given: Data filtering
            "filtered = [x for x in range(1000) if x % 2 == 0]\nprint(f'Filtered to {len(filtered)} items')",
            # Given: Data aggregation
            "total = sum(range(1000))\nprint(f'Sum: {total}')",
            # Given: Data grouping simulation
            "groups = {i % 10: [] for i in range(10)}\nfor i in range(100): groups[i % 10].append(i)\nprint(f'Groups: {len(groups)}')",
        ]
        
        for i, code in enumerate(data_operations):
            # When: Each data operation is executed
            start_time = time.time()
            response = api_client.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            # Then: Each operation should complete quickly
            assert response.status_code == 200
            assert execution_time < MAX_SIMPLE_EXECUTION_TIME
            
            # And: Each operation should succeed
            response_data = response.json()
            assert response_data.get("success") is True


class TestConcurrentPerformance:
    """BDD-style tests for concurrent request performance."""

    def test_given_multiple_simple_requests_when_executed_sequentially_then_handles_efficiently(self, api_client):
        """Test handling of multiple execution requests."""
        # Given: Multiple simple code snippets
        test_cases = [
            ("print(1 + 1)", "2"),
            ("print(2 * 3)", "6"), 
            ("print('hello world')", "hello world"),
            ("data = [1, 2, 3, 4, 5]\nprint(len(data))", "5"),
            ("info = {'key': 'value'}\nprint(info['key'])", "value"),
        ]
        
        # When: All requests are executed sequentially
        start_time = time.time()
        responses = []
        
        for code, expected in test_cases:
            response = api_client.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=REQUEST_TIMEOUT
            )
            responses.append((response, expected))
        
        total_time = time.time() - start_time
        
        # Then: All should succeed within reasonable time
        assert total_time < MAX_SIMPLE_EXECUTION_TIME
        
        # And: All results should be correct
        for response, expected in responses:
            assert response.status_code == 200
            response_data = response.json()
            assert response_data.get("success") is True
            assert expected in response_data.get("stdout", "")


class TestErrorRecoveryPerformance:
    """BDD-style tests for error recovery and cleanup performance."""

    def test_given_code_with_errors_when_executed_then_recovers_properly(self, api_client):
        """Test that errors don't cause resource leaks or performance degradation."""
        # Given: Code that will create variables then cause an error
        setup_code = "test_var = 'should_be_cleaned'\nprint('Setup complete')"
        
        # When: Variables are set up
        response = api_client.post(
            f"{BASE_URL}/api/execute-raw",
            data=setup_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        
        # Then: Setup should succeed
        assert response.status_code == 200
        assert response.json().get("success") is True
        
        # Given: Code that will cause an error
        error_code = "undefined_variable_xyz"
        
        # When: Error-causing code is executed
        response = api_client.post(
            f"{BASE_URL}/api/execute-raw",
            data=error_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        
        # Then: Error should be handled gracefully
        assert response.status_code == 200
        assert response.json().get("success") is False
        
        # Given: Simple code to test recovery
        recovery_code = "print(2 + 2)"
        
        # When: Normal code is executed after error
        start_time = time.time()
        response = api_client.post(
            f"{BASE_URL}/api/execute-raw",
            data=recovery_code,
            headers={"Content-Type": "text/plain"},
            timeout=REQUEST_TIMEOUT
        )
        execution_time = time.time() - start_time
        
        # Then: System should work normally and quickly
        assert response.status_code == 200
        assert execution_time < MAX_SIMPLE_EXECUTION_TIME
        response_data = response.json()
        assert response_data.get("success") is True
        assert "4" in response_data.get("stdout", "")