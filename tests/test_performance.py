"""BDD-style performance tests for the Pyodide Express Server API.

This module tests performance characteristics and resource limits,
following the Given-When-Then pattern typical of BDD testing.
"""

import time
import tempfile
import pytest
import requests

# Global Configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
SHORT_TIMEOUT = 10
LONG_TIMEOUT = 120


@pytest.mark.performance
def test_given_long_running_code_when_executing_then_should_timeout_appropriately(server, base_url):
    """
    Given: Code that takes a long time to execute
    When: Executing via /api/execute-raw
    Then: Should timeout appropriately without hanging the server
    """
    # Given - Code that should take a long time
    long_running_code = """
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.01)  # Small delays to extend execution time
print(f"Total: {total}")
    """
    
    # When
    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=long_running_code,
        headers={"Content-Type": "text/plain"},
        timeout=SHORT_TIMEOUT  # Use short timeout to test timeout behavior
    )
    execution_time = time.time() - start_time
    
    # Then - Should either complete quickly or timeout appropriately
    assert response.status_code == 200
    response_json = response.json()
    
    # Should either succeed quickly or timeout
    if response_json.get("success"):
        assert execution_time < SHORT_TIMEOUT
    else:
        # If it fails, should be due to timeout or execution limits
        error_msg = response_json.get("error", "").lower()
        assert any(keyword in error_msg for keyword in ["timeout", "limit", "time", "exceed"])


@pytest.mark.performance
def test_given_multiple_concurrent_requests_when_executing_then_should_handle_load(server, base_url):
    """
    Given: Multiple concurrent execution requests
    When: Sending concurrent requests to /api/execute-raw
    Then: Should handle the load without errors
    """
    # Given - Simple code that can be executed quickly
    test_code = """
import math
result = sum(i * math.sqrt(i) for i in range(100))
print(f"Computation result: {result:.2f}")
    """
    
    # When - Send multiple concurrent requests
    import threading
    import queue
    
    results = queue.Queue()
    num_requests = 5  # Reasonable number for testing
    
    def make_request():
        try:
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=test_code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT
            )
            results.put((response.status_code, response.json()))
        except Exception as e:
            results.put((0, {"error": str(e)}))
    
    # Create and start threads
    threads = []
    start_time = time.time()
    
    for _ in range(num_requests):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Then - Collect and verify results
    successful_responses = 0
    all_results = []
    
    while not results.empty():
        status_code, response_data = results.get()
        all_results.append((status_code, response_data))
        if status_code == 200 and response_data.get("success"):
            successful_responses += 1
    
    # Should have processed all requests reasonably quickly
    assert len(all_results) == num_requests
    assert successful_responses >= num_requests // 2  # At least half should succeed
    assert total_time < LONG_TIMEOUT  # Should complete within reasonable time


@pytest.mark.performance
def test_given_large_output_code_when_executing_then_should_handle_output_size(server, base_url):
    """
    Given: Code that generates large output
    When: Executing via /api/execute-raw
    Then: Should handle large output appropriately
    """
    # Given - Code that generates substantial output
    large_output_code = """
# Generate substantial output
for i in range(100):
    print(f"Line {i:03d}: This is a test line with some content to make it longer - " + "x" * 50)

print("Large output generation completed")
    """
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=large_output_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    
    stdout = response_json.get("stdout", "")
    assert "Line 000:" in stdout
    assert "Line 099:" in stdout
    assert "Large output generation completed" in stdout
    
    # Verify output size is reasonable
    assert len(stdout) > 1000  # Should have substantial content
    assert len(stdout) < 100000  # But not excessively large


@pytest.mark.performance
def test_given_memory_intensive_code_when_executing_then_should_handle_memory_usage(server, base_url):
    """
    Given: Code that uses significant memory
    When: Executing via /api/execute-raw  
    Then: Should handle memory usage appropriately
    """
    # Given - Code that uses memory but stays within reasonable limits
    memory_code = """
# Create data structures that use memory
data_lists = []
for i in range(10):
    # Create lists of numbers
    numbers = list(range(1000))  # Reasonable size
    data_lists.append(numbers)

print(f"Created {len(data_lists)} lists")
print(f"Total elements: {sum(len(lst) for lst in data_lists)}")

# Simple processing
total_sum = sum(sum(lst) for lst in data_lists)
print(f"Total sum: {total_sum}")

print("Memory usage test completed")
    """
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=memory_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    
    stdout = response_json.get("stdout", "")
    assert "Created 10 lists" in stdout
    assert "Total elements: 10000" in stdout
    assert "Memory usage test completed" in stdout


@pytest.mark.performance
def test_given_computation_intensive_code_when_executing_then_should_complete_efficiently(server, base_url):
    """
    Given: Computationally intensive code
    When: Executing via /api/execute-raw
    Then: Should complete efficiently within reasonable time
    """
    # Given - Code that does significant computation but finishes reasonably quickly
    computation_code = """
import math

# Perform mathematical computations
results = []
for i in range(1000):
    # Some mathematical operations
    value = math.sqrt(i) * math.log(i + 1) + math.sin(i * 0.1)
    results.append(value)

# Statistical calculations
total = sum(results)
mean = total / len(results)
variance = sum((x - mean) ** 2 for x in results) / len(results)
std_dev = math.sqrt(variance)

print(f"Processed {len(results)} computations")
print(f"Mean: {mean:.6f}")
print(f"Standard deviation: {std_dev:.6f}")
print("Computation test completed")
    """
    
    # When
    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=computation_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    execution_time = time.time() - start_time
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    
    stdout = response_json.get("stdout", "")
    assert "Processed 1000 computations" in stdout
    assert "Mean:" in stdout
    assert "Standard deviation:" in stdout
    assert "Computation test completed" in stdout
    
    # Should complete in reasonable time
    assert execution_time < DEFAULT_TIMEOUT


@pytest.mark.performance
def test_given_file_io_operations_when_executing_then_should_handle_io_efficiently(server, base_url):
    """
    Given: File I/O intensive operations
    When: Executing via /api/execute-raw
    Then: Should handle file operations efficiently
    """
    # Given - Code that performs multiple file operations
    file_io_code = """
import os

# Create multiple test files
file_data = {}
for i in range(10):
    filename = f"uploads/perf_test_{i}.txt"
    content = f"Performance test file {i}\\n" + "Test content\\n" * 50
    
    with open(filename, 'w') as f:
        f.write(content)
    
    file_data[filename] = len(content)

print(f"Created {len(file_data)} test files")

# Read all files back
total_content_length = 0
for filename in file_data:
    with open(filename, 'r') as f:
        content = f.read()
        total_content_length += len(content)

print(f"Total content read: {total_content_length} characters")

# Clean up files
cleaned_files = 0
for filename in file_data:
    try:
        os.remove(filename)
        cleaned_files += 1
    except:
        pass

print(f"Cleaned up {cleaned_files} files")
print("File I/O test completed")
    """
    
    # When
    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=file_io_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    execution_time = time.time() - start_time
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    
    stdout = response_json.get("stdout", "")
    assert "Created 10 test files" in stdout
    assert "Total content read:" in stdout
    assert "Cleaned up" in stdout
    assert "File I/O test completed" in stdout
    
    # Should complete efficiently
    assert execution_time < DEFAULT_TIMEOUT


@pytest.mark.performance
def test_given_rapid_successive_requests_when_executing_then_should_handle_rate_limiting(server, base_url):
    """
    Given: Rapid successive execution requests
    When: Sending requests in quick succession via /api/execute-raw
    Then: Should handle requests appropriately with or without rate limiting
    """
    # Given - Simple code for rapid execution
    simple_code = """
print("Quick execution test")
result = 2 + 2
print(f"Result: {result}")
    """
    
    # When - Send requests in rapid succession
    responses = []
    start_time = time.time()
    
    for i in range(10):
        try:
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=simple_code,
                headers={"Content-Type": "text/plain"},
                timeout=SHORT_TIMEOUT
            )
            responses.append((response.status_code, response.json()))
        except Exception as e:
            responses.append((0, {"error": str(e)}))
        
        # Small delay between requests
        time.sleep(0.1)
    
    total_time = time.time() - start_time
    
    # Then
    assert len(responses) == 10
    
    # Count successful responses
    successful = sum(1 for status, _ in responses if status == 200)
    
    # Should handle most requests successfully or provide appropriate error responses
    assert successful >= 5  # At least half should succeed
    assert total_time < SHORT_TIMEOUT * 2  # Should complete reasonably quickly
    
    # Check that successful responses are valid
    for status_code, response_data in responses:
        if status_code == 200 and response_data.get("success"):
            assert "Quick execution test" in response_data.get("stdout", "")
            assert "Result: 4" in response_data.get("stdout", "")


@pytest.mark.performance 
def test_given_stress_test_scenario_when_executing_then_should_maintain_stability(server, base_url):
    """
    Given: A stress test scenario with various operations
    When: Executing complex operations via /api/execute-raw
    Then: Should maintain system stability
    """
    # Given - Code that combines multiple operations
    stress_test_code = """
import math
import time

# Multi-faceted stress test
print("Starting stress test...")

# 1. Mathematical operations
math_results = []
for i in range(500):
    result = math.sin(i) * math.cos(i) + math.sqrt(i + 1)
    math_results.append(result)

print(f"Completed {len(math_results)} mathematical operations")

# 2. String operations
text_data = []
for i in range(100):
    text = f"Stress test string {i}: " + "content " * 10
    text_data.append(text.upper())

print(f"Processed {len(text_data)} string operations")

# 3. List operations
lists = []
for i in range(50):
    numbers = list(range(i, i + 20))
    lists.append(sorted(numbers, reverse=True))

print(f"Created and sorted {len(lists)} lists")

# 4. File operation
test_file = "uploads/stress_test.txt"
with open(test_file, 'w') as f:
    f.write("Stress test data\\n" * 100)

with open(test_file, 'r') as f:
    lines = f.readlines()

print(f"File operations: wrote and read {len(lines)} lines")

# Clean up
import os
try:
    os.remove(test_file)
    print("Cleaned up test file")
except:
    print("Test file cleanup failed")

print("Stress test completed successfully")
    """
    
    # When
    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=stress_test_code,
        headers={"Content-Type": "text/plain"},
        timeout=LONG_TIMEOUT  # Allow more time for stress test
    )
    execution_time = time.time() - start_time
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    
    # Should either succeed or fail gracefully
    if response_json.get("success"):
        stdout = response_json.get("stdout", "")
        assert "Starting stress test..." in stdout
        assert "Stress test completed successfully" in stdout
        assert execution_time < LONG_TIMEOUT
    else:
        # If it fails, should be due to resource limits, which is acceptable
        error_msg = response_json.get("error", "").lower()
        assert any(keyword in error_msg for keyword in ["timeout", "limit", "resource", "memory"])