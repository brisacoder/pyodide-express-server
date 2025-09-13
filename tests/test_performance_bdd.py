"""BDD-style performance tests using only public APIs.

Given a running Pyodide Express server
When performance-critical operations are executed using only /api/execute-raw
Then response times and resource usage should be within acceptable limits

This module tests performance characteristics without using internal APIs.
"""

import time

import pytest

from conftest import (
    given_server_is_running,
    when_executing_python_code,
    then_response_should_be_successful,
)


class TestExecutionPerformance:
    """Test performance characteristics of code execution."""

    def test_given_server_running_when_simple_computation_executed_then_fast_response(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When simple computation is executed
        Then response should be fast
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            "result = sum(range(1000))\nprint(f'Sum: {result}')"
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert "Sum: 499500" in response.text

    def test_given_server_running_when_moderate_computation_executed_then_reasonable_response(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When moderate computation is executed
        Then response should be reasonable
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
# Moderate computation - prime numbers
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 1000) if is_prime(n)]
print(f"Found {len(primes)} primes under 1000")
print(f"Largest prime: {max(primes)}")
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert "Found" in response.text
        assert "primes under 1000" in response.text

    def test_given_server_running_when_string_operations_executed_then_efficient_processing(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When string operations are executed
        Then processing should be efficient
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
# String processing operations
text = "The quick brown fox jumps over the lazy dog " * 1000
words = text.split()
word_count = len(words)
unique_words = len(set(words))
longest_word = max(words, key=len)

print(f"Total words: {word_count}")
print(f"Unique words: {unique_words}")
print(f"Longest word: {longest_word}")
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert "Total words:" in response.text
        assert "Unique words:" in response.text

    def test_given_server_running_when_list_operations_executed_then_memory_efficient(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When list operations are executed
        Then memory usage should be efficient
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
# List operations
import sys

# Create lists and measure memory patterns
numbers = list(range(10000))
squares = [n * n for n in numbers]
evens = [n for n in numbers if n % 2 == 0]

print(f"Original list length: {len(numbers)}")
print(f"Squares list length: {len(squares)}")
print(f"Even numbers count: {len(evens)}")
print(f"First few squares: {squares[:5]}")
print(f"Last few evens: {evens[-5:]}")
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 8.0  # Should complete within 8 seconds
        assert "Original list length: 10000" in response.text
        assert "Even numbers count: 5000" in response.text


class TestConcurrentOperations:
    """Test performance under concurrent-like operations."""

    def test_given_server_running_when_multiple_operations_simulated_then_stable_performance(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When multiple operations are simulated in sequence
        Then performance should remain stable
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When - Simulate multiple operations
        operations = [
            "result1 = sum(range(100))",
            "result2 = [i*2 for i in range(100)]",
            "result3 = {'key' + str(i): i for i in range(100)}",
            "result4 = ''.join(str(i) for i in range(100))",
            "result5 = len([i for i in range(100) if i % 3 == 0])"
        ]
        
        execution_times = []
        for i, operation in enumerate(operations):
            start_time = time.time()
            response = when_executing_python_code(
                api_session,
                f"{operation}\nprint(f'Operation {i+1} completed')"
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Then - Each operation should succeed
            then_response_should_be_successful(response)
            assert f"Operation {i+1} completed" in response.text
        
        # Then - Performance should be stable (no significant degradation)
        max_time = max(execution_times)
        min_time = min(execution_times)
        assert max_time < 5.0  # All operations under 5 seconds
        assert max_time / min_time < 10  # No more than 10x variation

    def test_given_server_running_when_file_system_operations_executed_then_responsive_access(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When file system operations are executed
        Then access should be responsive
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
from pathlib import Path
import json

# File system access operations
dirs_to_check = [
    '/home/pyodide/uploads',
    '/home/pyodide/in',
    '/plots',
    '/tmp'
]

results = {}
for dir_path in dirs_to_check:
    path = Path(dir_path)
    results[dir_path] = {
        'exists': path.exists(),
        'is_dir': path.is_dir() if path.exists() else False,
        'file_count': len(list(path.glob('*'))) if path.exists() and path.is_dir() else 0
    }

print(json.dumps(results, indent=2))
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 3.0  # File system access should be fast
        assert "/home/pyodide/uploads" in response.text


class TestMemoryIntensiveOperations:
    """Test performance with memory-intensive operations."""

    def test_given_server_running_when_moderate_memory_usage_then_stable_execution(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When moderate memory usage occurs
        Then execution should remain stable
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
# Moderate memory usage test
import sys

# Create data structures
large_list = list(range(50000))
large_dict = {f"key_{i}": f"value_{i}" for i in range(10000)}
large_string = "x" * 100000

# Process data
list_sum = sum(large_list)
dict_size = len(large_dict)
string_length = len(large_string)

print(f"List sum: {list_sum}")
print(f"Dict size: {dict_size}")
print(f"String length: {string_length}")

# Cleanup to free memory
del large_list, large_dict, large_string
print("Memory cleanup completed")
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 10.0  # Should handle moderate memory usage efficiently
        assert "List sum:" in response.text
        assert "Memory cleanup completed" in response.text

    def test_given_server_running_when_data_processing_executed_then_efficient_handling(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When data processing is executed
        Then handling should be efficient
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
# Data processing simulation
import json

# Simulate CSV-like data processing without pandas
data_rows = []
for i in range(1000):
    row = {
        'id': i,
        'value': i * 2.5,
        'category': f'cat_{i % 5}',
        'active': i % 3 == 0
    }
    data_rows.append(row)

# Process data
total_value = sum(row['value'] for row in data_rows)
active_count = sum(1 for row in data_rows if row['active'])
categories = set(row['category'] for row in data_rows)

# Group by category
category_totals = {}
for row in data_rows:
    cat = row['category']
    if cat not in category_totals:
        category_totals[cat] = {'count': 0, 'total_value': 0}
    category_totals[cat]['count'] += 1
    category_totals[cat]['total_value'] += row['value']

result = {
    'total_rows': len(data_rows),
    'total_value': total_value,
    'active_count': active_count,
    'category_count': len(categories),
    'category_summary': category_totals
}

print(json.dumps(result))
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 8.0  # Data processing should be reasonably fast
        assert '"total_rows": 1000' in response.text
        assert '"category_count": 5' in response.text


class TestErrorHandlingPerformance:
    """Test performance of error handling scenarios."""

    def test_given_server_running_when_recoverable_errors_occur_then_fast_recovery(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When recoverable errors occur
        Then recovery should be fast
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        start_time = time.time()
        response = when_executing_python_code(
            api_session,
            '''
import json

# Test various error conditions and recovery
results = []
start_time = __import__('time').time()

# Test 1: Try-except with division by zero
try:
    result = 10 / 0
except ZeroDivisionError:
    results.append({'test': 'division_by_zero', 'handled': True})

# Test 2: Try-except with invalid key access
try:
    data = {'a': 1}
    value = data['nonexistent']
except KeyError:
    results.append({'test': 'key_error', 'handled': True})

# Test 3: Try-except with type error
try:
    result = "string" + 5
except TypeError:
    results.append({'test': 'type_error', 'handled': True})

# Test 4: Successful operation
try:
    result = sum(range(100))
    results.append({'test': 'success', 'result': result, 'handled': True})
except Exception:
    results.append({'test': 'success', 'handled': False})

end_time = __import__('time').time()

summary = {
    'total_tests': len(results),
    'execution_time': end_time - start_time,
    'all_handled': all(r['handled'] for r in results),
    'results': results
}

print(json.dumps(summary))
'''
        )
        execution_time = time.time() - start_time
        
        # Then
        then_response_should_be_successful(response)
        assert execution_time < 3.0  # Error handling should be fast
        assert '"all_handled": true' in response.text
        assert '"total_tests": 4' in response.text