"""BDD-style simplified integration tests.

Given a running Pyodide Express server
When basic integration workflows are executed using only /api/execute-raw
Then proper functionality should be maintained without internal APIs

This module focuses on core functionality using only public endpoints.
"""

import pytest

from conftest import (
    given_server_is_running,
    when_executing_python_code,
    then_response_should_be_successful,
    then_response_should_contain_text,
)


class TestBasicIntegrationWorkflows:
    """Test basic integration workflows using only execute-raw."""

    def test_given_server_running_when_basic_python_execution_then_output_captured(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When basic Python code is executed
        Then output should be captured correctly
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = when_executing_python_code(
            api_session,
            "print('Hello World')\nresult = 2 + 3\nprint(f'Result: {result}')"
        )
        
        # Then
        then_response_should_be_successful(response)
        then_response_should_contain_text(response, "Hello World")
        then_response_should_contain_text(response, "Result: 5")

    def test_given_server_running_when_file_operations_executed_then_filesystem_accessible(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When file operations are executed
        Then the filesystem should be accessible
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When - Test file operations
        response = when_executing_python_code(
            api_session,
            '''
from pathlib import Path
import json

# Test filesystem access
uploads_dir = Path('/home/pyodide/uploads')
result = {
    'uploads_exists': uploads_dir.exists(),
    'is_directory': uploads_dir.is_dir() if uploads_dir.exists() else False,
    'file_count': len(list(uploads_dir.glob('*'))) if uploads_dir.exists() else 0
}
print(json.dumps(result))
'''
        )
        
        # Then
        then_response_should_be_successful(response)
        then_response_should_contain_text(response, '"uploads_exists": true')

    def test_given_server_running_when_json_processing_executed_then_data_handled_correctly(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When JSON processing is executed
        Then data should be handled correctly
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When - Test JSON processing
        response = when_executing_python_code(
            api_session,
            '''
import json

# Create sample data
data = {
    'items': [
        {'name': 'item1', 'value': 100},
        {'name': 'item2', 'value': 200}
    ],
    'total': 300
}

# Process data
processed = {
    'item_count': len(data['items']),
    'total_value': sum(item['value'] for item in data['items']),
    'average_value': sum(item['value'] for item in data['items']) / len(data['items'])
}

print(json.dumps(processed))
'''
        )
        
        # Then
        then_response_should_be_successful(response)
        then_response_should_contain_text(response, '"item_count": 2')
        then_response_should_contain_text(response, '"total_value": 300')
        then_response_should_contain_text(response, '"average_value": 150')

    def test_given_server_running_when_error_handling_executed_then_graceful_recovery(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When error handling scenarios are executed
        Then graceful recovery should occur
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When - Test error handling
        response = when_executing_python_code(
            api_session,
            '''
import json

errors = []
successes = []

# Test 1: Valid operation
try:
    result = 5 * 10
    successes.append({'test': 'multiplication', 'result': result})
except Exception as e:
    errors.append({'test': 'multiplication', 'error': str(e)})

# Test 2: File operation that might fail
try:
    from pathlib import Path
    test_path = Path('/nonexistent/directory/file.txt')
    exists = test_path.exists()
    successes.append({'test': 'path_check', 'exists': exists})
except Exception as e:
    errors.append({'test': 'path_check', 'error': str(e)})

# Test 3: Type conversion
try:
    value = int('123')
    successes.append({'test': 'type_conversion', 'value': value})
except Exception as e:
    errors.append({'test': 'type_conversion', 'error': str(e)})

result = {
    'total_tests': len(errors) + len(successes),
    'successes': len(successes),
    'errors': len(errors),
    'success_details': successes,
    'error_details': errors
}

print(json.dumps(result))
'''
        )
        
        # Then
        then_response_should_be_successful(response)
        then_response_should_contain_text(response, '"total_tests":')
        then_response_should_contain_text(response, '"successes":')

    def test_given_server_running_when_complex_computation_executed_then_accurate_results(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When complex computations are executed
        Then accurate results should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When - Test complex computation
        response = when_executing_python_code(
            api_session,
            '''
import json
import math

# Complex computation example
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Perform computations
fib_numbers = [fibonacci(i) for i in range(10)]
primes_under_50 = [i for i in range(2, 50) if is_prime(i)]

result = {
    'fibonacci_sequence': fib_numbers,
    'fibonacci_sum': sum(fib_numbers),
    'primes_under_50': primes_under_50,
    'prime_count': len(primes_under_50),
    'largest_prime_under_50': max(primes_under_50)
}

print(json.dumps(result))
'''
        )
        
        # Then
        then_response_should_be_successful(response)
        then_response_should_contain_text(response, '"fibonacci_sequence":')
        then_response_should_contain_text(response, '"primes_under_50":')
        then_response_should_contain_text(response, '"largest_prime_under_50": 47')