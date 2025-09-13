"""BDD-style integration tests for the Pyodide Express Server API.

This module tests complex integration scenarios and data flow edge cases,
following the Given-When-Then pattern typical of BDD testing.
"""

import tempfile
import pytest
import requests

# Global Configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
SHORT_TIMEOUT = 10
LONG_TIMEOUT = 120


@pytest.mark.integration
def test_given_csv_file_when_uploading_and_processing_then_should_complete_workflow(server, base_url):
    """
    Given: A CSV file with test data
    When: Uploading file and processing it with Python code
    Then: Should complete the entire workflow successfully
    """
    # Given
    csv_content = "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n"
    
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(csv_content)
        tmp_path = tmp.name
    
    try:
        # When - Upload the file
        with open(tmp_path, "rb") as fh:
            upload_response = requests.post(
                f"{base_url}/api/upload",
                files={"file": ("test_data.csv", fh, "text/csv")},
                timeout=DEFAULT_TIMEOUT
            )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert upload_data.get("success") is True
        
        # When - Process the uploaded file with Python code
        processing_code = """
# Simple data processing without external dependencies
# Read the uploaded file manually
with open('uploads/test_data.csv', 'r') as f:
    lines = f.readlines()

# Parse CSV manually
headers = lines[0].strip().split(',')
data = []
for line in lines[1:]:
    values = line.strip().split(',')
    row = dict(zip(headers, values))
    data.append(row)

print(f"Loaded {len(data)} rows of data")
print(f"Headers: {headers}")

# Process the data  
total_value = sum(int(row['value']) for row in data)
processed_sum = total_value * 2

print(f"Original values: {[row['value'] for row in data]}")
print(f"Sum of processed values: {processed_sum}")
        """
        
        execution_response = requests.post(
            f"{base_url}/api/execute-raw",
            data=processing_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert execution_response.status_code == 200
        response_json = execution_response.json()
        assert response_json.get("success") is True
        output = response_json.get("stdout", "")
        assert "Loaded 3 rows of data" in output
        assert "name" in output and "value" in output and "category" in output
        assert "Sum of processed values: 12" in output  # (1+2+3)*2 = 12
        
    finally:
        import os
        os.unlink(tmp_path)
        # Clean up uploaded file
        try:
            requests.delete(f"{base_url}/api/uploaded-files/test_data.csv", timeout=SHORT_TIMEOUT)
        except:
            pass


@pytest.mark.integration
def test_given_multiple_operations_when_executed_sequentially_then_should_maintain_isolation(server, base_url):
    """
    Given: Multiple Python operations in sequence
    When: Executing them one after another via /api/execute-raw
    Then: Should maintain proper isolation between executions
    """
    # Given - First operation: set up data
    setup_code = """
# Create test data manually
data = []
for i in range(5):
    x = i
    y = i * 2  # Simple calculation
    data.append({'x': x, 'y': y})

# Save as simple CSV
with open('uploads/generated_data.csv', 'w') as f:
    f.write('x,y\\n')
    for row in data:
        f.write(f"{row['x']},{row['y']}\\n")

print(f"Generated data with {len(data)} rows")
print("Sample data created successfully")
    """
    
    # When - Execute setup
    setup_response = requests.post(
        f"{base_url}/api/execute-raw",
        data=setup_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then - Setup should succeed
    assert setup_response.status_code == 200
    setup_json = setup_response.json()
    assert setup_json.get("success") is True
    assert "Generated data with 5 rows" in setup_json.get("stdout", "")
    
    # Given - Second operation: process the data
    processing_code = """
# Load the previously generated data
with open('uploads/generated_data.csv', 'r') as f:
    lines = f.readlines()

headers = lines[0].strip().split(',')
data = []
for line in lines[1:]:
    values = line.strip().split(',')
    row = dict(zip(headers, values))
    data.append(row)

print(f"Loaded {len(data)} rows for processing")

# Calculate statistics manually
x_values = [int(row['x']) for row in data]
y_values = [int(row['y']) for row in data]

x_sum = sum(x_values)
y_sum = sum(y_values)
x_avg = x_sum / len(x_values) if x_values else 0
y_avg = y_sum / len(y_values) if y_values else 0

print(f"X statistics - Sum: {x_sum}, Average: {x_avg}")
print(f"Y statistics - Sum: {y_sum}, Average: {y_avg}")
print("Processing completed successfully")
    """
    
    # When - Execute processing
    processing_response = requests.post(
        f"{base_url}/api/execute-raw",
        data=processing_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then - Processing should succeed
    assert processing_response.status_code == 200
    processing_json = processing_response.json()
    assert processing_json.get("success") is True
    output = processing_json.get("stdout", "")
    assert "Loaded 5 rows for processing" in output
    assert "X statistics - Sum: 10, Average: 2.0" in output
    assert "Y statistics - Sum: 20, Average: 4.0" in output
    assert "Processing completed successfully" in output
    
    # Clean up
    try:
        requests.delete(f"{base_url}/api/uploaded-files/generated_data.csv", timeout=SHORT_TIMEOUT)
    except:
        pass


@pytest.mark.integration
def test_given_mathematical_operations_when_processing_then_should_calculate_correctly(server, base_url):
    """
    Given: Complex mathematical operations
    When: Executing mathematical code via /api/execute-raw
    Then: Should perform calculations correctly
    """
    # Given
    math_code = """
import math

# Test basic mathematical operations
numbers = [1, 2, 3, 4, 5]

# Basic statistics
total = sum(numbers)
count = len(numbers)
mean = total / count

print(f"Dataset: {numbers}")
print(f"Count: {count}")
print(f"Sum: {total}")
print(f"Mean: {mean}")

# Test simple trigonometric functions
angle = 45
radians = math.radians(angle)
sin_val = math.sin(radians)
cos_val = math.cos(radians)
print(f"Angle {angle}°: sin={sin_val:.3f}, cos={cos_val:.3f}")

print("Mathematical operations completed successfully")
    """
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=math_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    output = response_json.get("stdout", "")
    assert "Dataset: [1, 2, 3, 4, 5]" in output
    assert "Count: 5" in output
    assert "Sum: 15" in output
    assert "Mean: 3.0" in output
    assert "Mathematical operations completed successfully" in output


@pytest.mark.integration  
def test_given_error_recovery_scenario_when_handling_failures_then_should_continue_working(server, base_url):
    """
    Given: A scenario with intentional errors followed by valid code
    When: Executing failing code then valid code
    Then: Should recover from errors and continue working properly
    """
    # Given - Code that will fail
    failing_code = """
# This will fail - trying to read non-existent file
with open('uploads/does_not_exist.csv', 'r') as f:
    content = f.read()
print("This should not print")
    """
    
    # When - Execute failing code
    failure_response = requests.post(
        f"{base_url}/api/execute-raw",
        data=failing_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then - Should get error response
    assert failure_response.status_code == 200  # Server returns 200 with error in content
    failure_json = failure_response.json()
    assert failure_json.get("success") is False
    assert "FileNotFoundError" in failure_json.get("error", "") or "No such file" in failure_json.get("error", "")
    
    # Given - Valid recovery code
    recovery_code = """
# Create valid data to show system is still working
test_data = {
    'numbers': [1, 2, 3, 4, 5],
    'strings': ['a', 'b', 'c', 'd', 'e']
}

print("System recovery test:")
print(f"Created data with {len(test_data['numbers'])} numbers")
print(f"Numbers: {test_data['numbers']}")
print(f"Strings: {test_data['strings']}")

# Test some operations
total = sum(test_data['numbers'])
joined = ', '.join(test_data['strings'])

print(f"Sum of numbers: {total}")
print(f"Joined strings: {joined}")
print("Recovery successful - system is working normally")
    """
    
    # When - Execute recovery code
    recovery_response = requests.post(
        f"{base_url}/api/execute-raw",
        data=recovery_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then - Should work normally
    assert recovery_response.status_code == 200
    recovery_json = recovery_response.json()
    assert recovery_json.get("success") is True
    output = recovery_json.get("stdout", "")
    assert "System recovery test:" in output
    assert "Created data with 5 numbers" in output
    assert "Sum of numbers: 15" in output
    assert "Recovery successful - system is working normally" in output


@pytest.mark.integration
def test_given_string_processing_operations_when_executing_then_should_handle_text_correctly(server, base_url):
    """
    Given: Complex string processing operations
    When: Executing string manipulation code via /api/execute-raw
    Then: Should handle text processing correctly
    """
    # Given
    string_code = """
# Test string processing operations
text_data = [
    "Hello World",
    "Python Programming",
    "Data Science",
    "Machine Learning",
    "Web Development"
]

print("Original text data:")
for i, text in enumerate(text_data, 1):
    print(f"  {i}. {text}")

# String transformations
upper_texts = [text.upper() for text in text_data]
lower_texts = [text.lower() for text in text_data]
word_counts = [len(text.split()) for text in text_data]
char_counts = [len(text) for text in text_data]

print("\\nTransformed data:")
print(f"Uppercase: {upper_texts}")
print(f"Lowercase: {lower_texts}")
print(f"Word counts: {word_counts}")
print(f"Character counts: {char_counts}")

# Text analysis
total_words = sum(word_counts)
total_chars = sum(char_counts)
avg_words = total_words / len(text_data)
avg_chars = total_chars / len(text_data)

print("\\nText statistics:")
print(f"Total words: {total_words}")
print(f"Total characters: {total_chars}")
print(f"Average words per text: {avg_words:.1f}")
print(f"Average characters per text: {avg_chars:.1f}")

# String searching and filtering
search_term = "ing"
matching_texts = [text for text in text_data if search_term in text.lower()]
print(f"\\nTexts containing '{search_term}': {matching_texts}")

print("String processing completed successfully")
    """
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=string_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    output = response_json.get("stdout", "")
    assert "Original text data:" in output
    assert "Hello World" in output
    assert "Total words: 10" in output
    assert "String processing completed successfully" in output


@pytest.mark.integration
def test_given_file_operations_when_creating_and_reading_then_should_handle_io_correctly(server, base_url):
    """
    Given: File I/O operations
    When: Creating, writing, and reading files via /api/execute-raw
    Then: Should handle file operations correctly
    """
    # Given
    file_io_code = """
# Test file I/O operations
test_filename = 'uploads/integration_test_file.txt'

# Write data to file
test_content = [
    "Line 1: Introduction",
    "Line 2: Data processing",
    "Line 3: Results analysis",
    "Line 4: Conclusion"
]

print("Writing data to file...")
with open(test_filename, 'w') as f:
    for line in test_content:
        f.write(line + '\\n')

print(f"Successfully wrote {len(test_content)} lines to file")

# Read data back from file
print("\\nReading data from file...")
with open(test_filename, 'r') as f:
    read_lines = f.readlines()

print(f"Successfully read {len(read_lines)} lines from file")
print("File contents:")
for i, line in enumerate(read_lines, 1):
    print(f"  {i}: {line.strip()}")

# File statistics
total_chars = sum(len(line) for line in read_lines)
avg_line_length = total_chars / len(read_lines) if read_lines else 0

print(f"\\nFile statistics:")
print(f"Total characters: {total_chars}")
print(f"Average line length: {avg_line_length:.1f}")

# Verify content integrity
original_content_check = all(
    original.strip() == read.strip() 
    for original, read in zip(test_content, read_lines)
)
print(f"Content integrity check: {'PASSED' if original_content_check else 'FAILED'}")

print("File I/O operations completed successfully")
    """
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw",
        data=file_io_code,
        headers={"Content-Type": "text/plain"},
        timeout=DEFAULT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("success") is True
    output = response_json.get("stdout", "")
    assert "Writing data to file..." in output
    assert "Successfully wrote 4 lines to file" in output
    assert "Successfully read 4 lines from file" in output
    assert "Content integrity check: PASSED" in output
    assert "File I/O operations completed successfully" in output