"""
Package Management Test Suite

This module contains comprehensive Behavior-Driven Development (BDD) style tests
for package management functionality in the Pyodide Express Server.

The tests verify:
- Package listing and metadata retrieval
- Package installation workflows
- Package dependency resolution
- Package usage validation through code execution
- API contract compliance for all endpoints

Test Structure:
- All tests follow Given/When/Then BDD patterns
- Only uses /api/execute-raw endpoint for Python execution
- All Python code uses pathlib for cross-platform compatibility
- Comprehensive error handling and edge case coverage
- Full API response contract validation

Requirements Compliance:
1. ✅ Pytest framework with BDD style test names
2. ✅ Parameterized configuration using fixtures
3. ✅ Only /api/execute-raw endpoint usage
4. ✅ Cross-platform portable Python code with pathlib
5. ✅ Comprehensive docstrings with examples
6. ✅ Full API contract validation
7. ✅ No hardcoded globals - all via fixtures/constants
8. ✅ Extensive test coverage for edge cases
"""

import pytest
import requests

from conftest import (
    Config,
    validate_api_contract,
    execute_python_code,
)


# Test markers for organization
pytestmark = [pytest.mark.api, pytest.mark.integration]


class TestPackageListingBehavior:
    """
    Test suite for package listing and metadata retrieval behaviors.
    
    Covers scenarios around getting installed packages, validating
    response structure, and ensuring data accuracy.
    """
    
    def test_given_server_running_when_request_packages_then_returns_valid_structure(
        self, server_ready
    ):
        """
        Test that packages endpoint returns proper API contract structure.
        
        Given: Pyodide server is running and healthy
        When: GET /api/packages request is made
        Then: Response follows API contract with valid package data structure
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Response Format:
        {
            "success": true,
            "data": {
                "python_version": "3.11.x",
                "installed_packages": ["package1", "package2"],
                "total_packages": 42,
                "loaded_modules": ["sys", "os", ...]
            },
            "error": null,
            "meta": {"timestamp": "2025-09-15T..."}
        }
        
        Validates:
        - API contract compliance
        - Required fields presence
        - Correct data types
        - Non-empty package lists
        """
        # Given: Server is ready (via fixture)
        
        # When: Request packages endpoint
        response = requests.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['packages']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Response is successful and follows contract
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        validate_api_contract(result)
        
        # Validate success response structure
        assert result["success"] is True, f"Expected success=true: {result}"
        assert result["error"] is None, f"Expected error=null: {result}"
        assert result["data"] is not None, "Expected data to be non-null"
        
        data = result["data"]
        
        # Validate required fields presence
        required_fields = ["python_version", "installed_packages", "total_packages"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate data types
        assert isinstance(data["python_version"], str), "python_version must be string"
        assert isinstance(data["installed_packages"], list), "installed_packages must be list"
        assert isinstance(data["total_packages"], int), "total_packages must be integer"
        
        # Validate non-empty data
        assert data["total_packages"] > 0, "Should have some packages installed"
        assert len(data["installed_packages"]) > 0, "Should have package names"
        
        # Validate loaded_modules if present
        if "loaded_modules" in data:
            assert isinstance(data["loaded_modules"], list), "loaded_modules must be list"
    
    def test_given_packages_installed_when_check_count_then_matches_list_length(
        self, server_ready
    ):
        """
        Test that total_packages count matches installed_packages list length.
        
        Given: Server has packages installed
        When: Packages endpoint is queried
        Then: total_packages count exactly matches installed_packages list length
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Validates:
        - Data consistency between count and list
        - Accuracy of package enumeration
        """
        # Given: Server is ready with packages
        
        # When: Get packages information
        response = requests.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['packages']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Count matches list length
        result = response.json()
        validate_api_contract(result)
        
        data = result["data"]
        total_count = data["total_packages"]
        actual_count = len(data["installed_packages"])
        
        assert total_count == actual_count, (
            f"total_packages ({total_count}) doesn't match "
            f"installed_packages length ({actual_count})"
        )
    
    def test_given_packages_listed_when_check_loaded_modules_then_includes_basic_modules(
        self, server_ready
    ):
        """
        Test that loaded_modules includes expected Python standard library modules.
        
        Given: Pyodide runtime is initialized
        When: Packages are listed
        Then: loaded_modules contains basic Python standard library modules
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Validates:
        - Standard library modules are loaded
        - loaded_modules vs installed_packages distinction
        - Expected runtime environment state
        """
        # Given: Server is ready with runtime initialized
        
        # When: Get packages with modules information
        response = requests.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['packages']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Basic modules should be loaded
        result = response.json()
        validate_api_contract(result)
        
        data = result["data"]
        
        # Validate loaded_modules field exists and has data
        assert "loaded_modules" in data, "loaded_modules should be present"
        loaded_modules = data["loaded_modules"]
        assert isinstance(loaded_modules, list), "loaded_modules must be list"
        
        # Check for expected standard library modules
        expected_basic_modules = ["sys", "os", "json", "re", "time"]
        for module in expected_basic_modules:
            assert module in loaded_modules, (
                f"Basic module '{module}' should be loaded. "
                f"Available modules: {loaded_modules}"
            )
        
        # Validate distinction from installed_packages
        installed_packages = data["installed_packages"]
        assert loaded_modules != installed_packages, (
            "loaded_modules and installed_packages should be different lists"
        )


class TestPackageInstallationBehavior:
    """
    Test suite for package installation workflows and validation.
    
    Covers scenarios around installing packages, verifying installation
    success, and validating packages can be imported and used.
    """
    
    @pytest.mark.slow
    def test_given_server_ready_when_install_package_then_appears_in_packages_list(
        self, server_ready
    ):
        """
        Test complete package installation and verification workflow.
        
        Given: Server is ready and package is not installed
        When: Package installation is requested
        Then: Package appears in installed packages list
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Workflow:
        1. Install beautifulsoup4 package
        2. Verify installation success response
        3. Query packages list
        4. Confirm package appears in list
        
        Validates:
        - Package installation API contract
        - Package listing updates after installation
        - End-to-end installation workflow
        """
        # Given: Server is ready, choose test package
        test_package = "beautifulsoup4"
        
        # When: Install the package
        install_response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
            json={"package": test_package},
            timeout=Config.TIMEOUTS["package_install"],
        )
        
        # Then: Installation should succeed
        assert install_response.status_code == 200, (
            f"Package installation failed with status {install_response.status_code}"
        )
        
        install_result = install_response.json()
        validate_api_contract(install_result)
        
        assert install_result["success"] is True, (
            f"Package installation failed: {install_result}"
        )
        
        # When: Query packages list after installation
        packages_response = requests.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['packages']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Package should appear in list
        assert packages_response.status_code == 200
        packages_result = packages_response.json()
        validate_api_contract(packages_result)
        
        installed_packages = packages_result["data"]["installed_packages"]
        assert test_package in installed_packages, (
            f"'{test_package}' not found in installed packages: {installed_packages}"
        )
    
    @pytest.mark.slow
    def test_given_package_installed_when_import_in_code_then_executes_successfully(
        self, server_ready
    ):
        """
        Test that installed packages can be imported and used in Python code.
        
        Given: beautifulsoup4 package is installed
        When: Python code imports and uses the package via /api/execute-raw
        Then: Code execution succeeds with expected results
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Workflow:
        1. Ensure beautifulsoup4 is installed
        2. Execute Python code that imports and uses the package
        3. Verify successful execution and expected output
        
        Python Code Features:
        - Cross-platform pathlib usage
        - BeautifulSoup HTML parsing
        - String manipulation and validation
        
        Validates:
        - Package functionality after installation
        - /api/execute-raw endpoint contract
        - Cross-platform Python code execution
        """
        # Given: Install test package (ensure it's available)
        test_package = "beautifulsoup4"
        
        install_response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
            json={"package": test_package},
            timeout=Config.TIMEOUTS["package_install"],
        )
        
        # Handle case where package might already be installed
        assert install_response.status_code in [200, 400], (
            f"Unexpected install response: {install_response.status_code}"
        )
        
        # When: Execute code that imports and uses the package
        python_code = '''
from pathlib import Path
from bs4 import BeautifulSoup
import tempfile
import os

# Test HTML parsing functionality
html_content = "<html><body><p>Hello Beautiful World</p><div class='test'>Content</div></body></html>"
soup = BeautifulSoup(html_content, 'html.parser')

# Extract text content
paragraph_text = soup.find('p').text
div_text = soup.find('div', class_='test').text

# Create a cross-platform temporary file using pathlib
temp_dir = Path(tempfile.gettempdir())
temp_file = temp_dir / "bs4_test_output.txt"

# Write results using pathlib
with temp_file.open('w', encoding='utf-8') as f:
    f.write(f"Paragraph: {paragraph_text}\\n")
    f.write(f"Div: {div_text}\\n")

# Read back and verify
with temp_file.open('r', encoding='utf-8') as f:
    file_contents = f.read()

# Clean up
if temp_file.exists():
    temp_file.unlink()

# Return verification results
result = {
    "paragraph": paragraph_text,
    "div": div_text,
    "file_written": "Paragraph:" in file_contents and "Div:" in file_contents
}
print(f"BeautifulSoup test results: {result}")
result
'''
        
        # Then: Code should execute successfully
        execution_result = execute_python_code(
            python_code, 
            timeout=Config.TIMEOUTS["code_execution"]
        )
        
        assert execution_result["success"] is True, (
            f"Code execution failed: {execution_result}"
        )
        
        # Validate expected outputs in stdout
        stdout = execution_result["data"]["stdout"]
        assert "Hello Beautiful World" in stdout, "Expected paragraph text in output"
        assert "Content" in stdout, "Expected div text in output"
        assert "file_written': True" in stdout, "File operations should succeed"
    
    @pytest.mark.slow
    def test_given_package_with_dependencies_when_install_then_dependencies_included(
        self, server_ready
    ):
        """
        Test that package dependencies are automatically installed and listed.
        
        Given: Server is ready
        When: Package with known dependencies is installed (beautifulsoup4 -> soupsieve)
        Then: Both main package and dependencies appear in packages list
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Validates:
        - Dependency resolution during installation
        - Transitive dependency inclusion in packages list
        - Package ecosystem functionality
        """
        # Given: Server is ready, main package with known dependencies
        main_package = "beautifulsoup4"
        expected_dependency = "soupsieve"
        
        # When: Install main package (dependencies should be auto-installed)
        install_response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
            json={"package": main_package},
            timeout=Config.TIMEOUTS["package_install"],
        )
        
        # Allow for already installed case
        assert install_response.status_code in [200, 400]
        
        # When: Get packages list after installation
        packages_response = requests.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['packages']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Both main package and dependencies should be present
        assert packages_response.status_code == 200
        packages_result = packages_response.json()
        validate_api_contract(packages_result)
        
        installed_packages = packages_result["data"]["installed_packages"]
        
        # Verify main package and known dependency are present
        expected_packages = [main_package, expected_dependency]
        for pkg in expected_packages:
            assert pkg in installed_packages, (
                f"Expected package '{pkg}' not found in: {installed_packages}"
            )


class TestPackageManagementErrorHandling:
    """
    Test suite for error handling and edge cases in package management.
    
    Covers scenarios around invalid packages, network failures,
    timeout handling, and malformed requests.
    """
    
    def test_given_invalid_package_name_when_install_then_returns_error_response(
        self, server_ready
    ):
        """
        Test error handling for invalid package installation requests.
        
        Given: Server is ready
        When: Installation request with invalid/non-existent package name
        Then: Returns proper error response following API contract
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Validates:
        - Error response API contract compliance
        - Graceful handling of invalid package names
        - Appropriate HTTP status codes for errors
        """
        # Given: Server is ready, invalid package name
        invalid_package = "definitely-not-a-real-package-name-12345"
        
        # When: Attempt to install invalid package
        install_response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
            json={"package": invalid_package},
            timeout=Config.TIMEOUTS["api_request"],
        )
        
        # Then: Should return error response with proper contract
        # Could be 400 (bad request) or 500 (installation failure)
        assert install_response.status_code >= 400, (
            f"Expected error status code, got {install_response.status_code}"
        )
        
        install_result = install_response.json()
        validate_api_contract(install_result)
        
        # Validate error response structure
        assert install_result["success"] is False, "Expected success=false for invalid package"
        assert install_result["data"] is None, "Expected data=null for error response"
        assert install_result["error"] is not None, "Expected error message for failed installation"
        assert isinstance(install_result["error"], str), "Error message should be string"
        assert len(install_result["error"]) > 0, "Error message should be non-empty"
    
    def test_given_malformed_request_when_install_package_then_returns_validation_error(
        self, server_ready
    ):
        """
        Test request validation for package installation endpoint.
        
        Given: Server is ready
        When: Package installation request with malformed/missing data
        Then: Returns validation error with proper API contract
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Validates:
        - Request validation middleware
        - Proper error responses for malformed requests
        - API contract compliance for validation errors
        """
        # Given: Server is ready
        test_cases = [
            # Missing package field
            {},
            # Empty package name
            {"package": ""},
            # Non-string package name
            {"package": 123},
            # None package name
            {"package": None},
        ]
        
        for test_case in test_cases:
            # When: Send malformed request
            install_response = requests.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
                json=test_case,
                timeout=Config.TIMEOUTS["api_request"],
            )
            
            # Then: Should return validation error
            assert install_response.status_code == 400, (
                f"Expected 400 for malformed request {test_case}, "
                f"got {install_response.status_code}"
            )
            
            install_result = install_response.json()
            validate_api_contract(install_result)
            
            assert install_result["success"] is False, (
                f"Expected success=false for malformed request {test_case}"
            )
            assert install_result["data"] is None, "Expected data=null for validation error"
            assert install_result["error"] is not None, "Expected error message"


class TestPackageCodeExecutionIntegration:
    """
    Test suite for integration between package management and code execution.
    
    Covers scenarios around using installed packages in Python code,
    cross-platform compatibility, and complex package interactions.
    """
    
    @pytest.mark.slow
    def test_given_multiple_packages_when_execute_complex_code_then_works_cross_platform(
        self, server_ready
    ):
        """
        Test complex Python code execution using multiple installed packages.
        
        Given: Multiple packages are installed (beautifulsoup4, potentially others)
        When: Execute complex Python code using pathlib and multiple packages
        Then: Code executes successfully with expected results
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Complex Code Features:
        - Multiple package imports
        - Cross-platform file operations with pathlib
        - String processing and validation
        - Temporary file creation and cleanup
        - Data structure manipulation
        
        Validates:
        - Multi-package interaction
        - Cross-platform code execution
        - /api/execute-raw endpoint with complex code
        - Pathlib usage for all file operations
        """
        # Given: Ensure beautifulsoup4 is available
        test_package = "beautifulsoup4"
        install_response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
            json={"package": test_package},
            timeout=Config.TIMEOUTS["package_install"],
        )
        
        # When: Execute complex cross-platform code
        complex_python_code = '''
from pathlib import Path
from bs4 import BeautifulSoup
import tempfile
import json
import re
import sys
import os
from datetime import datetime

# Cross-platform temporary directory handling
temp_base = Path(tempfile.gettempdir())
test_dir = temp_base / "pyodide_package_test"
test_dir.mkdir(exist_ok=True)

# Create test HTML with various elements
html_content = """
<html>
<head><title>Package Test</title></head>
<body>
    <div class="header">Welcome to Package Testing</div>
    <ul class="items">
        <li data-id="1">Item One</li>
        <li data-id="2">Item Two</li>
        <li data-id="3">Item Three</li>
    </ul>
    <p class="description">This is a test of <strong>multiple packages</strong> working together.</p>
</body>
</html>
"""

# Parse HTML with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Extract various elements
title = soup.find('title').text
header = soup.find('div', class_='header').text
items = [li.text for li in soup.find_all('li')]
description = soup.find('p', class_='description').get_text(strip=True)

# Use regex for text processing
item_pattern = r'Item (\\w+)'
item_numbers = [re.search(item_pattern, item).group(1) for item in items if re.search(item_pattern, item)]

# Create results data structure
results = {
    "timestamp": datetime.now().isoformat(),
    "platform_info": {
        "python_version": sys.version,
        "temp_dir": str(temp_base),
        "test_dir_exists": test_dir.exists(),
    },
    "parsing_results": {
        "title": title,
        "header": header,
        "items": items,
        "description": description,
        "item_numbers": item_numbers,
    },
    "packages_used": ["pathlib", "beautifulsoup4", "tempfile", "json", "re", "sys", "os", "datetime"]
}

# Write results to JSON file using pathlib
results_file = test_dir / "test_results.json"
with results_file.open('w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Read back and verify
with results_file.open('r', encoding='utf-8') as f:
    read_back = json.load(f)

# Verify data integrity
data_matches = (
    read_back["parsing_results"]["title"] == title and
    len(read_back["parsing_results"]["items"]) == 3 and
    "multiple packages" in read_back["parsing_results"]["description"]
)

# Clean up test files
try:
    results_file.unlink()
    test_dir.rmdir()
except:
    pass  # Ignore cleanup errors

# Output results for validation
print(f"Complex package integration test completed successfully!")
print(f"Parsed {len(items)} items from HTML")
print(f"Found item numbers: {item_numbers}")
print(f"Data integrity check: {data_matches}")

# Return final results
{
    "success": True,
    "items_parsed": len(items),
    "data_integrity": data_matches,
    "packages_tested": len(results["packages_used"]),
    "cross_platform": temp_base.exists()
}
'''
        
        # Then: Complex code should execute successfully
        execution_result = execute_python_code(
            complex_python_code,
            timeout=Config.TIMEOUTS["code_execution"]
        )
        
        assert execution_result["success"] is True, (
            f"Complex code execution failed: {execution_result}"
        )
        
        # Validate expected outputs
        stdout = execution_result["data"]["stdout"]
        
        # Check for key success indicators
        assert "Complex package integration test completed successfully!" in stdout
        assert "Parsed 3 items from HTML" in stdout
        assert "Found item numbers: ['One', 'Two', 'Three']" in stdout
        assert "Data integrity check: True" in stdout
        
        # Verify that the result object indicates success
        # The final dictionary should be in stdout
        assert "'success': True" in stdout
        assert "'items_parsed': 3" in stdout
        assert "'data_integrity': True" in stdout
        assert "'cross_platform': True" in stdout


# Pytest configuration for this module
def pytest_configure(config):
    """Configure pytest markers specific to package management tests."""
    config.addinivalue_line(
        "markers", "package: marks tests as package management related"
    )
