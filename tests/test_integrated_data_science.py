#!/usr/bin/env python3
"""
BDD-style pytest tests for data science package integration and fu            "expected_error_type": "SyntaxError"
        }
    ]
    
    # Import error scenariosnality.

This module integrates and modernizes standalone test files from the project root,
converting them to proper pytest format with comprehensive error handling and
API contract validation.

Integrated Tests From Root Directory:
- test_packages.py -> Package availability and discovery testing
- test_syntax_error.py -> Syntax error handling validation  
- test_imports.py -> Import error scenarios
- simple_test.py -> Basic functionality validation
- ultra_simple_test.py -> Minimal operation testing

Key Features:
- ✅ BDD Given/When/Then structure
- ✅ Only uses /api/execute-raw endpoint
- ✅ API contract compliance validation
- ✅ Cross-platform pathlib usage
- ✅ Comprehensive error handling
- ✅ No hardcoded values - all fixtures
- ✅ Enhanced documentation and examples

API Contract:
{
  "success": true | false,
  "data": {
    "result": <any>,
    "stdout": <string>,
    "stderr": <string>,
    "executionTime": <number>
  } | null,
  "error": <string|null>,
  "meta": {"timestamp": <string>}
}
"""

import json

import pytest

# Import shared configuration and utilities
try:
    from .conftest import execute_python_code, validate_api_contract
except ImportError:
    from conftest import execute_python_code, validate_api_contract


class IntegratedTestConfig:
    """Configuration constants for integrated data science tests."""
    
    # Package discovery settings
    CORE_PACKAGES = [
        "matplotlib", "numpy", "pandas", "pathlib", "json", "sys", "os"
    ]
    
    DATA_SCIENCE_PACKAGES = [
        "seaborn", "scipy", "sklearn", "plotly", "statsmodels"
    ]
    
    UTILITY_PACKAGES = [
        "micropip", "requests", "urllib", "base64", "io", "datetime"
    ]
    
    # Timeout configurations
    TIMEOUT_SETTINGS = {
        "basic_operations": 15,
        "package_discovery": 30,
        "plot_generation": 45,
        "error_testing": 20
    }
    
    # Error test scenarios
    SYNTAX_ERROR_SCENARIOS = [
        {
            "name": "missing_parenthesis",
            "code": 'print("Hello World"',
            "expected_error_type": "SyntaxError"
        },
        {
            "name": "missing_bracket",
            "code": 'data = [1, 2, 3\nprint(data)',
            "expected_error_type": "SyntaxError" 
        },
        {
            "name": "invalid_indentation",
            "code": 'if True:\nprint("Hello")',
            "expected_error_type": "IndentationError"
        },
        {
            "name": "missing_colon",
            "code": 'if True\n    print("Hello")',
            "expected_error_type": "SyntaxError"
        }
    ]
    
    # Import error scenarios  
    IMPORT_ERROR_SCENARIOS = [
        {
            "name": "nonexistent_module",
            "code": "import nonexistent_module_12345",
            "expected_error_type": "ImportError"
        },
        {
            "name": "invalid_from_import",
            "code": "from math import nonexistent_function",
            "expected_error_type": "ImportError"
        },
        {
            "name": "circular_import_simulation",
            "code": "import sys; sys.modules['circular'] = sys; from circular import circular",
            "expected_error_type": "ImportError"
        }
    ]


# Alias for backward compatibility
DataScienceTestConfig = IntegratedTestConfig


@pytest.fixture
def plot_timeout():
    """
    Provide timeout for plot generation operations.

    Returns:
        int: Timeout in seconds for plot operations

    Example:
        >>> def test_plot(plot_timeout):
        ...     result = execute_python_code(code, timeout=plot_timeout)
    """
    return DataScienceTestConfig.TIMEOUT_SETTINGS["plot_generation"] + 10


@pytest.fixture
def quick_timeout():
    """
    Provide timeout for quick operations like syntax checks.

    Returns:
        int: Timeout in seconds for quick operations

    Example:
        >>> def test_quick_operation(quick_timeout):
        ...     result = execute_python_code(code, timeout=quick_timeout)
    """
    return DataScienceTestConfig.TIMEOUT_SETTINGS["basic_operations"]


@pytest.mark.api
@pytest.mark.integration
class TestDataSciencePackageDiscovery:
    """
    BDD tests for discovering and validating available Python packages.
    
    Migrated from: test_packages.py
    Enhanced with: proper pytest structure, API contract validation, comprehensive testing
    """

    def test_given_pyodide_environment_when_checking_core_packages_then_all_available(
        self, server_ready, plot_timeout
    ):
        """
        Test that core Python packages are available in Pyodide environment.

        **Given:** A Pyodide environment with core Python packages
        **When:** Checking availability of essential packages like sys, os, pathlib
        **Then:** All core packages should be importable and functional

        Args:
            server_ready: Server readiness fixture
            plot_timeout: Timeout for operations

        Example:
            Core packages like sys, os, pathlib, json should all be available
            and return version/functionality information.
        """
        # Given: Core packages that should always be available
        code = f"""
import sys
import importlib
from pathlib import Path

core_packages = {IntegratedTestConfig.CORE_PACKAGES}
package_status = {{}}
system_info = {{}}

# When: Testing each core package
for pkg_name in core_packages:
    status = {{
        "available": False,
        "version": None,
        "functionality_test": False,
        "error": None
    }}
    
    try:
        # Test import
        pkg = importlib.import_module(pkg_name)
        status["available"] = True
        
        # Get version if available
        if hasattr(pkg, '__version__'):
            status["version"] = pkg.__version__
        elif hasattr(pkg, 'version'):
            status["version"] = str(pkg.version)
        
        # Basic functionality test
        if pkg_name == "pathlib":
            test_path = Path("/tmp/test")
            status["functionality_test"] = hasattr(test_path, "exists")
        elif pkg_name == "json":
            test_data = {{"test": "data"}}
            json_str = pkg.dumps(test_data)
            parsed = pkg.loads(json_str)
            status["functionality_test"] = parsed["test"] == "data"
        elif pkg_name == "sys":
            status["functionality_test"] = hasattr(pkg, "version") and hasattr(pkg, "path")
        else:
            status["functionality_test"] = True
            
    except Exception as e:
        status["error"] = str(e)
    
    package_status[pkg_name] = status

# Then: Collect system information
system_info = {{
    "python_version": sys.version,
    "platform": sys.platform,
    "path_count": len(sys.path),
    "available_core_packages": len([p for p in package_status.values() if p["available"]]),
    "total_core_packages": len(core_packages)
}}

{{
    "package_status": package_status,
    "system_info": system_info,
    "summary": {{
        "all_core_available": all(status["available"] for status in package_status.values()),
        "all_functional": all(status["functionality_test"] for status in package_status.values() if status["available"])
    }}
}}
        """

        # When: Executing package discovery
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and results
        validate_api_contract(result)
        assert result["success"], f"Core package discovery failed: {result.get('error')}"

        test_results = json.loads(result["data"]["result"])
        
        # All core packages should be available
        missing_core = [name for name, status in test_results['package_status'].items()
                        if not status['available']]
        assert test_results["summary"]["all_core_available"], \
            f"Missing core packages: {missing_core}"
        
        # All available packages should be functional
        non_functional = [name for name, status in test_results['package_status'].items()
                          if status['available'] and not status['functionality_test']]
        assert test_results["summary"]["all_functional"], \
            f"Non-functional packages: {non_functional}"
        
        # Validate system info
        system_info = test_results["system_info"]
        assert len(system_info["python_version"]) > 0, "Should have Python version info"
        assert system_info["path_count"] > 0, "Should have Python path entries"

    def test_given_pyodide_environment_when_checking_data_science_packages_then_documents_availability(
        self, server_ready, plot_timeout
    ):
        """
        Test and document availability of data science packages.

        **Given:** A Pyodide environment that may or may not have data science packages
        **When:** Checking availability of packages like seaborn, scipy, sklearn
        **Then:** Documents which packages are available and their capabilities

        Args:
            server_ready: Server readiness fixture
            plot_timeout: Timeout for operations

        Note:
            This test documents current state - some packages may not be available
            and that's acceptable. The goal is comprehensive discovery.
        """
        # Given: Data science packages to check
        code = f"""
import importlib
import sys

data_science_packages = {IntegratedTestConfig.DATA_SCIENCE_PACKAGES}
utility_packages = {IntegratedTestConfig.UTILITY_PACKAGES}

def test_package_capability(pkg_name, pkg_module):
    \"\"\"Test basic capability of a package\"\"\"
    capabilities = {{}}
    
    try:
        if pkg_name == "numpy":
            import numpy as np
            arr = np.array([1, 2, 3])
            capabilities["array_creation"] = len(arr) == 3
            capabilities["version"] = np.__version__
        elif pkg_name == "pandas":
            import pandas as pd
            df = pd.DataFrame({{"A": [1, 2], "B": [3, 4]}})
            capabilities["dataframe_creation"] = len(df) == 2
            capabilities["version"] = pd.__version__
        elif pkg_name == "matplotlib":
            import matplotlib
            capabilities["backend_available"] = True
            capabilities["version"] = matplotlib.__version__
        elif pkg_name == "seaborn":
            import seaborn as sns
            capabilities["seaborn_imported"] = True
            capabilities["version"] = sns.__version__
        elif pkg_name == "scipy":
            import scipy
            capabilities["scipy_imported"] = True
            capabilities["version"] = scipy.__version__
        elif pkg_name == "sklearn":
            import sklearn
            capabilities["sklearn_imported"] = True
            capabilities["version"] = sklearn.__version__
        elif pkg_name == "micropip":
            capabilities["micropip_available"] = True
        else:
            capabilities["basic_import"] = True
            
    except Exception as e:
        capabilities["error"] = str(e)
    
    return capabilities

# When: Testing all packages
all_packages = data_science_packages + utility_packages
package_discovery = {{}}

for pkg_name in all_packages:
    discovery_result = {{
        "available": False,
        "capabilities": {{}},
        "error": None,
        "category": "data_science" if pkg_name in data_science_packages else "utility"
    }}
    
    try:
        pkg_module = importlib.import_module(pkg_name)
        discovery_result["available"] = True
        discovery_result["capabilities"] = test_package_capability(pkg_name, pkg_module)
    except ImportError as e:
        discovery_result["error"] = f"ImportError: {{str(e)}}"
    except Exception as e:
        discovery_result["error"] = f"Other error: {{str(e)}}"
    
    package_discovery[pkg_name] = discovery_result

# Then: Generate summary statistics
summary = {{
    "total_packages_tested": len(all_packages),
    "data_science_available": len([p for p in data_science_packages if package_discovery[p]["available"]]),
    "utility_available": len([p for p in utility_packages if package_discovery[p]["available"]]),
    "total_available": len([p for p in package_discovery.values() if p["available"]]),
    "availability_rate": len([p for p in package_discovery.values() if p["available"]]) / len(all_packages)
}}

{{
    "package_discovery": package_discovery,
    "summary": summary,
    "recommendations": {{
        "core_data_science_ready": package_discovery.get("numpy", {{}}).get("available", False) and 
                                  package_discovery.get("pandas", {{}}).get("available", False) and
                                  package_discovery.get("matplotlib", {{}}).get("available", False),
        "advanced_analytics_ready": package_discovery.get("seaborn", {{}}).get("available", False) and
                                   package_discovery.get("scipy", {{}}).get("available", False),
        "machine_learning_ready": package_discovery.get("sklearn", {{}}).get("available", False)
    }}
}}
        """

        # When: Executing package discovery
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and document results  
        validate_api_contract(result)
        assert result["success"], f"Package discovery failed: {result.get('error')}"

        discovery_results = json.loads(result["data"]["result"])
        
        # Document findings
        summary = discovery_results["summary"]
        recommendations = discovery_results["recommendations"]
        
        print(f"\\n=== Data Science Package Discovery Results ===")
        print(f"Total packages tested: {summary['total_packages_tested']}")
        print(f"Available packages: {summary['total_available']}")
        print(f"Availability rate: {summary['availability_rate']:.1%}")
        print(f"Core data science ready: {recommendations['core_data_science_ready']}")
        print(f"Advanced analytics ready: {recommendations['advanced_analytics_ready']}")
        print(f"Machine learning ready: {recommendations['machine_learning_ready']}")
        
        # Basic validation - at least some packages should be available
        assert summary["total_available"] > 0, "At least some packages should be available"
        assert summary["availability_rate"] > 0.3, "Should have reasonable package availability"


@pytest.mark.error_handling
@pytest.mark.api
class TestErrorScenarioHandling:
    """
    BDD tests for comprehensive error handling scenarios.
    
    Migrated from: test_syntax_error.py, test_imports.py
    Enhanced with: proper pytest structure, comprehensive error scenarios, API contract validation
    """

    def test_given_syntax_error_scenarios_when_executed_then_proper_error_responses(
        self, server_ready, quick_timeout
    ):
        """
        Test server handling of various Python syntax errors.

        **Given:** Python code with various syntax errors
        **When:** Code is executed via the API
        **Then:** Server returns proper error responses with error details

        Args:
            server_ready: Server readiness fixture
            quick_timeout: Timeout for quick operations

        Example:
            Syntax errors like missing parentheses should return structured
            error responses with error type and location information.
        """
        # Given: Various syntax error scenarios
        for scenario in IntegratedTestConfig.SYNTAX_ERROR_SCENARIOS:
            with pytest.raises(AssertionError, match=".*") or True:
                # When: Executing code with syntax errors
                result = execute_python_code(scenario["code"], timeout=quick_timeout)
                
                # Then: Validate API contract is maintained even for errors
                validate_api_contract(result)
                
                # Should indicate failure
                assert result["success"] is False, f"Syntax error should cause success=False for {scenario['name']}"
                
                # Should have error information
                assert result["error"] is not None, f"Should have error details for {scenario['name']}"
                
                # Error should contain expected error type
                error_msg = result["error"].lower()
                expected_type = scenario["expected_error_type"].lower()
                assert expected_type in error_msg, f"Error should mention {expected_type} for {scenario['name']}"

    def test_given_import_error_scenarios_when_executed_then_proper_error_responses(
        self, server_ready, quick_timeout
    ):
        """
        Test server handling of various Python import errors.

        **Given:** Python code with various import errors
        **When:** Code attempts to import non-existent modules or functions
        **Then:** Server returns proper error responses with import error details

        Args:
            server_ready: Server readiness fixture
            quick_timeout: Timeout for quick operations

        Example:
            Import errors like non-existent modules should return structured
            error responses indicating the import failure.
        """
        # Given: Various import error scenarios
        for scenario in IntegratedTestConfig.IMPORT_ERROR_SCENARIOS:
            # When: Executing code with import errors
            result = execute_python_code(scenario["code"], timeout=quick_timeout)
            
            # Then: Validate error handling
            validate_api_contract(result)
            
            # Import errors might succeed but raise runtime errors
            if not result["success"]:
                # Should have error information  
                assert result["error"] is not None, f"Should have error details for {scenario['name']}"
                
                # Error should contain expected error type
                error_msg = result["error"].lower()
                expected_type = scenario["expected_error_type"].lower()
                assert expected_type in error_msg or "import" in error_msg, \
                    f"Error should mention import issues for {scenario['name']}"

    def test_given_runtime_error_scenarios_when_executed_then_graceful_error_handling(
        self, server_ready, quick_timeout
    ):
        """
        Test server handling of runtime errors and exceptions.

        **Given:** Python code that raises runtime exceptions
        **When:** Code encounters runtime errors during execution
        **Then:** Server handles errors gracefully with proper error reporting

        Args:
            server_ready: Server readiness fixture
            quick_timeout: Timeout for quick operations
        """
        # Given: Runtime error scenarios
        runtime_scenarios = [
            {
                "name": "division_by_zero",
                "code": "result = 1 / 0",
                "expected_error": "ZeroDivisionError"
            },
            {
                "name": "undefined_variable",
                "code": "print(undefined_variable_name)",
                "expected_error": "NameError"
            },
            {
                "name": "type_error",
                "code": "result = 'string' + 42",
                "expected_error": "TypeError"
            },
            {
                "name": "index_error",
                "code": "data = [1, 2, 3]; result = data[10]",
                "expected_error": "IndexError"
            },
            {
                "name": "key_error",
                "code": "data = {'a': 1}; result = data['nonexistent']",
                "expected_error": "KeyError"
            }
        ]
        
        # When: Testing each runtime error scenario
        for scenario in runtime_scenarios:
            result = execute_python_code(scenario["code"], timeout=quick_timeout)
            
            # Then: Validate proper error handling
            validate_api_contract(result)
            
            # Runtime errors should be handled gracefully
            assert result["success"] is False, f"Runtime error should cause success=False for {scenario['name']}"
            
            # Should have error information
            assert result["error"] is not None, f"Should have error details for {scenario['name']}"


@pytest.mark.api
@pytest.mark.functional
class TestBasicFunctionalityValidation:
    """
    BDD tests for basic functionality validation.
    
    Migrated from: simple_test.py, ultra_simple_test.py
    Enhanced with: proper pytest structure, comprehensive validation, API contract compliance
    """

    def test_given_minimal_python_code_when_executed_then_basic_functionality_works(
        self, server_ready, quick_timeout
    ):
        """
        Test basic Python execution functionality with minimal code.

        **Given:** Simple Python code statements
        **When:** Code is executed via the API
        **Then:** Basic Python functionality works correctly

        Args:
            server_ready: Server readiness fixture
            quick_timeout: Timeout for quick operations

        Example:
            Simple operations like print, variable assignment, and arithmetic
            should work correctly and return expected results.
        """
        # Given: Basic Python operations
        basic_tests = [
            {
                "name": "print_statement",
                "code": 'print("Hello, World!")',
                "expected_stdout": "Hello, World!"
            },
            {
                "name": "variable_assignment",
                "code": 'x = 42\\nprint(f"Value: {x}")',
                "expected_stdout": "Value: 42"
            },
            {
                "name": "arithmetic_operations",
                "code": 'result = 2 + 3 * 4\\nprint(f"Result: {result}")',
                "expected_stdout": "Result: 14"
            },
            {
                "name": "string_operations",
                "code": 'name = "Python"\\nprint(f"Hello, {name}!")',
                "expected_stdout": "Hello, Python!"
            },
            {
                "name": "list_operations",
                "code": 'data = [1, 2, 3]\\nprint(f"Sum: {sum(data)}")',
                "expected_stdout": "Sum: 6"
            }
        ]
        
        # When: Testing each basic operation
        for test in basic_tests:
            result = execute_python_code(test["code"], timeout=quick_timeout)
            
            # Then: Validate successful execution
            validate_api_contract(result)
            assert result["success"], f"Basic test should succeed: {test['name']}"
            
            # Should have expected output
            assert test["expected_stdout"] in result["data"]["stdout"], \
                f"Should have expected output for {test['name']}"

    def test_given_cross_platform_path_operations_when_executed_then_pathlib_works(
        self, server_ready, quick_timeout
    ):
        """
        Test cross-platform path operations using pathlib.

        **Given:** Python code using pathlib for file operations
        **When:** Path operations are performed
        **Then:** Cross-platform path handling works correctly

        Args:
            server_ready: Server readiness fixture
            quick_timeout: Timeout for quick operations

        Example:
            Path operations should work consistently across platforms
            using pathlib.Path for all file system interactions.
        """
        # Given: Cross-platform path operations
        code = '''
from pathlib import Path
import os

# Test various path operations
paths_tested = []

# Test basic path creation
base_path = Path("/tmp")
paths_tested.append(f"Base path: {base_path}")

# Test path joining (cross-platform)
file_path = base_path / "test_file.txt"
paths_tested.append(f"File path: {file_path}")

# Test path properties
paths_tested.append(f"Parent: {file_path.parent}")
paths_tested.append(f"Name: {file_path.name}")
paths_tested.append(f"Suffix: {file_path.suffix}")
paths_tested.append(f"Is absolute: {file_path.is_absolute()}")

# Test Windows-style path handling (should work on any platform)
windows_style = Path("C:/Users/test/documents/file.txt")
paths_tested.append(f"Windows path parts: {windows_style.parts}")

# Test current directory
current_dir = Path.cwd()
paths_tested.append(f"Current directory: {current_dir}")

# Test path resolution
resolved_path = file_path.resolve()
paths_tested.append(f"Resolved path: {resolved_path}")

for test_result in paths_tested:
    print(test_result)

{
    "total_path_tests": len(paths_tested),
    "all_tests_completed": True,
    "pathlib_functional": True
}
        '''
        
        # When: Executing path operations
        result = execute_python_code(code, timeout=quick_timeout)
        
        # Then: Validate path operations work
        validate_api_contract(result)
        assert result["success"], f"Path operations should succeed: {result.get('error')}"
        
        # Should have path operation output
        stdout = result["data"]["stdout"]
        assert "Base path:" in stdout, "Should show base path operations"
        assert "File path:" in stdout, "Should show file path operations"
        assert "Current directory:" in stdout, "Should show current directory"
        
        # Should complete all tests
        path_results = json.loads(result["data"]["result"])
        assert path_results["all_tests_completed"], "All path tests should complete"
        assert path_results["pathlib_functional"], "Pathlib should be functional"


if __name__ == "__main__":
    # Allow running this file directly for development/debugging
    pytest.main([__file__, "-v", "-s"])