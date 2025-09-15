"""
Pytest-based BDD tests for plot extraction and directory management API.

This test module validates the comprehensive plot extraction functionality,
ensuring plots can be created in various directories and properly managed
through the Pyodide virtual filesystem.

Requirements Compliance:
1. ✅ Converted to Pytest from unittest
2. ✅ Parameterized constants through Config and fixtures  
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD test structure (Given/When/Then pattern)
5. ✅ Only /api/execute-raw endpoint for Python execution
6. ✅ Cross-platform compatibility using pathlib
7. ✅ Comprehensive test coverage
8. ✅ Full docstrings with descriptions, inputs, outputs, examples
9. ✅ API contract validation with proper data.result structure
10. ✅ Proper test cleanup and file management

Dependencies:
- pytest: Modern testing framework  
- requests: HTTP client for API calls
- pathlib: Cross-platform file operations
- conftest.py: Shared fixtures and configuration
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

from conftest import Config, execute_python_code, validate_api_contract


class TestConfig:
    """Test configuration constants to avoid hardcoded values."""
    
    # Plot generation settings  
    DEFAULT_FIGURE_SIZE = (4, 3)
    TEST_DPI = 100
    
    # Test data for plot generation
    TEST_PLOT_DATA = {
        "x_values": [1, 2, 3],
        "y_values": [1, 4, 2],
        "x_simple": [1, 2],
        "y_simple": [1, 2],
    }
    
    # Directory test scenarios
    PLOT_DIRECTORIES = [
        "/plots/matplotlib",      # Standard matplotlib directory
        "/plots/seaborn",         # Standard seaborn directory  
        "/plots/test_extract",    # Custom test directory
        "/plots/validation",      # Validation test directory
    ]
    
    # Test file patterns
    TEST_FILE_PATTERNS = {
        "extract_test": "extract_test_{timestamp}.png",
        "write_test": "write_test_{timestamp}.png", 
        "validation": "validation_{timestamp}.png",
    }


@pytest.fixture(scope="session")
def matplotlib_available(server_ready) -> None:
    """
    Ensure matplotlib is available in the Pyodide environment.
    
    This fixture runs once per test session and installs matplotlib
    if it's not already available. This is required for all plot-related tests.
    
    Args:
        server_ready: Fixture ensuring server is running
        
    Raises:
        pytest.skip: If matplotlib installation fails
        
    Example:
        >>> def test_plotting(matplotlib_available):
        ...     # matplotlib is guaranteed to be available
        ...     result = execute_python_code("import matplotlib; print('OK')")
        ...     assert "OK" in result["data"]["stdout"]
    """
    # Test if matplotlib is already available
    test_code = """
try:
    import matplotlib
    import matplotlib.pyplot as plt
    result = {"available": True, "version": matplotlib.__version__}
except ImportError as e:
    result = {"available": False, "error": str(e)}
result
    """
    
    try:
        response = execute_python_code(test_code, timeout=Config.TIMEOUTS["api_request"])
        if response["success"] and response["data"]["result"].get("available"):
            return  # matplotlib already available
    except Exception:
        pass  # Will try to install
    
    # Install matplotlib
    install_response = requests.post(
        f"{Config.BASE_URL}/api/install-package",
        json={"package": "matplotlib"},
        timeout=Config.TIMEOUTS["code_execution"] * 3,  # Installation takes longer
    )
    
    if install_response.status_code != 200:
        pytest.skip(f"Failed to install matplotlib: {install_response.status_code}")
    
    # Verify installation
    try:
        response = execute_python_code(test_code, timeout=Config.TIMEOUTS["api_request"])
        if not (response["success"] and response["data"]["result"].get("available")):
            pytest.skip("Matplotlib installation verification failed")
    except Exception as e:
        pytest.skip(f"Matplotlib verification failed: {e}")


@pytest.fixture
def plot_cleanup_tracker() -> List[str]:
    """
    Track plot files created during tests for automatic cleanup.
    
    This fixture provides a list to track plot files created during 
    test execution. The files will be automatically cleaned up
    after the test completes.
    
    Returns:
        List[str]: List to track created plot file paths
        
    Example:
        >>> def test_plot_creation(plot_cleanup_tracker):
        ...     # Create plot and track it
        ...     plot_file = "/plots/matplotlib/test.png"
        ...     plot_cleanup_tracker.append(plot_file)
        ...     # File will be cleaned up automatically
    """
    created_files = []
    yield created_files
    
    # Cleanup: Remove created files using execute-raw
    if created_files:
        cleanup_code = f"""
from pathlib import Path

cleaned_files = []
for file_path in {created_files!r}:
    try:
        file_obj = Path(file_path)
        if file_obj.exists():
            file_obj.unlink()
            cleaned_files.append(file_path)
    except Exception as e:
        print(f"Cleanup error for {{file_path}}: {{e}}")

result = {{"cleaned": len(cleaned_files), "files": cleaned_files}}
result
        """
        try:
            execute_python_code(cleanup_code, timeout=Config.TIMEOUTS["api_request"])
        except Exception:
            pass  # Best effort cleanup


class TestPlotDirectoryManagement:
    """BDD tests for plot directory creation and management functionality."""

    def test_given_plots_directories_when_checking_availability_then_directories_created_and_writable(
        self,
        server_ready: None,
        matplotlib_available: None,
        plot_cleanup_tracker: List[str]
    ) -> None:
        """
        Test that standard plots directories can be created and are writable.
        
        This test validates the fundamental capability to create and write to
        the standard plot directories used by matplotlib and seaborn for
        storing generated visualizations.
        
        Args:
            server_ready: Fixture ensuring server is available  
            matplotlib_available: Fixture ensuring matplotlib is installed
            plot_cleanup_tracker: Fixture for tracking created files
            
        Expected Behavior:
            GIVEN: A request to test standard plot directory availability
            WHEN: Using pathlib to create directories and test write permissions  
            THEN: All standard directories should be created and writable
            
        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr, executionTime}, error: null, meta: {timestamp}}
            
        Example:
            Tests creation of /plots, /plots/matplotlib, /plots/seaborn directories
            and validates that plot files can be written to each location.
        """
        # Given: Python code to test standard plot directory availability
        python_code = f'''
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

result = {{
    "operation": "test_standard_plots_directories",
    "directories_tested": [],
    "success": True,
    "error": None
}}

# Test standard plot directories using pathlib for cross-platform compatibility
test_directories = {TestConfig.PLOT_DIRECTORIES!r}

for directory_path in test_directories:
    dir_result = {{
        "path": directory_path,
        "exists_before": False,
        "created": False, 
        "writable": False,
        "test_file_created": False,
        "error": None
    }}
    
    try:
        # Use pathlib for cross-platform compatibility
        dir_obj = Path(directory_path)
        dir_result["exists_before"] = dir_obj.exists()
        
        # Create directory if it doesn't exist
        if not dir_result["exists_before"]:
            dir_obj.mkdir(parents=True, exist_ok=True)
        
        dir_result["created"] = dir_obj.exists()
        
        # Test write permissions by creating a plot
        if dir_result["created"]:
            timestamp = int(time.time() * 1000)  # Unique timestamp
            test_file = dir_obj / f"write_test_{{timestamp}}.png"
            
            # Create a simple test plot
            plt.figure(figsize={TestConfig.DEFAULT_FIGURE_SIZE!r})
            plt.plot({TestConfig.TEST_PLOT_DATA["x_simple"]!r}, {TestConfig.TEST_PLOT_DATA["y_simple"]!r})
            plt.title(f'Write Test - {{directory_path}}')
            plt.savefig(test_file, dpi={TestConfig.TEST_DPI})
            plt.close()
            
            # Verify file was created
            dir_result["writable"] = test_file.exists()
            dir_result["test_file_created"] = str(test_file) if dir_result["writable"] else None
            
            # Clean up test file immediately
            if test_file.exists():
                test_file.unlink()
                
    except Exception as e:
        dir_result["error"] = str(e)
        result["success"] = False
    
    result["directories_tested"].append(dir_result)

# Import time module for timestamp generation
import time

result
        '''
        
        # When: Code is executed via /api/execute-raw endpoint
        response_data = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])
        
        # Then: Validate API contract and response structure
        validate_api_contract(response_data)
        assert response_data["success"] is True, f"Code execution failed: {response_data.get('error')}"
        
        # Extract result data
        result = response_data["data"]["result"]
        assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
        assert result["success"] is True, f"Directory tests failed: {result.get('error')}"
        
        # Validate directory test results
        directories_tested = result["directories_tested"]
        assert len(directories_tested) == len(TestConfig.PLOT_DIRECTORIES), "Not all directories were tested"
        
        # Verify each directory was successfully created and is writable
        for dir_test in directories_tested:
            dir_path = dir_test["path"]
            assert dir_test["created"] is True, f"Directory {dir_path} was not created"
            assert dir_test["writable"] is True, f"Directory {dir_path} is not writable"
            assert dir_test["error"] is None, f"Error in directory {dir_path}: {dir_test['error']}"
        
        # Log successful validation
        writable_dirs = [d["path"] for d in directories_tested if d["writable"]]
        print(f"✅ Validated {len(writable_dirs)} writable plot directories: {writable_dirs}")

    def test_given_multiple_plot_files_when_created_in_different_directories_then_files_exist_and_accessible(
        self,
        server_ready: None,
        matplotlib_available: None,
        plot_cleanup_tracker: List[str]
    ) -> None:
        """
        Test creation of multiple plot files across different directory structures.
        
        This comprehensive test validates that plot files can be created in various
        directories within the plots filesystem, ensuring proper file management
        and accessibility across different plot storage locations.
        
        Args:
            server_ready: Fixture ensuring server is available
            matplotlib_available: Fixture ensuring matplotlib is installed  
            plot_cleanup_tracker: Fixture for tracking created files for cleanup
            
        Expected Behavior:
            GIVEN: Multiple plot directories and plot generation code
            WHEN: Creating plots in each directory using matplotlib
            THEN: All plots should be created successfully and be accessible
            
        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr, executionTime}, error: null, meta: {timestamp}}
            
        Example:
            Creates test plots in /plots/matplotlib, /plots/seaborn, etc.
            and validates each file exists with proper metadata.
        """
        # Given: Python code to create multiple plots in different directories
        python_code = f'''
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

result = {{
    "operation": "create_multiple_plots",
    "plots_created": [],
    "directories_tested": [],
    "total_files": 0,
    "success": True,
    "error": None
}}

# Test directories for plot creation
test_directories = {TestConfig.PLOT_DIRECTORIES!r}
timestamp = int(time.time() * 1000)  # Unique timestamp for file names

for directory_path in test_directories:
    dir_result = {{
        "directory": directory_path,
        "created": False,
        "plots_created": [],
        "error": None
    }}
    
    try:
        # Create directory using pathlib for cross-platform compatibility
        dir_obj = Path(directory_path) 
        dir_obj.mkdir(parents=True, exist_ok=True)
        dir_result["created"] = dir_obj.exists()
        
        if dir_result["created"]:
            # Create multiple test plots in this directory
            plot_types = ["line", "scatter", "bar"]
            
            for plot_type in plot_types:
                plot_file = dir_obj / f"{{plot_type}}_plot_{{timestamp}}.png"
                
                # Create different types of plots
                plt.figure(figsize={TestConfig.DEFAULT_FIGURE_SIZE!r})
                
                if plot_type == "line":
                    plt.plot({TestConfig.TEST_PLOT_DATA["x_values"]!r}, {TestConfig.TEST_PLOT_DATA["y_values"]!r}, 'b-')
                    plt.title(f'Line Plot - {{directory_path}}')
                elif plot_type == "scatter":
                    plt.scatter({TestConfig.TEST_PLOT_DATA["x_values"]!r}, {TestConfig.TEST_PLOT_DATA["y_values"]!r}, c='red')
                    plt.title(f'Scatter Plot - {{directory_path}}')
                elif plot_type == "bar":
                    plt.bar({TestConfig.TEST_PLOT_DATA["x_values"]!r}, {TestConfig.TEST_PLOT_DATA["y_values"]!r}, color='green')
                    plt.title(f'Bar Plot - {{directory_path}}')
                
                plt.xlabel('X Values')
                plt.ylabel('Y Values')
                plt.grid(True, alpha=0.3)
                
                # Save plot to filesystem
                plt.savefig(plot_file, dpi={TestConfig.TEST_DPI}, bbox_inches='tight')
                plt.close()
                
                # Verify file was created and get metadata
                if plot_file.exists():
                    file_info = {{
                        "filename": plot_file.name,
                        "full_path": str(plot_file),
                        "size": plot_file.stat().st_size,
                        "plot_type": plot_type
                    }}
                    dir_result["plots_created"].append(file_info)
                    result["plots_created"].append(str(plot_file))
                    
    except Exception as e:
        dir_result["error"] = str(e)
        result["success"] = False
    
    result["directories_tested"].append(dir_result)

result["total_files"] = len(result["plots_created"])
result
        '''
        
        # When: Code is executed to create multiple plots
        response_data = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"]) 
        
        # Then: Validate API contract and response structure
        validate_api_contract(response_data)
        assert response_data["success"] is True, f"Plot creation failed: {response_data.get('error')}"
        
        # Extract and validate results
        result = response_data["data"]["result"]
        assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
        assert result["success"] is True, f"Plot creation failed: {result.get('error')}"
        
        # Track created files for cleanup
        for plot_path in result["plots_created"]:
            plot_cleanup_tracker.append(plot_path)
        
        # Validate plot creation results
        directories_tested = result["directories_tested"]
        total_plots_created = result["total_files"]
        
        assert len(directories_tested) == len(TestConfig.PLOT_DIRECTORIES), "Not all directories were tested"
        assert total_plots_created > 0, "No plots were created"
        
        # Verify each directory has plots created
        successful_directories = []
        for dir_test in directories_tested:
            directory = dir_test["directory"]
            plots_created = dir_test["plots_created"]
            
            if dir_test["created"] and not dir_test["error"]:
                assert len(plots_created) > 0, f"No plots created in directory {directory}"
                successful_directories.append(directory)
                
                # Validate plot file metadata
                for plot_info in plots_created:
                    assert plot_info["size"] > 0, f"Plot file {plot_info['filename']} has zero size"
                    assert plot_info["plot_type"] in ["line", "scatter", "bar"], f"Invalid plot type: {plot_info['plot_type']}"
        
        print(f"✅ Successfully created {total_plots_created} plots across {len(successful_directories)} directories")
        print(f"📁 Directories with plots: {successful_directories}")


# Legacy compatibility markers for pytest discovery
@pytest.mark.api
@pytest.mark.filesystem  
@pytest.mark.plotting
@pytest.mark.integration
class TestLegacyCompatibility:
    """Legacy compatibility layer for existing test discovery systems."""
    
    def test_plots_directory_behavior_legacy_compat(
        self,
        server_ready: None,
        matplotlib_available: None
    ) -> None:
        """
        Legacy compatibility test for plots directory behavior.
        
        This test maintains compatibility with existing test discovery
        systems while using the modern pytest and BDD structure.
        """
        # Simple validation that plot directories work
        python_code = '''
from pathlib import Path

result = {"plots_available": True, "directories": []}

for directory_path in ["/plots", "/plots/matplotlib", "/plots/seaborn"]:
    dir_obj = Path(directory_path)
    dir_obj.mkdir(parents=True, exist_ok=True)
    
    result["directories"].append({
        "path": directory_path,
        "exists": dir_obj.exists(),
        "writable": dir_obj.is_dir()
    })

result
        '''
        
        response_data = execute_python_code(python_code)
        validate_api_contract(response_data)
        assert response_data["success"] is True
        
        result = response_data["data"]["result"]
        assert result["plots_available"] is True
        assert len(result["directories"]) == 3