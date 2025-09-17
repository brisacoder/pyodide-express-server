"""
BDD tests for virtual filesystem behavior and debugging filesystem issues.

Migrated from: Original unittest test_virtual_filesystem.py
Enhanced with: pytest structure, comprehensive BDD coverage, API contract validation

This module tests:
- Virtual filesystem structure discovery and validation
- Directory creation and management operations
- File creation, reading, writing, and management
- Matplotlib plot saving to virtual filesystem locations
- Cross-platform compatibility using pathlib

All tests use the /api/execute-raw endpoint and follow the API contract:
{
  "success": true | false,
  "data": <object|null>,
  "error": <string|null>,
  "meta": { "timestamp": <string> }
}
"""

import pytest
import json
import time
from tests.conftest import execute_python_code, validate_api_contract


class VirtualFilesystemTestConfig:
    """Configuration constants for virtual filesystem tests."""
    
    # Filesystem test parameters
    FILESYSTEM_SETTINGS = {
        "default_timeout": 60,
        "plot_timeout": 90,
        "api_timeout": 45,
        "min_file_size": 1000,  # Minimum expected file size for plots
        "max_directory_listing": 20,  # Maximum items to list in directory
        "test_content": "Hello virtual filesystem test"
    }
    
    # Test paths for virtual filesystem
    FILESYSTEM_PATHS = [
        {
            "path": "/",
            "description": "Root directory - should always exist",
            "should_exist": True,
            "is_directory": True
        },
        {
            "path": "/tmp",
            "description": "Temporary directory - commonly available",
            "should_exist": None,  # May or may not exist
            "is_directory": True
        },
        {
            "path": "/home",
            "description": "Home directory - may exist in some environments",
            "should_exist": None,
            "is_directory": True
        },
        {
            "path": "/home/pyodide",
            "description": "Pyodide user home directory",
            "should_exist": None,
            "is_directory": True
        },
        {
            "path": "/home/pyodide/plots",
            "description": "Pyodide plots directory",
            "should_exist": None,
            "is_directory": True
        }
    ]
    
    # File operation test scenarios
    FILE_OPERATIONS = [
        {
            "name": "basic_text_file",
            "directory": "/test_vfs_basic",
            "filename": "test.txt",
            "content": "Basic test content",
            "operation_type": "text"
        },
        {
            "name": "json_data_file",
            "directory": "/test_vfs_json",
            "filename": "data.json",
            "content": '{"test": "data", "number": 42}',
            "operation_type": "json"
        },
        {
            "name": "multiline_file",
            "directory": "/test_vfs_multi",
            "filename": "multiline.txt",
            "content": "Line 1\nLine 2\nLine 3\n",
            "operation_type": "multiline"
        }
    ]


@pytest.fixture
def filesystem_timeout():
    """
    Provide timeout for filesystem operations.

    Returns:
        int: Timeout in seconds for filesystem operations

    Example:
        >>> def test_filesystem(filesystem_timeout):
        ...     result = execute_python_code(code, timeout=filesystem_timeout)
    """
    return VirtualFilesystemTestConfig.FILESYSTEM_SETTINGS["default_timeout"]


@pytest.fixture
def plot_timeout():
    """
    Provide timeout for plot generation and saving operations.

    Returns:
        int: Timeout in seconds for plot operations

    Example:
        >>> def test_plot_save(plot_timeout):
        ...     result = execute_python_code(code, timeout=plot_timeout)
    """
    return VirtualFilesystemTestConfig.FILESYSTEM_SETTINGS["plot_timeout"]


@pytest.fixture
def api_timeout():
    """
    Provide timeout for API operations.

    Returns:
        int: Timeout in seconds for API operations

    Example:
        >>> def test_api_call(api_timeout):
        ...     result = execute_python_code(code, timeout=api_timeout)
    """
    return VirtualFilesystemTestConfig.FILESYSTEM_SETTINGS["api_timeout"]


@pytest.fixture
def unique_timestamp():
    """
    Generate unique timestamp for test file naming.

    Returns:
        int: Unique timestamp in milliseconds

    Example:
        >>> def test_file_creation(unique_timestamp):
        ...     filename = f"test_{unique_timestamp}.txt"
        ...     # Use filename for unique test files
    """
    return int(time.time() * 1000)


@pytest.fixture
def cleanup_test_directories():
    """
    Cleanup fixture to remove test directories after tests.

    Yields:
        List[str]: List to append created directory paths for cleanup

    Example:
        >>> def test_directory_creation(cleanup_test_directories):
        ...     test_dir = "/test_cleanup_example"
        ...     cleanup_test_directories.append(test_dir)
        ...     # Directory will be automatically cleaned up
    """
    created_directories = []
    yield created_directories

    # Cleanup created directories
    if created_directories:
        cleanup_code = f"""
from pathlib import Path
import shutil

cleaned_dirs = []
errors = []

for dir_path in {created_directories}:
    try:
        test_dir = Path(dir_path)
        if test_dir.exists():
            if test_dir.is_dir():
                shutil.rmtree(str(test_dir))
            else:
                test_dir.unlink()
            cleaned_dirs.append(str(test_dir))
    except Exception as e:
        errors.append(f"{{dir_path}}: {{str(e)}}")

{{
    "cleaned_directories": cleaned_dirs,
    "errors": errors,
    "total_cleaned": len(cleaned_dirs)
}}
        """
        try:
            execute_python_code(cleanup_code)
        except Exception:
            # Don't fail tests due to cleanup issues
            pass


@pytest.mark.filesystem
@pytest.mark.virtual
@pytest.mark.api
class TestVirtualFilesystemStructure:
    """
    BDD tests for virtual filesystem structure discovery and validation.
    
    Migrated from: Original test_virtual_filesystem_structure
    Enhanced with: proper pytest structure, comprehensive path testing, enhanced validation
    """

    def test_given_pyodide_environment_when_exploring_filesystem_structure_then_discovers_virtual_paths(
        self, server_ready, filesystem_timeout
    ):
        """
        Test comprehensive virtual filesystem structure discovery.

        **Given:** Pyodide environment with virtual filesystem
        **When:** Exploring various filesystem paths and structures
        **Then:** Discovers and validates virtual filesystem layout

        Args:
            server_ready: Server readiness fixture
            filesystem_timeout: Timeout for filesystem operations

        Example:
            Explores root directory, common paths like /tmp, /home, and
            validates their existence, accessibility, and basic properties.
        """
        # Given: Virtual filesystem paths to explore
        filesystem_paths = VirtualFilesystemTestConfig.FILESYSTEM_PATHS
        max_listing = VirtualFilesystemTestConfig.FILESYSTEM_SETTINGS["max_directory_listing"]
        
        code = f"""
from pathlib import Path
import json

# When: Exploring virtual filesystem structure
result = {{
    "current_directory": str(Path.cwd()),
    "filesystem_discovery": {{}},
    "summary": {{}}
}}

test_paths = {filesystem_paths}

for path_info in test_paths:
    path_str = path_info['path']
    description = path_info['description']
    should_exist = path_info['should_exist']
    is_directory = path_info['is_directory']
    
    try:
        test_path = Path(path_str)
        
        path_result = {{
            "description": description,
            "path": path_str,
            "exists": test_path.exists(),
            "is_directory": test_path.is_dir() if test_path.exists() else None,
            "is_file": test_path.is_file() if test_path.exists() else None,
            "should_exist": should_exist,
            "expected_directory": is_directory
        }}
        
        # If path exists and is directory, list contents
        if test_path.exists() and test_path.is_dir():
            try:
                contents = list(test_path.iterdir())
                path_result["contents"] = [
                    {{
                        "name": p.name,
                        "is_dir": p.is_dir(),
                        "is_file": p.is_file()
                    }}
                    for p in contents[:{max_listing}]
                ]
                path_result["total_items"] = len(contents)
                path_result["listing_truncated"] = len(contents) > {max_listing}
            except Exception as list_error:
                path_result["listing_error"] = str(list_error)
        
        result["filesystem_discovery"][path_str] = path_result
        
    except Exception as e:
        result["filesystem_discovery"][path_str] = {{
            "description": description,
            "path": path_str,
            "error": str(e),
            "accessible": False
        }}

# Then: Generate summary statistics
total_paths = len(test_paths)
existing_paths = sum(1 for info in result["filesystem_discovery"].values()
                    if info.get("exists", False))
accessible_paths = sum(1 for info in result["filesystem_discovery"].values()
                      if not info.get("error"))
directories_found = sum(1 for info in result["filesystem_discovery"].values()
                       if info.get("is_directory", False))

result["summary"] = {{
    "total_paths_tested": total_paths,
    "existing_paths": existing_paths,
    "accessible_paths": accessible_paths,
    "directories_found": directories_found,
    "filesystem_responsive": accessible_paths > 0,
    "root_accessible": result["filesystem_discovery"].get("/", {{}}).get("exists", False)
}}

print(f"Filesystem exploration complete: {{existing_paths}}/{{total_paths}} paths exist")
json.dumps(result)
        """

        # When: Executing filesystem structure discovery
        result = execute_python_code(code, timeout=filesystem_timeout)

        # Then: Validate API contract and filesystem results
        validate_api_contract(result)
        assert result["success"], f"Filesystem structure discovery failed: {result.get('error')}"

        filesystem_results = json.loads(result["data"]["result"])
        
        # Validate basic filesystem responsiveness
        summary = filesystem_results["summary"]
        assert summary["filesystem_responsive"], "Virtual filesystem should be responsive"
        assert summary["root_accessible"], "Root directory should be accessible"
        
        # Validate discovery results
        discovery = filesystem_results["filesystem_discovery"]
        assert len(discovery) > 0, "Should have discovered filesystem paths"
        
        # Root directory should always exist
        root_info = discovery.get("/")
        assert root_info is not None, "Root directory should be discovered"
        assert root_info["exists"], "Root directory should exist"
        assert root_info["is_directory"], "Root should be a directory"
        
        # Validate expected path behavior
        for path_str, path_info in discovery.items():
            if path_info.get("should_exist") is True:
                assert path_info["exists"], f"Path {path_str} should exist according to expectations"
            
            # If path exists and should be directory, validate
            if path_info.get("exists") and path_info.get("expected_directory"):
                assert path_info["is_directory"], f"Path {path_str} should be a directory"


@pytest.mark.filesystem
@pytest.mark.operations
@pytest.mark.api
class TestVirtualFilesystemOperations:
    """
    BDD tests for virtual filesystem directory and file operations.
    
    Migrated from: Original test_directory_creation_and_file_operations
    Enhanced with: proper pytest structure, comprehensive operation testing, cleanup handling
    """

    def test_given_writable_filesystem_when_creating_directories_then_directory_operations_succeed(
        self, server_ready, filesystem_timeout, unique_timestamp, cleanup_test_directories
    ):
        """
        Test directory creation and management operations.

        **Given:** Writable virtual filesystem locations
        **When:** Creating, navigating, and managing directories
        **Then:** Directory operations succeed with proper validation

        Args:
            server_ready: Server readiness fixture
            filesystem_timeout: Timeout for filesystem operations
            unique_timestamp: Unique timestamp for test isolation
            cleanup_test_directories: Cleanup tracking list

        Example:
            Creates test directories, validates their existence, tests nested
            directory creation, and verifies directory listing operations.
        """
        # Given: Directory operation test scenarios
        test_base = f"/test_dirs_{unique_timestamp}"
        cleanup_test_directories.append(test_base)
        
        code = f"""
from pathlib import Path
import json

# When: Testing directory creation and management
result = {{
    "directory_operations": [],
    "final_state": {{}},
    "operation_summary": {{}}
}}

test_timestamp = {unique_timestamp}
base_dir = Path("/test_dirs_{{test_timestamp}}")

# Test 1: Basic directory creation
try:
    base_dir.mkdir(parents=True, exist_ok=True)
    
    result["directory_operations"].append({{
        "operation": "create_base_directory",
        "path": str(base_dir),
        "success": base_dir.exists(),
        "is_directory": base_dir.is_dir() if base_dir.exists() else False
    }})
except Exception as e:
    result["directory_operations"].append({{
        "operation": "create_base_directory",
        "path": str(base_dir),
        "error": str(e),
        "success": False
    }})

# Test 2: Nested directory creation
nested_dir = base_dir / "level1" / "level2" / "level3"
try:
    nested_dir.mkdir(parents=True, exist_ok=True)
    
    result["directory_operations"].append({{
        "operation": "create_nested_directories",
        "path": str(nested_dir),
        "success": nested_dir.exists(),
        "is_directory": nested_dir.is_dir() if nested_dir.exists() else False,
        "parent_exists": nested_dir.parent.exists()
    }})
except Exception as e:
    result["directory_operations"].append({{
        "operation": "create_nested_directories",
        "path": str(nested_dir),
        "error": str(e),
        "success": False
    }})

# Test 3: Multiple sibling directories
sibling_dirs = ["subdir_a", "subdir_b", "subdir_c"]
for subdir_name in sibling_dirs:
    subdir_path = base_dir / subdir_name
    try:
        subdir_path.mkdir(exist_ok=True)
        
        result["directory_operations"].append({{
            "operation": "create_sibling_directory",
            "path": str(subdir_path),
            "name": subdir_name,
            "success": subdir_path.exists(),
            "is_directory": subdir_path.is_dir() if subdir_path.exists() else False
        }})
    except Exception as e:
        result["directory_operations"].append({{
            "operation": "create_sibling_directory",
            "path": str(subdir_path),
            "name": subdir_name,
            "error": str(e),
            "success": False
        }})

# Test 4: Directory listing and validation
if base_dir.exists():
    try:
        dir_contents = list(base_dir.iterdir())
        
        result["directory_operations"].append({{
            "operation": "list_directory_contents",
            "path": str(base_dir),
            "success": True,
            "contents": [
                {{
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "is_file": p.is_file(),
                    "path": str(p)
                }}
                for p in dir_contents
            ],
            "item_count": len(dir_contents)
        }})
    except Exception as e:
        result["directory_operations"].append({{
            "operation": "list_directory_contents",
            "path": str(base_dir),
            "error": str(e),
            "success": False
        }})

# Then: Generate final state and summary
try:
    result["final_state"] = {{
        "base_directory_exists": base_dir.exists(),
        "base_directory_is_dir": base_dir.is_dir() if base_dir.exists() else False,
        "nested_directory_exists": nested_dir.exists(),
        "sibling_directories_created": sum(
            1 for subdir_name in sibling_dirs 
            if (base_dir / subdir_name).exists()
        ),
        "total_items_in_base": len(list(base_dir.iterdir())) if base_dir.exists() else 0
    }}
except Exception as e:
    result["final_state"] = {{"error": str(e)}}

# Operation summary
successful_ops = sum(1 for op in result["directory_operations"] if op.get("success", False))
total_ops = len(result["directory_operations"])

result["operation_summary"] = {{
    "total_operations": total_ops,
    "successful_operations": successful_ops,
    "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
    "all_operations_successful": successful_ops == total_ops,
    "directory_creation_functional": successful_ops > 0
}}

print(f"Directory operations complete: {{successful_ops}}/{{total_ops}} successful")
json.dumps(result)
        """

        # When: Executing directory operations
        result = execute_python_code(code, timeout=filesystem_timeout)

        # Then: Validate directory operations success
        validate_api_contract(result)
        assert result["success"], f"Directory operations failed: {result.get('error')}"

        directory_results = json.loads(result["data"]["result"])
        
        # Validate operation summary
        summary = directory_results["operation_summary"]
        assert summary["directory_creation_functional"], "Directory creation should be functional"
        assert summary["successful_operations"] > 0, "Should have at least one successful directory operation"
        
        # Validate final state
        final_state = directory_results["final_state"]
        assert final_state.get("base_directory_exists"), "Base test directory should exist"
        assert final_state.get("base_directory_is_dir"), "Base directory should be a directory"
        
        # Validate individual operations
        operations = directory_results["directory_operations"]
        create_ops = [op for op in operations if "create" in op.get("operation", "")]
        successful_creates = [op for op in create_ops if op.get("success")]
        assert len(successful_creates) > 0, "Should have at least one successful directory creation"

    def test_given_created_directories_when_performing_file_operations_then_file_management_succeeds(
        self, server_ready, filesystem_timeout, unique_timestamp, cleanup_test_directories
    ):
        """
        Test comprehensive file creation, reading, writing, and management.

        **Given:** Created directories in virtual filesystem
        **When:** Creating, writing, reading, and managing files
        **Then:** File operations succeed with proper content validation

        Args:
            server_ready: Server readiness fixture
            filesystem_timeout: Timeout for filesystem operations
            unique_timestamp: Unique timestamp for test isolation
            cleanup_test_directories: Cleanup tracking list

        Example:
            Creates various file types (text, JSON, multiline), validates
            content integrity, tests file properties, and file management.
        """
        # Given: File operation test scenarios
        file_operations = VirtualFilesystemTestConfig.FILE_OPERATIONS
        test_base = f"/test_files_{unique_timestamp}"
        cleanup_test_directories.append(test_base)
        
        code = f"""
from pathlib import Path
import json

# When: Testing comprehensive file operations
result = {{
    "file_operations": [],
    "content_validation": [],
    "file_properties": [],
    "operation_summary": {{}}
}}

test_timestamp = {unique_timestamp}
file_scenarios = {file_operations}

# Process each file operation scenario
for scenario in file_scenarios:
    scenario_name = scenario['name']
    directory = scenario['directory'] + f"_{{test_timestamp}}"
    filename = scenario['filename']
    content = scenario['content']
    operation_type = scenario['operation_type']
    
    # Create directory for this scenario
    test_dir = Path(directory)
    file_path = test_dir / filename
    
    try:
        # Create directory
        test_dir.mkdir(parents=True, exist_ok=True)
        
        result["file_operations"].append({{
            "scenario": scenario_name,
            "operation": "create_directory",
            "path": str(test_dir),
            "success": test_dir.exists()
        }})
        
        # Create and write file
        file_path.write_text(content)
        
        result["file_operations"].append({{
            "scenario": scenario_name,
            "operation": "create_file",
            "path": str(file_path),
            "filename": filename,
            "success": file_path.exists(),
            "operation_type": operation_type
        }})
        
        # Read and validate content
        if file_path.exists():
            read_content = file_path.read_text()
            content_matches = read_content == content
            
            result["content_validation"].append({{
                "scenario": scenario_name,
                "path": str(file_path),
                "content_matches": content_matches,
                "original_length": len(content),
                "read_length": len(read_content),
                "operation_type": operation_type
            }})
            
            # Test specific content validation based on type
            if operation_type == "json":
                try:
                    json_data = json.loads(read_content)
                    result["content_validation"][-1]["valid_json"] = True
                    result["content_validation"][-1]["json_data"] = json_data
                except Exception as json_error:
                    result["content_validation"][-1]["valid_json"] = False
                    result["content_validation"][-1]["json_error"] = str(json_error)
            
            # Get file properties
            try:
                file_stat = file_path.stat()
                
                result["file_properties"].append({{
                    "scenario": scenario_name,
                    "path": str(file_path),
                    "size": file_stat.st_size,
                    "is_file": file_path.is_file(),
                    "is_dir": file_path.is_dir(),
                    "name": file_path.name,
                    "suffix": file_path.suffix,
                    "parent_exists": file_path.parent.exists()
                }})
            except Exception as stat_error:
                result["file_properties"].append({{
                    "scenario": scenario_name,
                    "path": str(file_path),
                    "stat_error": str(stat_error)
                }})
    
    except Exception as e:
        result["file_operations"].append({{
            "scenario": scenario_name,
            "operation": "file_scenario",
            "error": str(e),
            "success": False
        }})

# Then: Generate operation summary
total_file_ops = len([op for op in result["file_operations"] if op.get("operation") == "create_file"])
successful_file_ops = len([op for op in result["file_operations"] 
                          if op.get("operation") == "create_file" and op.get("success")])

valid_content = len([cv for cv in result["content_validation"] if cv.get("content_matches")])
total_content_tests = len(result["content_validation"])

result["operation_summary"] = {{
    "total_file_operations": total_file_ops,
    "successful_file_operations": successful_file_ops,
    "file_operation_success_rate": successful_file_ops / total_file_ops if total_file_ops > 0 else 0,
    "content_validation_tests": total_content_tests,
    "valid_content_results": valid_content,
    "content_validation_success_rate": valid_content / total_content_tests if total_content_tests > 0 else 0,
    "all_files_created": successful_file_ops == total_file_ops,
    "all_content_valid": valid_content == total_content_tests,
    "file_operations_functional": successful_file_ops > 0 and valid_content > 0
}}

print(f"File operations complete: {{successful_file_ops}}/{{total_file_ops}} files created, {{valid_content}}/{{total_content_tests}} content valid")
json.dumps(result)
        """

        # When: Executing file operations
        result = execute_python_code(code, timeout=filesystem_timeout)

        # Then: Validate file operations success
        validate_api_contract(result)
        assert result["success"], f"File operations failed: {result.get('error')}"

        file_results = json.loads(result["data"]["result"])
        
        # Validate operation summary
        summary = file_results["operation_summary"]
        assert summary["file_operations_functional"], "File operations should be functional"
        assert summary["all_files_created"], "All test files should be created successfully"
        assert summary["all_content_valid"], "All file content should be valid after read-back"
        
        # Validate content validation
        content_validation = file_results["content_validation"]
        assert len(content_validation) > 0, "Should have content validation results"
        for cv in content_validation:
            assert cv["content_matches"], f"Content should match for scenario: {cv['scenario']}"
        
        # Validate file properties
        file_properties = file_results["file_properties"]
        assert len(file_properties) > 0, "Should have file property results"
        for fp in file_properties:
            assert fp.get("is_file", False), f"Should be a file: {fp['scenario']}"
            assert fp.get("size", 0) > 0, f"File should have content: {fp['scenario']}"


@pytest.mark.filesystem
@pytest.mark.matplotlib
@pytest.mark.api
class TestVirtualFilesystemMatplotlibIntegration:
    """
    BDD tests for matplotlib plot saving and virtual filesystem integration.
    
    Migrated from: Original test_matplotlib_virtual_filesystem_plot_save
    Enhanced with: proper pytest structure, multiple save locations, comprehensive validation
    """

    def test_given_matplotlib_available_when_saving_plots_to_filesystem_then_plot_files_created(
        self, server_ready, plot_timeout, unique_timestamp, cleanup_test_directories
    ):
        """
        Test matplotlib plot saving to various virtual filesystem locations.

        **Given:** Matplotlib available and virtual filesystem writable locations
        **When:** Creating plots and saving to different filesystem paths
        **Then:** Plot files are created successfully with valid content

        Args:
            server_ready: Server readiness fixture
            plot_timeout: Timeout for plot operations
            unique_timestamp: Unique timestamp for test isolation
            cleanup_test_directories: Cleanup tracking list

        Example:
            Creates matplotlib plots and saves to /tmp, root, and custom
            directories, validates file creation, size, and accessibility.
        """
        # Given: Plot saving test scenarios with unique names
        min_file_size = VirtualFilesystemTestConfig.FILESYSTEM_SETTINGS["min_file_size"]
        test_dirs = [f"/test_plots_{unique_timestamp}", f"/tmp/test_plots_{unique_timestamp}"]
        for test_dir in test_dirs:
            cleanup_test_directories.append(test_dir)
        
        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import time

# When: Creating and saving matplotlib plots to virtual filesystem
result = {{
    "plot_creation": {{}},
    "save_operations": [],
    "file_validation": [],
    "filesystem_state": {{}},
    "summary": {{}}
}}

test_timestamp = {unique_timestamp}
min_expected_size = {min_file_size}

# Create test plot data
try:
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Basic line plot
    ax1.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    ax1.set_title('Sine Wave')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiple series
    ax2.plot(x, y1, 'b-', label='sin(x)')
    ax2.plot(x, y2, 'r--', label='cos(x)')
    ax2.plot(x, y3, 'g:', label='sin(x)*cos(x)')
    ax2.set_title('Multiple Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    scatter_x = np.random.randn(50)
    scatter_y = np.random.randn(50)
    ax3.scatter(scatter_x, scatter_y, alpha=0.6, c=np.arange(50), cmap='viridis')
    ax3.set_title('Scatter Plot')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram
    hist_data = np.random.normal(0, 1, 1000)
    ax4.hist(hist_data, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('Histogram')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Virtual Filesystem Plot Test - {{test_timestamp}}', fontsize=16)
    plt.tight_layout()
    
    result["plot_creation"] = {{
        "success": True,
        "plot_type": "comprehensive_multi_subplot",
        "subplots": 4,
        "figure_size": [12, 10]
    }}
    
except Exception as e:
    result["plot_creation"] = {{
        "success": False,
        "error": str(e)
    }}
    plt.close('all')

# Test saving to multiple filesystem locations
if result["plot_creation"]["success"]:
    save_locations = [
        {{
            "path": f"/tmp/vfs_test_{{test_timestamp}}.png",
            "description": "Temporary directory save"
        }},
        {{
            "path": f"/vfs_test_root_{{test_timestamp}}.png", 
            "description": "Root directory save"
        }},
        {{
            "path": f"/test_plots_{{test_timestamp}}/vfs_test_subdir.png",
            "description": "Custom directory save"
        }}
    ]
    
    for location in save_locations:
        file_path = Path(location["path"])
        description = location["description"]
        
        try:
            # Create parent directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            plt.savefig(file_path, 
                       dpi=150, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       format='png')
            
            # Validate file creation
            file_exists = file_path.exists()
            file_size = file_path.stat().st_size if file_exists else 0
            
            result["save_operations"].append({{
                "path": str(file_path),
                "description": description,
                "save_successful": True,
                "parent_created": file_path.parent.exists(),
                "file_exists": file_exists,
                "file_size": file_size,
                "size_adequate": file_size > min_expected_size
            }})
            
            # Additional file validation
            if file_exists:
                try:
                    result["file_validation"].append({{
                        "path": str(file_path),
                        "is_file": file_path.is_file(),
                        "is_readable": True,  # If we got size, it's readable
                        "filename": file_path.name,
                        "suffix": file_path.suffix,
                        "parent_directory": str(file_path.parent)
                    }})
                except Exception as validation_error:
                    result["file_validation"].append({{
                        "path": str(file_path),
                        "validation_error": str(validation_error)
                    }})
            
        except Exception as save_error:
            result["save_operations"].append({{
                "path": str(file_path),
                "description": description,
                "save_error": str(save_error),
                "save_successful": False
            }})
    
    plt.close('all')

# Then: Check final filesystem state
try:
    filesystem_checks = [
        Path("/tmp"),
        Path("/"),
        Path(f"/test_plots_{{test_timestamp}}")
    ]
    
    for check_path in filesystem_checks:
        if check_path.exists() and check_path.is_dir():
            contents = [p.name for p in check_path.iterdir() if 'vfs_test' in p.name]
            result["filesystem_state"][str(check_path)] = {{
                "exists": True,
                "test_files_found": contents,
                "test_file_count": len(contents)
            }}
        else:
            result["filesystem_state"][str(check_path)] = {{
                "exists": check_path.exists()
            }}

except Exception as fs_error:
    result["filesystem_state"] = {{"error": str(fs_error)}}

# Generate summary
successful_saves = sum(1 for op in result["save_operations"] if op.get("save_successful"))
total_saves = len(result["save_operations"])
valid_files = sum(1 for op in result["save_operations"] 
                 if op.get("file_exists") and op.get("size_adequate"))

result["summary"] = {{
    "plot_creation_successful": result["plot_creation"]["success"],
    "total_save_attempts": total_saves,
    "successful_saves": successful_saves,
    "valid_files_created": valid_files,
    "save_success_rate": successful_saves / total_saves if total_saves > 0 else 0,
    "file_validation_rate": valid_files / total_saves if total_saves > 0 else 0,
    "matplotlib_filesystem_functional": successful_saves > 0 and valid_files > 0,
    "all_saves_successful": successful_saves == total_saves
}}

print(f"Matplotlib filesystem test complete: {{successful_saves}}/{{total_saves}} saves successful, {{valid_files}} valid files")
json.dumps(result)
        """

        # When: Executing matplotlib plot saving operations
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate matplotlib filesystem integration
        validate_api_contract(result)
        assert result["success"], f"Matplotlib filesystem integration failed: {result.get('error')}"

        plot_results = json.loads(result["data"]["result"])
        
        # Validate plot creation
        plot_creation = plot_results["plot_creation"]
        assert plot_creation["success"], "Matplotlib plot creation should succeed"
        
        # Validate save operations
        summary = plot_results["summary"]
        assert summary["matplotlib_filesystem_functional"], "Matplotlib filesystem integration should be functional"
        assert summary["successful_saves"] > 0, "Should have at least one successful plot save"
        assert summary["valid_files_created"] > 0, "Should have created at least one valid plot file"
        
        # Validate individual save operations
        save_operations = plot_results["save_operations"]
        successful_saves = [op for op in save_operations if op.get("save_successful")]
        assert len(successful_saves) > 0, "Should have successful save operations"
        
        # Validate file properties
        for op in successful_saves:
            assert op["file_exists"], f"Saved plot file should exist: {op['path']}"
            assert op["size_adequate"], f"Plot file should be adequate size: {op['path']} ({op['file_size']} bytes)"

    def test_given_virtual_filesystem_when_creating_and_validating_plots_then_filesystem_integration_works(
        self, server_ready, api_timeout, unique_timestamp, cleanup_test_directories
    ):
        """
        Test comprehensive plot creation and filesystem integration validation.

        **Given:** Virtual filesystem with matplotlib capabilities
        **When:** Creating plots, saving them, and validating filesystem state
        **Then:** Filesystem operations succeed and integration works correctly

        Args:
            server_ready: Server readiness fixture
            api_timeout: Timeout for API operations
            unique_timestamp: Unique timestamp for test isolation
            cleanup_test_directories: Cleanup tracking list

        Example:
            Creates plot files in virtual filesystem, validates their creation,
            checks filesystem state, and ensures proper integration behavior.
        """
        # Given: Comprehensive filesystem integration test scenario
        test_plots_dir = f"/tmp/test_plots_{unique_timestamp}"
        cleanup_test_directories.append(test_plots_dir)
        
        integrated_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Combined setup and validation for filesystem integration
result = {{
    "setup_phase": {{}},
    "plot_creation": {{}},
    "filesystem_validation": {{}},
    "integration_test": {{}},
    "summary": {{}}
}}

# When: Testing complete integration workflow
test_timestamp = {unique_timestamp}
test_plots_dir = Path("/tmp/test_plots_{{test_timestamp}}")

try:
    # Phase 1: Setup and create test directory
    test_plots_dir.mkdir(parents=True, exist_ok=True)
    
    result["setup_phase"] = {{
        "directory_created": test_plots_dir.exists(),
        "directory_path": str(test_plots_dir),
        "directory_writable": test_plots_dir.is_dir() if test_plots_dir.exists() else False
    }}
    
    # Phase 2: Create and save plot
    plt.figure(figsize=(8, 6))
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
    plt.title('Filesystem Integration Test Plot')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True, alpha=0.3)
    
    plot_file = test_plots_dir / f"integration_test_{{test_timestamp}}.png"
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Validate immediate plot creation
    file_exists = plot_file.exists()
    file_size = plot_file.stat().st_size if file_exists else 0
    
    result["plot_creation"] = {{
        "plot_saved": True,
        "file_exists": file_exists,
        "file_size": file_size,
        "filename": str(plot_file),
        "file_has_content": file_size > 100  # Basic size check
    }}
    
    # Phase 3: Filesystem validation
    result["filesystem_validation"] = {{
        "test_directory_exists": test_plots_dir.exists(),
        "test_directory_accessible": test_plots_dir.is_dir() if test_plots_dir.exists() else False
    }}
    
    # Check directory contents
    if test_plots_dir.exists() and test_plots_dir.is_dir():
        try:
            contents = list(test_plots_dir.iterdir())
            plot_files = [f for f in contents if f.name.endswith('.png')]
            
            result["filesystem_validation"]["directory_contents"] = {{
                "total_files": len(contents),
                "plot_files": len(plot_files),
                "file_names": [f.name for f in contents]
            }}
        except Exception as list_error:
            result["filesystem_validation"]["listing_error"] = str(list_error)
    
    # Phase 4: Integration testing
    # Test basic filesystem operations
    temp_file = test_plots_dir / "temp_test.txt"
    try:
        temp_file.write_text("Integration test content")
        temp_content = temp_file.read_text()
        temp_file.unlink()  # Clean up
        
        result["integration_test"] = {{
            "file_write_read_successful": temp_content == "Integration test content",
            "file_cleanup_successful": not temp_file.exists(),
            "basic_operations_functional": True
        }}
    except Exception as integration_error:
        result["integration_test"] = {{
            "integration_error": str(integration_error),
            "basic_operations_functional": False
        }}
    
    # Phase 5: Generate summary
    directory_functional = result["filesystem_validation"].get("test_directory_accessible", False)
    plot_created = result["plot_creation"].get("file_exists", False)
    plot_has_content = result["plot_creation"].get("file_has_content", False)
    integration_works = result["integration_test"].get("basic_operations_functional", False)
    
    result["summary"] = {{
        "filesystem_functional": directory_functional,
        "plot_creation_successful": plot_created and plot_has_content,
        "integration_operations_work": integration_works,
        "overall_integration_success": directory_functional and plot_created and integration_works,
        "test_directory_accessible": directory_functional,
        "files_created": result["filesystem_validation"].get("directory_contents", {{}}).get("total_files", 0)
    }}

except Exception as e:
    result["setup_phase"] = {{"error": str(e)}}
    result["summary"] = {{"overall_error": str(e), "overall_integration_success": False}}

print(f"Integration test complete: {{result.get('summary', {{}}).get('files_created', 0)}} files created")
json.dumps(result)
        """

        # When: Executing integrated filesystem test
        integration_result = execute_python_code(integrated_code, timeout=api_timeout)

        # Then: Validate comprehensive integration
        validate_api_contract(integration_result)
        assert integration_result["success"], f"Integration test execution failed: {integration_result.get('error')}"

        integration_data = json.loads(integration_result["data"]["result"])
        
        # Validate integration results
        summary = integration_data["summary"]
        assert summary.get("filesystem_functional"), "Virtual filesystem should be functional"
        assert summary.get("plot_creation_successful"), "Plot creation should be successful"
        assert summary.get("integration_operations_work"), "Basic filesystem operations should work"
        
        # Validate setup phase
        setup_phase = integration_data["setup_phase"]
        assert setup_phase.get("directory_created"), "Test directory should be created"
        assert setup_phase.get("directory_writable"), "Test directory should be writable"
        
        # Validate plot creation
        plot_creation = integration_data["plot_creation"]
        assert plot_creation.get("plot_saved"), "Plot should be saved"
        assert plot_creation.get("file_exists"), "Plot file should exist"
        assert plot_creation.get("file_has_content"), "Plot file should have content"


if __name__ == "__main__":
    # Allow running this file directly for development/debugging
    pytest.main([__file__, "-v", "-s"])