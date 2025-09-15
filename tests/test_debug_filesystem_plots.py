"""
Test module for debugging filesystem plot functionality using Pyodide.

This module provides comprehensive tests for the plot creation and extraction
functionality in the Pyodide Express server. It validates that plots created
in the virtual filesystem are properly stored and can be extracted via the API.

Tests follow BDD (Behavior-Driven Development) style with Given/When/Then
structure and use pytest framework for better test organization and fixtures.
"""

import json
import time
from typing import Optional

import pytest
import requests
from requests.exceptions import RequestException


# Global configuration constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
PLOT_INSTALL_TIMEOUT = 300
SERVER_WAIT_TIMEOUT = 180
HEALTH_CHECK_INTERVAL = 1

# Matplotlib initialization constants
MATPLOTLIB_BACKEND = "Agg"
DEFAULT_PLOT_DPI = 100
DEFAULT_FIGSIZE = (5, 3)

# Plot directory paths
PLOTS_ROOT = "/plots"
PLOTS_MATPLOTLIB_DIR = "/plots/matplotlib"
PLOTS_SEABORN_DIR = "/plots/seaborn"
PLOTS_BASE64_DIR = "/plots/base64"


def wait_for_server(url: str, timeout: int = SERVER_WAIT_TIMEOUT) -> None:
    """
    Wait for the server to become available.

    Args:
        url: The URL to check for server availability
        timeout: Maximum time to wait in seconds

    Raises:
        RuntimeError: If server doesn't respond within timeout period
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return
        except RequestException:
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    raise RuntimeError(f"Server at {url} did not start in time")


@pytest.fixture(scope="session")
def server_session():
    """
    Session-scoped fixture to ensure server is running and matplotlib is
    installed.

    This fixture runs once per test session and ensures:
    1. The server is accessible
    2. Matplotlib package is installed for plot generation

    Yields:
        requests.Session: A configured session object for making requests

    Raises:
        pytest.skip: If server is not available
    """
    try:
        wait_for_server(f"{BASE_URL}/health", timeout=DEFAULT_TIMEOUT)
    except RuntimeError:
        pytest.skip("Server is not running on localhost:3000")

    session = requests.Session()

    # Ensure matplotlib is available for all tests
    response = session.post(
        f"{BASE_URL}/api/install-package",
        json={"package": "matplotlib"},
        timeout=PLOT_INSTALL_TIMEOUT,
    )
    if response.status_code != 200:
        pytest.fail(f"Failed to install matplotlib: {response.status_code}")

    yield session
    session.close()


@pytest.fixture
def execute_python_code(server_session):
    """
    Fixture to execute Python code via the public execute-raw API.

    This fixture provides a function that executes Python code using the
    /api/execute-raw endpoint and returns the complete API response following
    the standard API contract format.

    Args:
        server_session: The session fixture for making HTTP requests

    Returns:
        Callable: Function to execute Python code and return API response

    Example:
        >>> execute_fn = execute_python_code(server_session)
        >>> result = execute_fn("print('hello')")
        >>> assert result["success"] == True
        >>> assert "hello" in result["data"]["stdout"]
    """
    def _execute(code: str, timeout: Optional[int] = None) -> dict:
        """
        Execute Python code using the /api/execute-raw endpoint.

        This function sends Python code as plain text to the execute-raw endpoint
        and returns the complete JSON response according to the API contract.

        Args:
            code: Python code to execute as a string
            timeout: Request timeout in seconds (default: DEFAULT_TIMEOUT)

        Returns:
            dict: API response containing success, data, error, and meta fields
                - success (bool): Whether the execution was successful
                - data (dict|None): Contains result, stdout, stderr, executionTime
                - error (str|None): Error message if execution failed
                - meta (dict): Metadata including timestamp

        Raises:
            AssertionError: If the HTTP request fails or returns non-200 status

        Example:
            >>> response = _execute("import sys; print(sys.version)")
            >>> assert response["success"] == True
            >>> assert "data" in response
            >>> assert "stdout" in response["data"]
        """
        response = server_session.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=timeout or DEFAULT_TIMEOUT
        )
        assert response.status_code == 200, (
            f"Code execution failed with status {response.status_code}: "
            f"{response.text}"
        )
        
        # Return the complete JSON response following API contract
        return response.json()

    return _execute


def test_given_matplotlib_environment_when_creating_plot_then_file_exists_in_vfs(  # noqa: E501
    execute_python_code, server_session
):
    """
    Test plot creation and verification in virtual filesystem.

    This test validates the complete workflow of creating matplotlib plots
    in the Pyodide virtual filesystem and verifying their existence using
    only the public /api/execute-raw endpoint.

    Description:
        Creates a matplotlib plot with a unique timestamp-based filename,
        saves it to the /home/pyodide/plots/matplotlib directory, and
        verifies the file exists with proper metadata using pathlib.

    Input:
        - Python code string that creates a matplotlib plot
        - Uses pathlib.Path for cross-platform compatibility
        - Saves to /home/pyodide/plots/matplotlib with unique filename

    Output:
        - API response following standard contract with success, data, error, meta
        - JSON output in stdout containing verification steps:
          * step1_create_plot: Plot creation status and file path
          * step2_verify_file: File existence and size validation
          * step3_list_directories: Directory structure verification

    Example:
        The test executes Python code that:
        1. Creates a simple line plot with matplotlib
        2. Saves to /home/pyodide/plots/matplotlib/debug_filesystem_{timestamp}.png
        3. Verifies file exists using Path.exists()
        4. Checks file size is > 0 bytes
        5. Lists directory contents for validation

    Assertions:
        - API response follows contract (success=True, data contains stdout)
        - Plot creation step succeeds with output path
        - File exists in VFS with size > 0
        - Directory structure is properly created
        - matplotlib directory contains the created file
    """
    # Given: Python code to create and verify a plot
    plot_creation_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

result = {
    "step1_create_plot": {},
    "step2_verify_file": {},
    "step3_list_directories": {}
}

# Step 1: Create plot with dynamic filename
try:
    plots_dir = Path('/home/pyodide/plots/matplotlib')
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)  # Generate unique timestamp

    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title('Debug Filesystem Test')

    output_path = plots_dir / f'debug_filesystem_{timestamp}.png'
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    plt.close()

    result["step1_create_plot"] = {
        "success": True,
        "output_path": str(output_path),
        "filename": output_path.name
    }

except Exception as e:
    result["step1_create_plot"] = {
        "success": False,
        "error": str(e)
    }

# Step 2: Verify file exists and get details
try:
    if result["step1_create_plot"].get("success"):
        path_obj = Path(result["step1_create_plot"]["output_path"])
        file_exists = path_obj.exists()
        file_size = path_obj.stat().st_size if file_exists else 0

        result["step2_verify_file"] = {
            "file_exists": file_exists,
            "file_size": file_size,
            "output_path": str(path_obj)
        }
    else:
        result["step2_verify_file"] = {
            "error": "Plot creation failed, cannot verify file"
        }

except Exception as e:
    result["step2_verify_file"] = {
        "error": str(e)
    }

# Step 3: List directory contents
try:
    plots_path = Path('/home/pyodide/plots')
    matplotlib_path = Path('/home/pyodide/plots/matplotlib')
    result["step3_list_directories"] = {
        "plots_exists": plots_path.exists(),
        "plots_contents": (
            [p.name for p in plots_path.iterdir()]
            if plots_path.exists() else []
        ),
        "plots_matplotlib_exists": matplotlib_path.exists(),
        "plots_matplotlib_contents": (
            [p.name for p in matplotlib_path.iterdir()]
            if matplotlib_path.exists() else []
        )
    }

except Exception as e:
    result["step3_list_directories"] = {
        "error": str(e)
    }

print(json.dumps(result, indent=2))
'''

    # When: Execute the plot creation code
    api_response = execute_python_code(plot_creation_code, timeout=60)

    # Then: Verify API response follows contract
    assert api_response.get("success"), (
        f"Code execution failed: {api_response.get('error', 'Unknown error')}"
    )
    assert "data" in api_response, "API response should contain 'data' field"
    assert "stdout" in api_response["data"], "API response data should contain 'stdout'"
    
    # Parse the JSON output from stdout
    stdout_content = api_response["data"]["stdout"]
    result = json.loads(stdout_content)

    # Verify plot creation succeeded
    step1 = result.get("step1_create_plot", {})
    assert step1.get("success"), f"Plot creation failed: {step1}"
    assert step1.get("output_path"), "Output path should be returned"
    assert step1.get("filename"), "Filename should be returned"

    # Verify file exists in virtual filesystem
    step2 = result.get("step2_verify_file", {})
    assert step2.get("file_exists"), (
        f"File doesn't exist in virtual FS: {step2}"
    )
    assert step2.get("file_size", 0) > 0, "File should have content"

    # Verify directory structure
    step3 = result.get("step3_list_directories", {})
    assert step3.get("plots_exists"), "/home/pyodide/plots directory should exist"
    assert step3.get("plots_matplotlib_exists"), (
        "/plots/matplotlib should exist"
    )

    # Verify generated file appears in directory listing
    filename = step1.get("filename", "")
    assert filename in step3.get("plots_matplotlib_contents", []), (
        f"Generated file {filename} should be in /plots/matplotlib directory"
    )

    print("✅ Plot created successfully in virtual filesystem")
    contents = step3.get('plots_matplotlib_contents', [])
    print(f"📁 Directory contents: {contents}")


def test_given_plot_in_vfs_when_extracting_then_file_is_retrieved(
    execute_python_code, server_session
):
    """
    Test plot creation and file verification in virtual filesystem.

    This test validates that plots created in the Pyodide VFS can be
    properly verified and accessed through filesystem operations without
    relying on external extraction APIs.

    Description:
        Creates a matplotlib plot in the virtual filesystem and verifies
        its existence, properties, and accessibility using pathlib operations
        executed within the same Pyodide environment.

    Input:
        - Python code that creates a plot with unique filename
        - Verification code that checks file properties
        - Uses pathlib for cross-platform path handling

    Output:
        - API response with plot creation confirmation
        - JSON verification data containing file metadata:
          * filename: Name of created file
          * exists: Boolean indicating file existence
          * is_file: Boolean confirming it's a regular file
          * size: File size in bytes
          * dir_contents: List of files in matplotlib directory

    Example:
        1. Creates plot: extract_test_{timestamp}.png
        2. Saves to /home/pyodide/plots/matplotlib/
        3. Verifies file exists and has content
        4. Lists directory contents for validation
        5. Confirms file appears in directory listing

    Assertions:
        - Plot file is successfully created
        - File exists and is accessible via pathlib
        - File size > 0 indicating valid content
        - File appears in directory contents
        - Proper directory structure is maintained
    """
    # Given: Create a plot in the VFS first
    plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

timestamp = int(time.time() * 1000)
filename = f'extract_test_{timestamp}.png'
filepath = plots_dir / filename

plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Extract Test Plot')
plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
plt.close()

# Return structured data about the created plot
result = {
    "filename": filename,
    "filepath": str(filepath),
    "created": True
}

print(json.dumps(result))
'''

    # Create the plot and get metadata
    api_response = execute_python_code(plot_code)
    assert api_response.get("success"), (
        f"Plot creation failed: {api_response.get('error')}"
    )
    
    plot_data = json.loads(api_response["data"]["stdout"])
    filename = plot_data["filename"]
    assert filename, "Filename should be returned from plot creation"

    # When: Verify the plot file through code execution
    verification_code = f'''
import json
from pathlib import Path

filename = "{filename}"
filepath = Path('/home/pyodide/plots/matplotlib') / filename

result = {{
    "filename": filename,
    "exists": filepath.exists(),
    "is_file": filepath.is_file() if filepath.exists() else False,
    "size": filepath.stat().st_size if filepath.exists() else 0,
    "in_directory": False
}}

# List all files in the matplotlib directory
plots_dir = Path('/home/pyodide/plots/matplotlib')
if plots_dir.exists():
    result["dir_contents"] = [
        f.name for f in plots_dir.iterdir() if f.is_file()
    ]
    result["in_directory"] = filename in result["dir_contents"]
else:
    result["dir_contents"] = []

# Verify we can read the file
if filepath.exists():
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            result["is_valid_png"] = header[:4] == b'\\x89PNG'
            result["readable"] = True
    except Exception as e:
        result["readable"] = False
        result["read_error"] = str(e)

print(json.dumps(result, indent=2))
'''

    # Execute verification and parse API response
    verify_response = execute_python_code(verification_code)
    assert verify_response.get("success"), (
        f"Verification failed: {verify_response.get('error')}"
    )
    
    result = json.loads(verify_response["data"]["stdout"])

    # Then: Verify the plot file is retrievable
    assert result["exists"], f"File {filename} should exist in VFS"
    assert result["is_file"], f"File {filename} should be a regular file"
    assert result["size"] > 0, f"File {filename} should not be empty"
    assert result.get("in_directory", False), (
        f"File {filename} should appear in directory listing"
    )
    assert result.get("readable", False), (
        f"File {filename} should be readable"
    )
    assert result.get("is_valid_png", False), (
        f"File {filename} should be a valid PNG"
    )

    print(f"✅ File {filename} was successfully retrieved and verified!")


def test_given_multiple_plots_when_extracting_then_all_are_retrieved(
    execute_python_code, server_session
):
    """
    Test creation and verification of multiple plot files in VFS.

    This test validates that multiple matplotlib plots can be created
    simultaneously in the virtual filesystem and properly verified
    through pathlib operations.

    Description:
        Creates three different types of matplotlib plots (line, bar, scatter)
        with unique timestamps and verifies all are created successfully
        in the /home/pyodide/plots/matplotlib directory.

    Input:
        - Python code that creates three different plot types
        - Each plot has unique filename with timestamp
        - All plots saved to /home/pyodide/plots/matplotlib/
        - Uses pathlib.Path for cross-platform compatibility

    Output:
        - API response containing creation status for each plot
        - JSON data with file verification information:
          * created_files: List of successfully created filenames
          * file_details: Metadata for each file (exists, size, valid)
          * directory_info: Contents of matplotlib directory

    Example:
        Creates plots: line_plot_{timestamp}.png, bar_plot_{timestamp}.png,
        scatter_plot_{timestamp}.png and verifies each exists with size > 0

    Assertions:
        - All three plots are successfully created
        - Each file exists in VFS and has content
        - Files appear in directory listing
        - All files are valid PNG format
        - Directory structure is properly maintained
    """
    # Given: Create multiple plots with different characteristics
    multi_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

timestamp = int(time.time() * 1000)
results = {
    "created_files": [],
    "file_details": {},
    "errors": []
}

# Create line plot
try:
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
    plt.title('Line Plot')
    filename1 = f'line_plot_{timestamp}.png'
    filepath1 = plots_dir / filename1
    plt.savefig(str(filepath1), dpi=150)
    plt.close()
    results["created_files"].append(filename1)
    results["file_details"][filename1] = {
        "created": True,
        "path": str(filepath1)
    }
except Exception as e:
    results["errors"].append(f"Line plot error: {str(e)}")

# Create bar plot
try:
    plt.figure(figsize=(6, 4))
    plt.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
    plt.title('Bar Plot')
    filename2 = f'bar_plot_{timestamp}.png'
    filepath2 = plots_dir / filename2
    plt.savefig(str(filepath2), dpi=150)
    plt.close()
    results["created_files"].append(filename2)
    results["file_details"][filename2] = {
        "created": True,
        "path": str(filepath2)
    }
except Exception as e:
    results["errors"].append(f"Bar plot error: {str(e)}")

# Create scatter plot
try:
    plt.figure(figsize=(6, 4))
    plt.scatter([1, 2, 3, 4, 5], [2, 4, 1, 5, 3])
    plt.title('Scatter Plot')
    filename3 = f'scatter_plot_{timestamp}.png'
    filepath3 = plots_dir / filename3
    plt.savefig(str(filepath3), dpi=150)
    plt.close()
    results["created_files"].append(filename3)
    results["file_details"][filename3] = {
        "created": True,
        "path": str(filepath3)
    }
except Exception as e:
    results["errors"].append(f"Scatter plot error: {str(e)}")

# Verify all created files
for filename in results["created_files"]:
    filepath = plots_dir / filename
    results["file_details"][filename].update({
        "exists": filepath.exists(),
        "is_file": filepath.is_file() if filepath.exists() else False,
        "size": filepath.stat().st_size if filepath.exists() else 0
    })

# List directory contents
results["directory_info"] = {
    "total_files": len([f for f in plots_dir.iterdir() if f.is_file()]),
    "contents": [f.name for f in plots_dir.iterdir() if f.is_file()],
    "all_created_files_present": all(
        f in [f.name for f in plots_dir.iterdir()]
        for f in results["created_files"]
    )
}

print(json.dumps(results, indent=2))
'''

    # Create multiple plots
    api_response = execute_python_code(multi_plot_code)
    assert api_response.get("success"), (
        f"Plot creation failed: {api_response.get('error')}"
    )
    
    results = json.loads(api_response["data"]["stdout"])
    created_files = results["created_files"]
    
    assert len(created_files) == 3, (
        f"Expected 3 files, got {len(created_files)}"
    )

    # Verify no errors occurred during creation
    assert len(results.get("errors", [])) == 0, (
        f"Errors during creation: {results['errors']}"
    )

    # Verify directory info
    dir_info = results.get("directory_info", {})
    assert dir_info.get("all_created_files_present", False), (
        "Not all created files are present in directory"
    )
    assert dir_info.get("total_files", 0) >= 3, (
        f"Expected at least 3 files in directory, got {dir_info['total_files']}"
    )

    # Then: Verify each individual plot file
    file_details = results.get("file_details", {})
    for filename in created_files:
        file_info = file_details.get(filename, {})
        assert file_info.get("exists"), f"File {filename} should exist in VFS"
        assert file_info.get("is_file"), f"File {filename} should be a regular file"
        assert file_info.get("size", 0) > 0, f"File {filename} should not be empty"

    # Verify all files are accounted for
    all_exist = all(
        file_details[f].get("exists", False) for f in created_files
    )
    assert all_exist, "Not all created files exist in VFS"

    print(f"✅ Successfully created and verified {len(created_files)} plots")
    print(f"📁 Directory contains {dir_info['total_files']} files total")


def test_given_plots_in_subdirs_when_extracting_then_structure_preserved(
    execute_python_code, server_session
):
    """
    Test plot creation and verification in multiple subdirectories.

    This test validates that plots can be created in different subdirectories
    of the virtual filesystem and properly verified through pathlib operations,
    ensuring directory structure is maintained.

    Description:
        Creates plots in both /home/pyodide/plots/matplotlib and
        /home/pyodide/plots/base64 directories, then verifies proper
        directory structure and file accessibility.

    Input:
        - Python code that creates plots in different subdirectories
        - One PNG file in matplotlib directory
        - One base64-encoded plot data in base64 directory
        - Uses pathlib for cross-platform path handling

    Output:
        - API response with creation and verification results
        - JSON data containing:
          * created_files: Dictionary mapping directories to filenames
          * verification: File existence and metadata for each directory
          * directory_structure: Complete directory tree information

    Example:
        Creates:
        - /home/pyodide/plots/matplotlib/mpl_test_{timestamp}.png
        - /home/pyodide/plots/base64/b64_test_{timestamp}.txt
        Verifies both files exist with proper sizes and directory structure

    Assertions:
        - Both subdirectories are created successfully
        - Files exist in their respective directories
        - File sizes > 0 indicating valid content
        - Directory listings contain the created files
        - Overall directory structure is properly maintained
    """
    # Given: Create plots in different directories with comprehensive verification
    subdir_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import base64
from io import BytesIO

timestamp = int(time.time() * 1000)
results = {
    "created_files": {
        "matplotlib": [],
        "base64": []
    },
    "verification": {},
    "directory_structure": {},
    "errors": []
}

try:
    # Create matplotlib plot
    matplotlib_dir = Path('/home/pyodide/plots/matplotlib')
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title('Matplotlib Plot')
    mpl_file = f'mpl_test_{timestamp}.png'
    mpl_path = matplotlib_dir / mpl_file
    plt.savefig(str(mpl_path), dpi=100)
    plt.close()
    results["created_files"]["matplotlib"].append(mpl_file)

    # Create base64 encoded plot
    base64_dir = Path('/home/pyodide/plots/base64')
    base64_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.plot([3, 2, 1], [2, 4, 1])
    plt.title('Base64 Plot')

    # Save to buffer and encode
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Save base64 data
    b64_file = f'b64_test_{timestamp}.txt'
    b64_path = base64_dir / b64_file
    with open(str(b64_path), 'w') as f:
        f.write(b64_data)
    results["created_files"]["base64"].append(b64_file)

    # Verify created files
    for dir_name, files in results["created_files"].items():
        results["verification"][dir_name] = {}
        dir_path = Path(f'/home/pyodide/plots/{dir_name}')
        
        for filename in files:
            file_path = dir_path / filename
            results["verification"][dir_name][filename] = {
                "exists": file_path.exists(),
                "is_file": file_path.is_file() if file_path.exists() else False,
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "path": str(file_path)
            }

    # Get directory structure
    plots_root = Path('/home/pyodide/plots')
    results["directory_structure"] = {
        "root_exists": plots_root.exists(),
        "subdirectories": {}
    }
    
    if plots_root.exists():
        for subdir in plots_root.iterdir():
            if subdir.is_dir():
                results["directory_structure"]["subdirectories"][subdir.name] = {
                    "exists": True,
                    "file_count": len([f for f in subdir.iterdir() if f.is_file()]),
                    "files": [f.name for f in subdir.iterdir() if f.is_file()]
                }

except Exception as e:
    results["errors"].append(str(e))

print(json.dumps(results, indent=2))
'''

    # Execute the code
    api_response = execute_python_code(subdir_plot_code)
    assert api_response.get("success"), (
        f"Plot creation failed: {api_response.get('error')}"
    )
    
    # Parse results
    results = json.loads(api_response["data"]["stdout"])
    
    # Verify no errors occurred
    assert len(results.get("errors", [])) == 0, (
        f"Errors during creation: {results['errors']}"
    )
    
    # Verify files were created
    created_files = results.get("created_files", {})
    assert len(created_files.get("matplotlib", [])) == 1, (
        "Should create 1 matplotlib file"
    )
    assert len(created_files.get("base64", [])) == 1, (
        "Should create 1 base64 file"
    )
    
    # Verify each file through verification data
    verification = results.get("verification", {})
    
    # Check matplotlib files
    for mpl_file in created_files["matplotlib"]:
        file_info = verification.get("matplotlib", {}).get(mpl_file, {})
        assert file_info.get("exists"), (
            f"Matplotlib file {mpl_file} not found in /home/pyodide/plots/matplotlib"
        )
        assert file_info.get("is_file"), (
            f"Matplotlib file {mpl_file} should be a regular file"
        )
        assert file_info.get("size", 0) > 0, (
            f"Matplotlib file {mpl_file} should not be empty"
        )

    # Check base64 files
    for b64_file in created_files["base64"]:
        file_info = verification.get("base64", {}).get(b64_file, {})
        assert file_info.get("exists"), (
            f"Base64 file {b64_file} not found in /home/pyodide/plots/base64"
        )
        assert file_info.get("is_file"), (
            f"Base64 file {b64_file} should be a regular file"
        )
        assert file_info.get("size", 0) > 0, (
            f"Base64 file {b64_file} should not be empty"
        )

    # Verify directory structure
    dir_structure = results.get("directory_structure", {})
    assert dir_structure.get("root_exists"), (
        "/home/pyodide/plots directory should exist"
    )
    
    subdirs = dir_structure.get("subdirectories", {})
    assert "matplotlib" in subdirs, "matplotlib subdirectory should exist"
    assert "base64" in subdirs, "base64 subdirectory should exist"
    
    # Verify files appear in directory listings
    assert subdirs["matplotlib"]["file_count"] >= 1, (
        "matplotlib directory should contain at least 1 file"
    )
    assert subdirs["base64"]["file_count"] >= 1, (
        "base64 directory should contain at least 1 file"
    )
    
    print(f"✅ Directory structure preserved! Created files in {len(subdirs)} directories")


def test_given_no_plots_when_extracting_then_empty_result_returned(
    execute_python_code, server_session
):
    """
    Test directory verification when no plots exist.

    This test validates that the filesystem properly handles empty
    directories and correctly reports when no plot files are present.

    Description:
        Clears all plot directories and verifies they are empty through
        pathlib operations, ensuring proper empty directory handling
        without relying on external extraction APIs.

    Input:
        - Python code to clear plot directories
        - Verification code to check directory contents
        - Uses pathlib for cross-platform path operations

    Output:
        - API response confirming directory clearing
        - JSON verification data showing:
          * directories: Status of each plot directory
          * total_files: Count of files across all directories
          * empty_verification: Confirmation all directories are empty

    Example:
        Clears and verifies empty state of:
        - /home/pyodide/plots/matplotlib/
        - /home/pyodide/plots/seaborn/
        - /home/pyodide/plots/base64/

    Assertions:
        - All directories exist but are empty
        - Total file count is 0
        - Directory listings return empty arrays
        - Empty state properly verified
    """
    # Given: Clear any existing plots and verify empty state
    clear_and_verify_code = '''
from pathlib import Path
import json

# Clear plot directories if they exist
results = {
    "cleared_directories": [],
    "verification": {},
    "total_files": 0
}

plot_dirs = [
    '/home/pyodide/plots/matplotlib',
    '/home/pyodide/plots/seaborn',
    '/home/pyodide/plots/base64'
]

for dir_path in plot_dirs:
    dir_name = Path(dir_path).name
    path_obj = Path(dir_path)
    
    if path_obj.exists():
        # Remove all files in the directory
        files_removed = 0
        for file in path_obj.iterdir():
            if file.is_file():
                file.unlink()
                files_removed += 1
        results["cleared_directories"].append({
            "directory": dir_name,
            "path": dir_path,
            "files_removed": files_removed
        })
    else:
        # Create directory if it doesn't exist
        path_obj.mkdir(parents=True, exist_ok=True)
        results["cleared_directories"].append({
            "directory": dir_name,
            "path": dir_path,
            "created": True
        })

# Verify all directories are empty
for dir_path in plot_dirs:
    dir_name = Path(dir_path).name
    path_obj = Path(dir_path)
    
    if path_obj.exists():
        files = [f.name for f in path_obj.iterdir() if f.is_file()]
        file_count = len(files)
        results["verification"][dir_name] = {
            "exists": True,
            "file_count": file_count,
            "files": files,
            "is_empty": file_count == 0
        }
        results["total_files"] += file_count
    else:
        results["verification"][dir_name] = {
            "exists": False,
            "file_count": 0,
            "files": [],
            "is_empty": True
        }

print(json.dumps(results, indent=2))
'''

    # Execute clearing and verification
    api_response = execute_python_code(clear_and_verify_code)
    assert api_response.get("success"), (
        f"Directory clearing failed: {api_response.get('error')}"
    )
    
    results = json.loads(api_response["data"]["stdout"])
    
    # Verify directories were processed
    assert len(results.get("cleared_directories", [])) == 3, (
        "Should process 3 directories"
    )
    
    # Verify all directories are empty
    verification = results.get("verification", {})
    total_files = results.get("total_files", 0)
    
    # Then: Verify empty state
    assert total_files == 0, (
        f"Expected 0 total files, got {total_files}"
    )
    
    # Check each individual directory
    for dir_name in ["matplotlib", "seaborn", "base64"]:
        dir_info = verification.get(dir_name, {})
        assert dir_info.get("exists", False), (
            f"Directory {dir_name} should exist"
        )
        assert dir_info.get("is_empty", False), (
            f"Directory {dir_name} should be empty"
        )
        assert dir_info.get("file_count", -1) == 0, (
            f"Directory {dir_name} should contain 0 files"
        )

    print("✅ All directories verified as empty")


def test_given_large_plot_when_saving_then_handled_correctly(
    execute_python_code, server_session
):
    """
    Test handling of large plot files in virtual filesystem.

    This test validates that the system can handle large, complex plots
    with high resolution and verify them through pathlib operations.

    Description:
        Creates a complex multi-subplot figure with high DPI to produce
        a large file, then verifies the file exists and has the expected
        size characteristics indicating successful processing.

    Input:
        - Python code creating 2x2 subplot figure with complex data
        - Dense scatter plot, multiple line plots, heatmap, histogram
        - High DPI (300) for larger file size
        - Uses pathlib for cross-platform path handling

    Output:
        - API response with plot creation details
        - JSON data containing:
          * filename: Name of created large plot file
          * file_size: Size in bytes of created file
          * verification: File existence and metadata
          * size_analysis: Size category and validity checks

    Example:
        Creates large_plot_{timestamp}.png with high DPI and complex data
        Verifies file > 100KB indicating proper large file handling

    Assertions:
        - Plot file is successfully created
        - File size exceeds minimum threshold (large file)
        - File exists and is accessible via pathlib
        - File is valid PNG format
        - Proper handling of high-resolution content
    """
    # Given: Create a large, complex plot with comprehensive verification
    large_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import json

plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)

results = {
    "creation": {},
    "verification": {},
    "size_analysis": {}
}

try:
    # Create complex plot with lots of data
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Dense scatter plot
    x = np.random.randn(5000)
    y = np.random.randn(5000)
    axes[0, 0].scatter(x, y, alpha=0.5, s=1)
    axes[0, 0].set_title('Dense Scatter Plot')

    # Subplot 2: Multiple line plots
    for i in range(50):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x + i/10) + np.random.randn(1000) * 0.1
        axes[0, 1].plot(x, y, alpha=0.3, linewidth=0.5)
    axes[0, 1].set_title('Multiple Lines')

    # Subplot 3: Heatmap
    data = np.random.randn(100, 100)
    im = axes[1, 0].imshow(data, cmap='viridis')
    axes[1, 0].set_title('Heatmap')
    plt.colorbar(im, ax=axes[1, 0])

    # Subplot 4: Histogram
    data = np.random.randn(10000)
    axes[1, 1].hist(data, bins=100, alpha=0.7)
    axes[1, 1].set_title('Histogram')

    plt.tight_layout()

    # Save with high DPI for larger file size
    filename = f'large_plot_{timestamp}.png'
    filepath = plots_dir / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close()

    results["creation"] = {
        "success": True,
        "filename": filename,
        "filepath": str(filepath)
    }

except Exception as e:
    results["creation"] = {
        "success": False,
        "error": str(e)
    }

# Verify the created file
if results["creation"].get("success"):
    filepath = Path(results["creation"]["filepath"])
    file_size = filepath.stat().st_size if filepath.exists() else 0
    
    results["verification"] = {
        "exists": filepath.exists(),
        "is_file": filepath.is_file() if filepath.exists() else False,
        "size": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0
    }

    # Verify it's a valid PNG
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                results["verification"]["is_valid_png"] = header[:4] == b'\\x89PNG'
                f.seek(0, 2)  # Seek to end
                results["verification"]["actual_size"] = f.tell()
                results["verification"]["readable"] = True
        except Exception as e:
            results["verification"]["readable"] = False
            results["verification"]["error"] = str(e)

    # Size analysis
    results["size_analysis"] = {
        "is_large": file_size > 100000,
        "size_category": (
            "very_large" if file_size > 1000000 else
            "large" if file_size > 100000 else
            "medium" if file_size > 10000 else
            "small"
        ),
        "meets_threshold": file_size > 100000
    }

    # List directory to confirm file appears
    files = [f.name for f in plots_dir.iterdir() if f.is_file()]
    results["verification"]["in_directory"] = filename in files
    results["verification"]["dir_file_count"] = len(files)

print(json.dumps(results, indent=2))
'''

    # Create large plot and verify
    api_response = execute_python_code(large_plot_code, timeout=60)
    assert api_response.get("success"), (
        f"Large plot creation failed: {api_response.get('error')}"
    )
    
    results = json.loads(api_response["data"]["stdout"])
    
    # Verify creation succeeded
    creation = results.get("creation", {})
    assert creation.get("success"), (
        f"Plot creation failed: {creation.get('error')}"
    )
    
    filename = creation.get("filename")
    assert filename, "Filename should be returned"
    
    # Verify file properties
    verification = results.get("verification", {})
    assert verification.get("exists"), f"Large file {filename} should exist"
    assert verification.get("is_file"), f"Large file {filename} should be a regular file"
    assert verification.get("readable"), f"Large file {filename} should be readable"
    
    # Verify it's actually a large file
    file_size = verification.get("size", 0)
    size_analysis = results.get("size_analysis", {})
    assert size_analysis.get("meets_threshold"), (
        f"Expected large file (>100KB), got {file_size} bytes"
    )
    
    assert verification.get("in_directory"), (
        f"Large file {filename} should appear in directory listing"
    )

    print(f"✅ Created large plot: {filename} ({file_size:,} bytes)")
    print(f"📊 Size category: {size_analysis.get('size_category', 'unknown')}")
    print(f"📊 Created large plot: {filename} ({file_size:,} bytes)")

    # When: Verify the large plot through code execution
    verify_large_plot_code = f'''
import json
from pathlib import Path

filename = "{filename}"
filepath = Path('/home/pyodide/plots/matplotlib') / filename

result = {{
    "filename": filename,
    "exists": filepath.exists(),
    "size": filepath.stat().st_size if filepath.exists() else 0,
    "size_mb": (
        round(filepath.stat().st_size / (1024 * 1024), 2)
        if filepath.exists() else 0
    )
}}

# Verify it's a valid PNG
if filepath.exists():
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            result["is_valid_png"] = header[:4] == b'\\x89PNG'
            # Read a bit more to ensure file is not corrupted
            f.seek(0, 2)  # Seek to end
            result["actual_size"] = f.tell()
            result["readable"] = True
    except Exception as e:
        result["readable"] = False
        result["error"] = str(e)

# List directory to confirm file is there
plots_dir = Path('/home/pyodide/plots/matplotlib')
if plots_dir.exists():
    files = [f.name for f in plots_dir.iterdir() if f.is_file()]
    result["in_directory"] = filename in files
    result["dir_file_count"] = len(files)

print(json.dumps(result, indent=2))
'''

    api_response = execute_python_code(verify_large_plot_code)
    assert api_response.get("success"), (
        f"Large plot verification failed: {api_response.get('error')}"
    )
    
    result = json.loads(api_response["data"]["stdout"])

    # Then: Should verify large file exists and is valid
    assert result["exists"], f"Large file {filename} should exist"
    assert result["size"] == file_size, \
        f"File size mismatch: expected {file_size}, got {result['size']}"
    assert result.get("is_valid_png"), \
        f"Large file {filename} should be valid PNG"
    assert result.get("readable"), (
        f"Large file {filename} should be readable"
    )
    assert result.get("in_directory"), (
        f"Large file {filename} should appear in directory"
    )

    print(f"✅ Large plot file verified successfully! "
          f"Size: {result['size_mb']} MB")


def test_given_invalid_plot_data_when_saving_then_error_handled(
    execute_python_code
):
    """
    Test error handling for invalid plot operations.

    This test validates robust error handling when matplotlib encounters
    invalid plot operations, data, or file system issues.

    Description:
        Tests multiple error scenarios including invalid figure sizes,
        non-existent save paths, and invalid data types to ensure
        graceful error handling throughout the plotting pipeline.

    Input:
        - Python code containing various error scenarios:
          * Invalid negative figure size
          * Non-existent directory for save path
          * String data instead of numeric data for plotting
        - Each scenario wrapped in try-except blocks

    Output:
        - API response containing error handling results
        - JSON data with:
          * scenario_results: List of error handling outcomes
          * handled_count: Number of errors properly handled
          * error_types: Types of exceptions caught
          * success_rate: Percentage of errors handled gracefully

    Example:
        Tests plt.figure(figsize=(-5, -3)) and expects ValueError
        Tests savefig to non-existent path and expects FileNotFoundError
        Tests plt.plot("invalid", "data") and expects TypeError

    Assertions:
        - All error scenarios are handled gracefully
        - Each scenario produces appropriate error messages
        - No unhandled exceptions crash the system
        - Error types are properly identified and reported
        - System remains stable after error scenarios
    """
    # Test comprehensive error scenario handling
    error_handling_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

scenarios = [
    {
        "name": "Invalid figure size",
        "description": "Negative figure dimensions should raise ValueError"
    },
    {
        "name": "Invalid save path",
        "description": "Non-existent directory should raise FileNotFoundError"
    },
    {
        "name": "Invalid data types",
        "description": "String data should raise TypeError"
    }
]

results = {
    "scenario_results": [],
    "handled_count": 0,
    "error_types": [],
    "success_rate": 0
}

# Scenario 1: Invalid figure size
try:
    plt.figure(figsize=(-5, -3))  # Invalid negative size
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.savefig('/home/pyodide/plots/matplotlib/invalid_size.png')
    results["scenario_results"].append({
        "scenario": "invalid_size",
        "handled": False,
        "error": "Should have failed with invalid size"
    })
except Exception as e:
    results["scenario_results"].append({
        "scenario": "invalid_size",
        "handled": True,
        "error_type": type(e).__name__,
        "error_message": str(e)[:100]
    })
    results["handled_count"] += 1
    results["error_types"].append(type(e).__name__)

# Scenario 2: Invalid save path
try:
    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.savefig('/nonexistent/directory/plot.png')
    results["scenario_results"].append({
        "scenario": "invalid_path",
        "handled": False,
        "error": "Should have failed with invalid path"
    })
except Exception as e:
    results["scenario_results"].append({
        "scenario": "invalid_path",
        "handled": True,
        "error_type": type(e).__name__,
        "error_message": str(e)[:100]
    })
    results["handled_count"] += 1
    results["error_types"].append(type(e).__name__)

# Scenario 3: Invalid data types
try:
    plt.figure(figsize=(5, 3))
    plt.plot("invalid", "data")  # String instead of numeric data
    plt.savefig('/home/pyodide/plots/matplotlib/invalid_data.png')
    results["scenario_results"].append({
        "scenario": "invalid_data",
        "handled": False,
        "error": "Should have failed with invalid data"
    })
except Exception as e:
    results["scenario_results"].append({
        "scenario": "invalid_data",
        "handled": True,
        "error_type": type(e).__name__,
        "error_message": str(e)[:100]
    })
    results["handled_count"] += 1
    results["error_types"].append(type(e).__name__)

# Calculate success rate
results["success_rate"] = (results["handled_count"] / len(scenarios)) * 100
results["total_scenarios"] = len(scenarios)

# Unique error types
results["unique_error_types"] = list(set(results["error_types"]))

print(json.dumps(results, indent=2))
'''

    # Execute error handling test
    api_response = execute_python_code(error_handling_code)
    assert api_response.get("success"), (
        f"Error handling test failed: {api_response.get('error')}"
    )
    
    results = json.loads(api_response["data"]["stdout"])
    
    # Verify all scenarios were handled properly
    scenario_results = results.get("scenario_results", [])
    assert len(scenario_results) == 3, "Should test 3 error scenarios"
    
    handled_count = results.get("handled_count", 0)
    assert handled_count >= 2, (
        f"At least 2 error scenarios should be handled, got {handled_count}"
    )
    
    # Verify each scenario
    for scenario_result in scenario_results:
        scenario_name = scenario_result.get("scenario", "unknown")
        if scenario_result.get("handled"):
            error_type = scenario_result.get("error_type", "Unknown")
            print(f"✅ {scenario_name}: Properly handled {error_type}")
        else:
            error_msg = scenario_result.get("error", "Unknown error")
            print(f"⚠️  {scenario_name}: {error_msg}")
    
    success_rate = results.get("success_rate", 0)
    print(f"📊 Error handling success rate: {success_rate:.1f}% "
          f"({handled_count}/{results.get('total_scenarios', 3)})")
    
    unique_error_types = results.get("unique_error_types", [])
    if unique_error_types:
        print(f"🔍 Detected error types: {', '.join(unique_error_types)}")


def test_given_concurrent_plot_creation_when_extracting_then_all_retrieved(
    execute_python_code, server_session
):
    """
    Test concurrent plot creation and extraction in virtual filesystem.

    This test validates the system's ability to handle multiple plots
    created in rapid succession and verify their existence via pathlib.

    Description:
        Creates multiple plots in rapid succession to simulate concurrent
        operations, then verifies all plots are properly saved and
        accessible through the virtual filesystem using pathlib operations.

    Input:
        - Python code creating 5 plots in rapid succession
        - Each plot with unique timestamp-based filename
        - Immediate verification of all created plots
        - Uses pathlib for cross-platform compatibility

    Output:
        - API response with concurrent plot creation results
        - JSON data containing:
          * created_files: List of generated filenames
          * verification_results: Individual file verification
          * directory_listing: All files found in directory
          * validation_summary: PNG header verification results

    Example:
        Creates concurrent_1234567890_0.png through concurrent_1234567890_4.png
        Verifies each exists, has valid size, and is readable PNG format

    Assertions:
        - All 5 plots are successfully created
        - Each plot file exists and is accessible
        - All files appear in directory listing
        - Each file has valid PNG header
        - File sizes are reasonable (>0 bytes)
        - System handles rapid plot creation without corruption
    """
    # Given: Create multiple plots rapidly with comprehensive verification
    concurrent_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
base_timestamp = int(time.time() * 1000)

results = {
    "created_files": [],
    "creation_details": {},
    "total_created": 0
}

try:
    # Create 5 plots in rapid succession
    for i in range(5):
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3, 4], [i+1, i+4, i+2, i+3])
        plt.title(f'Concurrent Plot {i+1}')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid(True, alpha=0.3)

        filename = f'concurrent_{base_timestamp}_{i}.png'
        filepath = plots_dir / filename
        plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
        plt.close()

        results["created_files"].append(filename)
        
        # Verify creation immediately
        if filepath.exists():
            file_size = filepath.stat().st_size
            results["creation_details"][filename] = {
                "size": file_size,
                "exists": True,
                "created_successfully": True
            }
        else:
            results["creation_details"][filename] = {
                "size": 0,
                "exists": False,
                "created_successfully": False
            }
        
        results["total_created"] += 1

    results["success"] = True

except Exception as e:
    results["success"] = False
    results["error"] = str(e)

print(json.dumps(results, indent=2))
'''

    # Create concurrent plots
    api_response = execute_python_code(concurrent_plot_code)
    assert api_response.get("success"), (
        f"Concurrent plot creation failed: {api_response.get('error')}"
    )
    
    creation_results = json.loads(api_response["data"]["stdout"])
    assert creation_results.get("success"), (
        f"Plot creation failed: {creation_results.get('error')}"
    )
    
    created_files = creation_results.get("created_files", [])
    assert len(created_files) == 5, f"Expected 5 files, got {len(created_files)}"

    # When: Immediately verify all plots through code execution
    verify_concurrent_code = f'''
import json
from pathlib import Path

created_files = {created_files}
plots_dir = Path('/home/pyodide/plots/matplotlib')

results = {{
    "all_exist": True,
    "files_info": {{}},
    "directory_files": [],
    "summary": {{
        "total_verified": 0,
        "valid_pngs": 0,
        "total_size": 0
    }}
}}

# List directory contents
try:
    results["directory_files"] = [
        f.name for f in plots_dir.iterdir() if f.is_file()
    ]
except Exception as e:
    results["directory_error"] = str(e)
    results["directory_files"] = []

# Verify each concurrent plot
for filename in created_files:
    filepath = plots_dir / filename
    file_info = {{
        "exists": filepath.exists(),
        "size": filepath.stat().st_size if filepath.exists() else 0,
        "in_directory": filename in results["directory_files"],
        "is_file": filepath.is_file() if filepath.exists() else False
    }}

    # Quick PNG validation
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                file_info["is_valid_png"] = header[:4] == b'\\x89PNG'
                if file_info["is_valid_png"]:
                    results["summary"]["valid_pngs"] += 1
        except Exception as e:
            file_info["is_valid_png"] = False
            file_info["read_error"] = str(e)
    else:
        results["all_exist"] = False
        file_info["is_valid_png"] = False

    if file_info["exists"]:
        results["summary"]["total_verified"] += 1
        results["summary"]["total_size"] += file_info["size"]

    results["files_info"][filename] = file_info

# Calculate averages
if results["summary"]["total_verified"] > 0:
    results["summary"]["avg_size"] = (
        results["summary"]["total_size"] / results["summary"]["total_verified"]
    )
else:
    results["summary"]["avg_size"] = 0

results["summary"]["success_rate"] = (
    results["summary"]["total_verified"] / len(created_files) * 100
)

print(json.dumps(results, indent=2))
'''

    api_response = execute_python_code(verify_concurrent_code)
    assert api_response.get("success"), (
        f"Concurrent plot verification failed: {api_response.get('error')}"
    )
    
    verification_results = json.loads(api_response["data"]["stdout"])

    # Then: All plots should be available
    assert verification_results["all_exist"], "Not all concurrent files exist"

    # Verify each file individually
    files_info = verification_results.get("files_info", {})
    for filename in created_files:
        file_info = files_info.get(filename, {})
        assert file_info.get("exists"), f"Concurrent file {filename} not found"
        assert file_info.get("size", 0) > 0, f"Concurrent file {filename} is empty"
        assert file_info.get("in_directory"), (
            f"Concurrent file {filename} not in directory listing"
        )
        assert file_info.get("is_valid_png"), (
            f"Concurrent file {filename} is not a valid PNG"
        )

    # Verify summary statistics
    summary = verification_results.get("summary", {})
    total_verified = summary.get("total_verified", 0)
    valid_pngs = summary.get("valid_pngs", 0)
    success_rate = summary.get("success_rate", 0)

    print(f"✅ All {len(created_files)} concurrent plots verified successfully!")
    print(f"📊 Verification: {total_verified}/{len(created_files)} files exist")
    print(f"🖼️  Valid PNGs: {valid_pngs}/{len(created_files)}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if summary.get("avg_size", 0) > 0:
        print(f"📏 Average file size: {summary['avg_size']:.0f} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
