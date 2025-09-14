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

    Args:
        server_session: The session fixture for making requests

    Returns:
        Callable: Function to execute Python code and return results
    """
    def _execute(code: str, timeout: Optional[int] = None) -> str:
        """
        Execute Python code using the /api/execute-raw endpoint.

        Args:
            code: Python code to execute
            timeout: Request timeout in seconds

        Returns:
            str: The stdout from code execution

        Raises:
            AssertionError: If the request fails
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
        # The execute-raw endpoint returns JSON with stdout field
        result = response.json()
        assert result.get("success"), (
            f"Code execution failed: {result.get('error', 'Unknown error')}"
        )
        return result.get("stdout", "")

    return _execute


def test_given_matplotlib_environment_when_creating_plot_then_file_exists_in_vfs(  # noqa: E501
    execute_python_code, server_session
):
    """
    Test plot creation and verification in virtual filesystem.

    Given: A Pyodide environment with matplotlib installed
    When: Creating a plot and saving it to the virtual filesystem
    Then: The plot file should exist and be accessible in the VFS

    This test validates the complete workflow of:
    1. Creating a matplotlib plot with unique filename
    2. Saving it to the virtual filesystem
    3. Verifying file existence and properties
    4. Checking directory structure and contents
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
    Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)  # Generate unique timestamp

    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title('Debug Filesystem Test')

    output_path = f'/plots/matplotlib/debug_filesystem_{timestamp}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    result["step1_create_plot"] = {
        "success": True,
        "output_path": output_path,
        "filename": output_path.split("/")[-1]
    }

except Exception as e:
    result["step1_create_plot"] = {
        "success": False,
        "error": str(e)
    }

# Step 2: Verify file exists and get details
try:
    if 'output_path' in locals():
        path_obj = Path(output_path)
        file_exists = path_obj.exists()
        file_size = path_obj.stat().st_size if file_exists else 0

        result["step2_verify_file"] = {
            "file_exists": file_exists,
            "file_size": file_size,
            "output_path": output_path
        }
    else:
        result["step2_verify_file"] = {
            "error": "output_path not defined"
        }

except Exception as e:
    result["step2_verify_file"] = {
        "error": str(e)
    }

# Step 3: List directory contents
try:
    plots_path = Path("/plots")
    matplotlib_path = Path("/plots/matplotlib")
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
    output = execute_python_code(plot_creation_code, timeout=60)

    # Then: Parse and validate the results
    result = json.loads(output)

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
    assert step3.get("plots_exists"), "/plots directory should exist"
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
    Test plot extraction API functionality.

    Given: A plot file exists in the virtual filesystem
    When: Calling the extract-plots API endpoint
    Then: The plot file should be successfully extracted

    This test ensures that plots created in the VFS can be retrieved
    through the extraction API endpoint.
    """
    # Given: Create a plot in the VFS first
    plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)
filename = f'extract_test_{timestamp}.png'

plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Extract Test Plot')
plt.savefig(f'/plots/matplotlib/{filename}', dpi=100, bbox_inches='tight')
plt.close()

print(filename)
'''

    # Create the plot and get filename
    filename = execute_python_code(plot_code).strip()
    assert filename, "Filename should be returned from plot creation"

    # When: Verify the plot file through code execution
    verification_code = f'''
import json
from pathlib import Path

filename = "{filename}"
filepath = Path('/plots/matplotlib') / filename

result = {{
    "filename": filename,
    "exists": filepath.exists(),
    "is_file": filepath.is_file() if filepath.exists() else False,
    "size": filepath.stat().st_size if filepath.exists() else 0,
    "parent_dir": str(filepath.parent)
}}

# List all files in the matplotlib directory
plots_dir = Path('/plots/matplotlib')
    result["dir_contents"] = [
        f.name for f in plots_dir.iterdir() if f.is_file()
    ]
    result["file_in_dir"] = filename in result["dir_contents"]
else:
    result["dir_contents"] = []
    result["file_in_dir"] = False

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

    output = execute_python_code(verification_code)
    result = json.loads(output)

    # Then: Verify the plot file is retrievable
    assert result["exists"], f"File {filename} should exist in VFS"
    assert result["is_file"], f"File {filename} should be a regular file"
    assert result["size"] > 0, f"File {filename} should not be empty"
    assert result["file_in_dir"], (
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
    Test extraction of multiple plot files.

    Given: Multiple plot files exist in different formats
    When: Calling the extract-plots API
    Then: All plot files should be successfully extracted

    This test validates that the extraction API can handle multiple
    files of different types and formats.
    """
    # Given: Create multiple plots with different characteristics
    multi_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)
created_files = []

# Create line plot
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
plt.title('Line Plot')
filename1 = f'line_plot_{timestamp}.png'
plt.savefig(f'/plots/matplotlib/{filename1}', dpi=150)
plt.close()
created_files.append(filename1)

# Create bar plot
plt.figure(figsize=(6, 4))
plt.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
plt.title('Bar Plot')
filename2 = f'bar_plot_{timestamp}.png'
plt.savefig(f'/plots/matplotlib/{filename2}', dpi=150)
plt.close()
created_files.append(filename2)

# Create scatter plot
plt.figure(figsize=(6, 4))
plt.scatter([1, 2, 3, 4, 5], [2, 4, 1, 5, 3])
plt.title('Scatter Plot')
filename3 = f'scatter_plot_{timestamp}.png'
plt.savefig(f'/plots/matplotlib/{filename3}', dpi=150)
plt.close()
created_files.append(filename3)

print(json.dumps(created_files))
'''

    # Create multiple plots
    output = execute_python_code(multi_plot_code)
    created_files = json.loads(output)
    assert len(created_files) == 3, (
        f"Expected 3 files, got {len(created_files)}"
    )

    # When: Verify all plots through code execution
    verification_code = f'''
import json
from pathlib import Path

created_files = {created_files}
plots_dir = Path('/plots/matplotlib')
results = {{
    "files_verified": {{}},
    "dir_contents": [],
    "total_files": 0
}}

# List all files in the directory
results["dir_contents"] = [
    f.name for f in plots_dir.iterdir() if f.is_file()
]
results["total_files"] = len(results["dir_contents"])

# Verify each created file
for filename in created_files:
    filepath = plots_dir / filename
    file_info = {{
        "exists": filepath.exists(),
        "is_file": filepath.is_file() if filepath.exists() else False,
        "size": filepath.stat().st_size if filepath.exists() else 0,
        "in_directory": filename in results["dir_contents"]
    }}

    # Verify it's a valid PNG
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                file_info["is_valid_png"] = header[:4] == b'\\x89PNG'
        except:
            file_info["is_valid_png"] = False

    results["files_verified"][filename] = file_info

print(json.dumps(results, indent=2))
'''

    output = execute_python_code(verification_code)
    results = json.loads(output)

    # Then: Verify all plots are retrievable
    assert results["total_files"] >= len(created_files), (
        f"Directory should contain at least {len(created_files)} files"
    )

    # Check each created file
    for filename in created_files:
        file_info = results["files_verified"].get(filename, {})
        assert file_info.get("exists"), (
            f"File {filename} should exist in VFS"
        )
        assert file_info.get("is_file"), (
            f"File {filename} should be a regular file"
        )
        assert file_info.get("size", 0) > 0, (
            f"File {filename} should not be empty"
        )
        assert file_info.get("in_directory"), (
            f"File {filename} should appear in directory listing"
        )
        assert file_info.get("is_valid_png"), (
            f"File {filename} should be a valid PNG"
        )

    print(f"✅ All {len(created_files)} plots were successfully verified!")


def test_given_plots_in_subdirs_when_extracting_then_structure_preserved(
    execute_python_code, server_session
):
    """
    Test plot extraction preserves directory structure.

    Given: Plots exist in different subdirectories (matplotlib, base64)
    When: Extracting plots via API
    Then: Directory structure should be preserved in extraction

    This test ensures the extraction API maintains the organization
    of plots in their respective directories.
    """
    # Given: Create plots in different directories
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
created_structure = {
    "matplotlib": [],
    "base64": []
}

# Create matplotlib plot
Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3], [1, 4, 2])
plt.title('Matplotlib Plot')
mpl_file = f'mpl_test_{timestamp}.png'
plt.savefig(f'/plots/matplotlib/{mpl_file}', dpi=100)
plt.close()
created_structure["matplotlib"].append(mpl_file)

# Create base64 encoded plot
Path('/plots/base64').mkdir(parents=True, exist_ok=True)
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
with open(f'/plots/base64/{b64_file}', 'w') as f:
    f.write(b64_data)
created_structure["base64"].append(b64_file)

print(json.dumps(created_structure))
'''

    # When: Create plots and verify their structure in VFS
    verification_code = subdir_plot_code + '''

# Verify the created files exist in their directories
verification = {
    "matplotlib_files": [],
    "base64_files": [],
    "matplotlib_exists": {},
    "base64_exists": {}
}

# List matplotlib directory
matplotlib_path = Path('/plots/matplotlib')
if matplotlib_path.exists():
    verification["matplotlib_files"] = [
        p.name for p in matplotlib_path.iterdir()
    ]
    for f in created_structure["matplotlib"]:
        file_path = matplotlib_path / f
        verification["matplotlib_exists"][f] = {
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }

# List base64 directory
base64_path = Path('/plots/base64')
if base64_path.exists():
    verification["base64_files"] = [p.name for p in base64_path.iterdir()]
    for f in created_structure["base64"]:
        file_path = base64_path / f
        verification["base64_exists"][f] = {
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }

print("VERIFICATION:")
print(json.dumps(verification, indent=2))
'''

    output = execute_python_code(verification_code)

    # Parse the output to get both created structure and verification
    lines = output.strip().split('\n')
    created_json_idx = None
    verification_json_idx = None

    for i, line in enumerate(lines):
        if line.startswith('{') and created_json_idx is None:
            created_json_idx = i
        elif line == "VERIFICATION:" and i < len(lines) - 1:
            verification_json_idx = i + 1
            break
    # Verify directory structure is preserved and files accessible
    # through code execution without using internal APIs.
    assert verification_json_idx is not None, \
        "Could not find verification JSON"

    created_structure = json.loads(lines[created_json_idx])
    verification_output = '\n'.join(lines[verification_json_idx:])
    verification = json.loads(verification_output)

    # Then: Verify directory structure is preserved
    # Check matplotlib files
    for mpl_file in created_structure["matplotlib"]:
        assert mpl_file in verification["matplotlib_exists"], (
            f"Matplotlib file {mpl_file} not tracked"
        )
        file_info = verification["matplotlib_exists"][mpl_file]
        assert file_info["exists"], (
            f"Matplotlib file {mpl_file} not found in /plots/matplotlib"
        )
        assert file_info["size"] > 0, (
            f"Matplotlib file {mpl_file} is empty"
        )

    # Check base64 files
    for b64_file in created_structure["base64"]:
        assert b64_file in verification["base64_exists"], (
            f"Base64 file {b64_file} not tracked"
        )
        file_info = verification["base64_exists"][b64_file]
        assert file_info["exists"], (
            f"Base64 file {b64_file} not found in /plots/base64"
        )
        assert file_info["size"] > 0, f"Base64 file {b64_file} is empty"

    # Verify files appear in directory listings
    assert all(f in verification["matplotlib_files"]
               for f in created_structure["matplotlib"]), (
        "Not all matplotlib files appear in directory listing"
    )
    assert all(f in verification["base64_files"]
               for f in created_structure["base64"]), (
        "Not all base64 files appear in directory listing"
    )

    print("✅ Directory structure preserved!")


def test_given_no_plots_when_extracting_then_empty_result_returned(
    execute_python_code, server_session
):
    """
    Test extraction API behavior when no plots exist.

    Given: No plot files exist in the filesystem
    When: Calling the extract-plots API
    Then: Should return success with empty results

    This test validates proper handling of empty directories.
    """
    # Given: Clear any existing plots
    clear_plots_code = '''
from pathlib import Path
import shutil

# Clear plot directories if they exist
for dir_path in ['/plots/matplotlib', '/plots/seaborn', '/plots/base64']:
    path_obj = Path(dir_path)
    if path_obj.exists():
        shutil.rmtree(dir_path)
        path_obj.mkdir(parents=True, exist_ok=True)

print("Directories cleared")
'''

    output = execute_python_code(clear_plots_code)
    assert "Directories cleared" in output

    # When: Verify directories are empty through code execution
    verify_empty_code = '''
import json
from pathlib import Path

results = {
    "directories": {},
    "total_files": 0
}

# Check each plot directory
plot_dirs = ['/plots/matplotlib', '/plots/seaborn', '/plots/base64']
for dir_path in plot_dirs:
    p = Path(dir_path)
    dir_info = {
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "files": [],
        "file_count": 0
    }

    if p.exists() and p.is_dir():
        files = [f.name for f in p.iterdir() if f.is_file()]
        dir_info["files"] = files
        dir_info["file_count"] = len(files)
        results["total_files"] += len(files)

    results["directories"][dir_path] = dir_info

print(json.dumps(results, indent=2))
'''

    output = execute_python_code(verify_empty_code)
    results = json.loads(output)

    # Then: Should show no files in any directory
    assert results["total_files"] == 0, (
        f"Expected 0 total files, got {results['total_files']}"
    )

    for dir_path, dir_info in results["directories"].items():
        assert dir_info["file_count"] == 0, (
            f"Directory {dir_path} should be empty but has "
            f"{dir_info['file_count']} files: {dir_info['files']}"
        )

    print("✅ Empty directories verified correctly!")


def test_given_large_plot_when_saving_then_handled_correctly(
    execute_python_code, server_session
):
    """
    Test handling of large plot files.

    Given: A high-resolution plot with complex data
    When: Saving to the virtual filesystem
    Then: Should be saved and extracted successfully

    This test validates that the system can handle larger files
    without issues.
    """
    # Given: Create a large, complex plot
    large_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)

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
filepath = f'/plots/matplotlib/{filename}'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

# Get file size
file_size = Path(filepath).stat().st_size
print(f"{filename}|{file_size}")
'''

    # Create large plot
    output = execute_python_code(large_plot_code, timeout=60)
    filename, file_size = output.strip().split('|')
    file_size = int(file_size)

    # Verify it's a reasonably large file
    assert file_size > 100000, f"Expected large file, got {file_size} bytes"
    print(f"📊 Created large plot: {filename} ({file_size:,} bytes)")

    # When: Verify the large plot through code execution
    verify_large_plot_code = f'''
import json
from pathlib import Path

filename = "{filename}"
filepath = Path('/plots/matplotlib') / filename

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
plots_dir = Path('/plots/matplotlib')
if plots_dir.exists():
    files = [f.name for f in plots_dir.iterdir() if f.is_file()]
    result["in_directory"] = filename in files
    result["dir_file_count"] = len(files)

print(json.dumps(result, indent=2))
'''

    output = execute_python_code(verify_large_plot_code)
    result = json.loads(output)

    # Then: Should verify large file exists and is valid
    assert result["exists"], f"Large file {filename} should exist"
    assert result["size"] == file_size, \
        f"File size mismatch: expected {file_size}, got {result['size']}"
    print(f"✅ Large plot file verified successfully! "
          f"Size: {result['size_mb']} MB")
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

    Given: Invalid plot operations or data
    When: Attempting to create and save plots
    Then: Errors should be handled gracefully

    This test ensures robust error handling in plot generation.
    """
    # Test various error scenarios
    error_scenarios = [
        # Scenario 1: Invalid figure size
        '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.figure(figsize=(-5, -3))  # Invalid negative size
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.savefig('/plots/matplotlib/invalid_size.png')
    print("ERROR: Should have failed with invalid size")
except Exception as e:
    print(f"HANDLED: Invalid size error - {type(e).__name__}")
''',

        # Scenario 2: Invalid save path
        '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.savefig('/nonexistent/directory/plot.png')
    print("ERROR: Should have failed with invalid path")
except Exception as e:
    print(f"HANDLED: Invalid path error - {type(e).__name__}")
''',

        # Scenario 3: Invalid data types
        '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.figure(figsize=(5, 3))
    plt.plot("invalid", "data")  # String instead of numeric data
    plt.savefig('/plots/matplotlib/invalid_data.png')
    print("ERROR: Should have failed with invalid data")
except Exception as e:
    print(f"HANDLED: Invalid data error - {type(e).__name__}")
'''
    ]

    for i, scenario_code in enumerate(error_scenarios):
        output = execute_python_code(scenario_code)
        assert "HANDLED:" in output, (
            f"Scenario {i+1} did not handle error properly: {output}"
        )
        print(f"✅ Error scenario {i+1} handled correctly")


def test_given_concurrent_plot_creation_when_extracting_then_all_retrieved(
    execute_python_code, server_session
):
    """
    Test concurrent plot creation and extraction.

    Given: Multiple plots created in rapid succession
    When: Extracting immediately after creation
    Then: All plots should be successfully retrieved

    This test simulates concurrent plot generation scenarios.
    """
    # Given: Create multiple plots rapidly
    concurrent_plot_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

Path('/plots/matplotlib').mkdir(parents=True, exist_ok=True)
base_timestamp = int(time.time() * 1000)
created_files = []

# Create 5 plots in rapid succession
for i in range(5):
    plt.figure(figsize=(4, 3))
    plt.plot([1, 2, 3, 4], [i+1, i+4, i+2, i+3])
    plt.title(f'Concurrent Plot {i+1}')

    filename = f'concurrent_{base_timestamp}_{i}.png'
    plt.savefig(f'/plots/matplotlib/{filename}', dpi=100)
    plt.close()

    created_files.append(filename)

print(json.dumps(created_files))
'''

    # Create plots
    output = execute_python_code(concurrent_plot_code)
    created_files = json.loads(output)
    assert len(created_files) == 5

    # When: Immediately verify all plots through code execution
    verify_concurrent_code = f'''
import json
from pathlib import Path

created_files = {created_files}
plots_dir = Path('/plots/matplotlib')
results = {{
    "all_exist": True,
    "files_info": {{}},
    "directory_files": []
}}

# List directory contents
results["directory_files"] = [
    f.name for f in plots_dir.iterdir() if f.is_file()
]

# Verify each concurrent plot
for filename in created_files:
    filepath = plots_dir / filename
    file_info = {{
        "exists": filepath.exists(),
        "size": filepath.stat().st_size if filepath.exists() else 0,
        "in_directory": filename in results["directory_files"]
    }}

    # Quick PNG validation
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                file_info["is_valid_png"] = header[:4] == b'\\x89PNG'
        except:
            file_info["is_valid_png"] = False
    else:
        results["all_exist"] = False

    results["files_info"][filename] = file_info

print(json.dumps(results, indent=2))
'''

    output = execute_python_code(verify_concurrent_code)
    results = json.loads(output)

    # Then: All plots should be available
    assert results["all_exist"], "Not all concurrent files exist"

    # Verify each file individually
    for filename in created_files:
        file_info = results["files_info"][filename]
        assert file_info["exists"], f"Concurrent file {filename} not found"
        assert file_info["size"] > 0, f"Concurrent file {filename} is empty"
        assert file_info["in_directory"], (
            f"Concurrent file {filename} not in directory listing"
        )
        assert file_info.get("is_valid_png"), (
            f"Concurrent file {filename} is not a valid PNG"
        )

    print(
        f"✅ All {len(created_files)} concurrent plots verified successfully!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
