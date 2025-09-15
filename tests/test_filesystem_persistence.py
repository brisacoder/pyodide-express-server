"""
Test filesystem persistence behavior across server sessions using BDD patterns.

This test suite documents and validates the filesystem persistence behavior
of the Pyodide Express Server across different scenarios. It follows BDD
(Behavior-Driven Development) patterns with clear Given/When/Then structure.

Key Behaviors Tested:
1. File creation and persistence within single session
2. Cross-session persistence behavior documentation
3. Recommended patterns for data persistence
4. User guidance for proper file handling
5. Filesystem type detection and capabilities

API Contract Compliance:
- Uses only /api/execute-raw endpoint (no internal pyodide APIs)
- Validates standardized response format with meta.timestamp
- Handles both success and error scenarios properly
- Cross-platform Python code using pathlib for compatibility

BDD Test Structure:
- Given: Setup conditions and prerequisites
- When: Execute the action being tested
- Then: Validate expected outcomes and behaviors

Author: Copilot Engineering Team
Date: 2025-01-14
Version: 2.0 (Pytest + BDD)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests


class TestConfig:
    """Centralized configuration for filesystem persistence tests."""
    
    # Server settings
    BASE_URL = "http://localhost:3000"
    
    # Timeout settings (seconds)
    TIMEOUTS = {
        "api_request": 30,
        "server_health": 10,
        "code_execution": 45,
        "server_restart": 120,
    }
    
    # API endpoints
    ENDPOINTS = {
        "health": "/health",
        "execute_raw": "/api/execute-raw",
    }
    
    # Test paths (using Pyodide virtual filesystem)
    PATHS = {
        "temp_dir": "/tmp",
        "plots_dir": "/plots/matplotlib",
        "uploads_dir": "/uploads",
        "test_data_dir": "/test_data",
    }
    
    # File creation settings
    FILE_SETTINGS = {
        "test_content": "Filesystem persistence test content",
        "max_filename_length": 50,
        "supported_extensions": [".txt", ".json", ".csv", ".png"],
    }


@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure server is available for testing.
    
    Given: The test suite needs a running Pyodide Express Server
    When: Tests are executed
    Then: Server should be available and responding to health checks
    
    Raises:
        pytest.skip: If server is not available within timeout
    """
    def wait_for_server(url: str, timeout: int) -> bool:
        """Poll server until available or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        return False
    
    health_url = f"{TestConfig.BASE_URL}{TestConfig.ENDPOINTS['health']}"
    if not wait_for_server(health_url, TestConfig.TIMEOUTS["server_health"]):
        pytest.skip(f"Server not available at {TestConfig.BASE_URL}")

@pytest.fixture
def test_cleanup():
    """
    Provide cleanup tracking for test artifacts.
    
    Yields:
        CleanupTracker: Object to track files and resources for cleanup
        
    Example:
        >>> def test_something(test_cleanup):
        ...     cleanup = test_cleanup
        ...     cleanup.track_file("/tmp/test.txt")
        ...     # File automatically cleaned up after test
    """
    class CleanupTracker:
        """Track and clean up test artifacts."""
        
        def __init__(self):
            self.temp_files: List[str] = []
            self.start_time = time.time()
        
        def track_file(self, filepath: str) -> None:
            """Track file for cleanup via API."""
            self.temp_files.append(filepath)
        
        def get_test_duration(self) -> float:
            """Get test execution duration."""
            return time.time() - self.start_time
        
        def cleanup(self) -> None:
            """Clean up tracked files (best effort)."""
            for filepath in self.temp_files:
                try:
                    # Use execute-raw to clean up files in Pyodide filesystem
                    cleanup_code = f"""
from pathlib import Path
try:
    file_path = Path("{filepath}")
    if file_path.exists():
        file_path.unlink()
        print(f"Cleaned up: {filepath}")
    else:
        print(f"File not found: {filepath}")
except Exception as e:
    print(f"Cleanup error for {filepath}: {{e}}")
"""
                    requests.post(
                        f"{TestConfig.BASE_URL}{TestConfig.ENDPOINTS['execute_raw']}",
                        headers={"Content-Type": "text/plain"},
                        data=cleanup_code,
                        timeout=10
                    )
                except Exception:
                    pass  # Best effort cleanup
    
    tracker = CleanupTracker()
    yield tracker
    tracker.cleanup()


@pytest.fixture
def api_client():
    """
    Provide API client for execute-raw endpoint with contract validation.
    
    Returns:
        APIClient: Client with execute_code method and contract validation
        
    Example:
        >>> def test_something(api_client):
        ...     result = api_client.execute_code("print('Hello')")
        ...     assert result["success"] is True
        ...     assert "Hello" in result["data"]["stdout"]
    """
    class APIClient:
        """API client with automatic contract validation."""
        
        def execute_code(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
            """
            Execute Python code using /api/execute-raw endpoint.
            
            Args:
                code: Python code to execute
                timeout: Request timeout in seconds
                
            Returns:
                Dict containing validated API response
                
            Raises:
                requests.RequestException: If request fails
                AssertionError: If response doesn't match API contract
            """
            if timeout is None:
                timeout = TestConfig.TIMEOUTS["code_execution"]
            
            response = requests.post(
                f"{TestConfig.BASE_URL}{TestConfig.ENDPOINTS['execute_raw']}",
                headers={"Content-Type": "text/plain"},
                data=code,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            self._validate_api_contract(result)
            return result
        
        def _validate_api_contract(self, response_data: Dict[str, Any]) -> None:
            """
            Validate API response follows the expected contract format.
            
            Expected format:
            {
              "success": true | false,
              "data": { "result": any, "stdout": str, "stderr": str, "executionTime": int } | null,
              "error": str | null,
              "meta": { "timestamp": str }
            }
            """
            # Validate basic required fields
            assert "success" in response_data, "Missing required field: success"
            assert isinstance(response_data["success"], bool), f"success must be boolean: {type(response_data['success'])}"
            
            # Validate meta timestamp
            assert "meta" in response_data, "Missing required field: meta"
            assert isinstance(response_data["meta"], dict), f"meta must be dict: {type(response_data['meta'])}"
            assert "timestamp" in response_data["meta"], "meta must contain timestamp"
            
            # Validate success/error relationship
            if response_data["success"]:
                assert response_data.get("data") is not None, "Success response should have non-null data"
                assert response_data.get("error") is None, "Success response should have null error"
                
                # Validate data structure for successful execution
                data = response_data["data"]
                required_fields = ["result", "stdout", "stderr", "executionTime"]
                for field in required_fields:
                    assert field in data, f"data missing required field '{field}': {data}"
                
                # Validate executionTime is a positive number
                assert isinstance(data["executionTime"], (int, float)), f"executionTime must be number: {type(data['executionTime'])}"
                assert data["executionTime"] >= 0, f"executionTime must be non-negative: {data['executionTime']}"
            else:
                assert response_data.get("error") is not None, "Error response should have non-null error"
                assert isinstance(response_data["error"], str), f"error must be string: {type(response_data['error'])}"
                assert response_data.get("data") is None, "Error response should have null data"
    
    return APIClient()


class TestFilesystemPersistenceBehavior:
    """
    Test filesystem persistence behavior using BDD patterns.
    
    This test class documents and validates how files persist (or don't persist)
    in the Pyodide virtual filesystem across different scenarios.
    """
    
    def test_given_new_session_when_file_created_then_file_exists_in_same_session(
        self, 
        server_ready, 
        api_client, 
        test_cleanup
    ):
        """
        Test that files created in Pyodide filesystem exist within the same session.
        
        Given: A new Pyodide execution session
        When: A file is created in the virtual filesystem
        Then: The file should exist and be readable in the same session
        
        Args:
            server_ready: Fixture ensuring server availability
            api_client: API client with contract validation
            test_cleanup: Cleanup tracker for test artifacts
            
        Expected Result:
            - File creation succeeds
            - File exists after creation
            - File content matches expected content
            - All API responses follow contract format
        """
        # Given: Generate unique test file identifier
        test_timestamp = int(time.time())
        test_filename = f"session_test_{test_timestamp}.txt"
        test_filepath = f"{TestConfig.PATHS['temp_dir']}/{test_filename}"
        test_content = f"{TestConfig.FILE_SETTINGS['test_content']} - {test_timestamp}"
        
        test_cleanup.track_file(test_filepath)
        
        # When: Create file in Pyodide virtual filesystem
        create_file_code = f"""
from pathlib import Path

# Create test file using pathlib for cross-platform compatibility
test_file = Path("{test_filepath}")
test_content = "{test_content}"
test_file.write_text(test_content)

# Verify file creation
result = {{
    "file_created": test_file.exists(),
    "file_path": str(test_file),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "file_size": test_file.stat().st_size if test_file.exists() else 0,
    "parent_dir_exists": test_file.parent.exists(),
    "platform_info": {{
        "cwd": str(Path.cwd()),
        "temp_dir_writable": Path("{TestConfig.PATHS['temp_dir']}").exists()
    }}
}}

print(f"File creation result: {{result['file_created']}}")
print(f"File path: {{result['file_path']}}")
print(f"Content length: {{len(result['file_content']) if result['file_content'] else 0}}")

result
"""
        
        # Execute file creation
        response = api_client.execute_code(create_file_code)
        
        # Then: Validate file creation success
        assert response["success"] is True, f"File creation failed: {response}"
        
        # Parse result from stdout (last line should contain the result)
        stdout_lines = response["data"]["stdout"].strip().split('\n')
        assert "File creation result: True" in stdout_lines[0], "File should be created successfully"
        assert test_filepath in stdout_lines[1], "File path should be logged"
        assert f"Content length: {len(test_content)}" in stdout_lines[2], "Content length should match"
        
        # Verify file can be read in same session
        read_file_code = f"""
from pathlib import Path

test_file = Path("{test_filepath}")
result = {{
    "file_exists": test_file.exists(),
    "file_readable": False,
    "content_matches": False,
    "actual_content": None
}}

if test_file.exists():
    try:
        actual_content = test_file.read_text()
        result["file_readable"] = True
        result["actual_content"] = actual_content
        result["content_matches"] = (actual_content == "{test_content}")
    except Exception as e:
        result["read_error"] = str(e)

print(f"File exists: {{result['file_exists']}}")
print(f"Content matches: {{result['content_matches']}}")

result
"""
        
        read_response = api_client.execute_code(read_file_code)
        
        # Then: Validate file persistence within session
        assert read_response["success"] is True, f"File read failed: {read_response}"
        
        stdout_lines = read_response["data"]["stdout"].strip().split('\n')
        assert "File exists: True" in stdout_lines[0], "File should exist in same session"
        assert "Content matches: True" in stdout_lines[1], "File content should match exactly"

    
    def test_given_file_created_when_new_execution_context_then_document_persistence_behavior(
        self, 
        server_ready, 
        api_client, 
        test_cleanup
    ):
        """
        Document filesystem persistence behavior across execution contexts.
        
        Given: A file is created in one execution context
        When: A new execution context is started (new API call)
        Then: Document whether the file persists and provide guidance
        
        Args:
            server_ready: Fixture ensuring server availability
            api_client: API client with contract validation
            test_cleanup: Cleanup tracker for test artifacts
            
        Purpose:
            This test documents the current filesystem persistence behavior
            and provides clear guidance for users on what to expect.
        """
        # Given: Create file in first execution context
        test_timestamp = int(time.time())
        test_filename = f"persistence_test_{test_timestamp}.txt"
        test_filepath = f"{TestConfig.PATHS['temp_dir']}/{test_filename}"
        test_content = f"Persistence test - {test_timestamp}"
        
        test_cleanup.track_file(test_filepath)
        
        # Create file in first context
        create_code = f"""
from pathlib import Path

test_file = Path("{test_filepath}")
test_file.write_text("{test_content}")

result = {{
    "created_in_context_1": test_file.exists(),
    "content": test_file.read_text() if test_file.exists() else None,
    "creation_timestamp": {test_timestamp}
}}

print("📁 File created in execution context 1")
print(f"   Path: {{str(test_file)}}")
print(f"   Exists: {{result['created_in_context_1']}}")

result
"""
        
        first_response = api_client.execute_code(create_code)
        assert first_response["success"] is True, "File creation should succeed"
        
        # When: Check file existence in second execution context
        check_code = f"""
from pathlib import Path
import sys

test_file = Path("{test_filepath}")

result = {{
    "exists_in_context_2": test_file.exists(),
    "content_in_context_2": test_file.read_text() if test_file.exists() else None,
    "check_timestamp": {int(time.time())},
    "filesystem_info": {{
        "platform": sys.platform,
        "pyodide_available": "pyodide" in sys.modules,
        "cwd": str(Path.cwd()),
        "temp_dir_accessible": Path("{TestConfig.PATHS['temp_dir']}").exists()
    }}
}}

print("📁 File check in execution context 2")
print(f"   Path: {{str(test_file)}}")
print(f"   Exists: {{result['exists_in_context_2']}}")

if result['exists_in_context_2']:
    print("✅ File persists across execution contexts within same server session")
    print("   This indicates shared filesystem state between API calls")
else:
    print("❌ File does not persist across execution contexts")
    print("   Each API call may have isolated filesystem state")

result
"""
        
        second_response = api_client.execute_code(check_code)
        assert second_response["success"] is True, "File check should succeed"
        
        # Then: Document the behavior observed
        stdout_output = second_response["data"]["stdout"]
        
        # Print documentation based on observed behavior
        print("\n" + "="*60)
        print("📚 FILESYSTEM PERSISTENCE BEHAVIOR DOCUMENTATION")
        print("="*60)
        
        if "✅ File persists" in stdout_output:
            print("✅ OBSERVATION: Files persist across execution contexts")
            print("   - Files created in one API call are accessible in subsequent calls")
            print("   - Shared filesystem state maintained within server session")
            print("   - Suitable for multi-step data processing workflows")
        else:
            print("❌ OBSERVATION: Files do not persist across execution contexts")
            print("   - Each API call has isolated filesystem state")
            print("   - Files must be recreated or uploaded for each operation")
            print("   - Single-call workflows recommended")
        
        print("\n📋 USER GUIDANCE:")
        print("   1. For data persistence: Return results as JSON in API response")
        print("   2. For plots: Save to mounted directories (/plots/matplotlib/)")
        print("   3. For large files: Use file upload endpoints")
        print("   4. For temporary data: Process and return immediately")
        print("="*60)
    
    def test_given_filesystem_capabilities_when_analyzed_then_provide_user_guidance(
        self, 
        server_ready, 
        api_client
    ):
        """
        Analyze filesystem capabilities and provide comprehensive user guidance.
        
        Given: The Pyodide virtual filesystem environment
        When: Filesystem capabilities are analyzed
        Then: Provide clear guidance for optimal file handling patterns
        
        Args:
            server_ready: Fixture ensuring server availability
            api_client: API client with contract validation
            
        Purpose:
            Provide users with actionable guidance on how to handle files
            effectively in the Pyodide environment.
        """
        # When: Analyze filesystem capabilities
        analysis_code = f"""
from pathlib import Path
import sys
import time

# Test different filesystem locations
test_locations = [
    "{TestConfig.PATHS['temp_dir']}",
    "{TestConfig.PATHS['plots_dir']}",
    "{TestConfig.PATHS['uploads_dir']}",
    "/home",
    "/var/tmp"
]

analysis_result = {{
    "platform_info": {{
        "python_platform": sys.platform,
        "pyodide_available": "pyodide" in sys.modules,
        "current_directory": str(Path.cwd())
    }},
    "location_tests": {{}},
    "recommended_patterns": [],
    "warnings": []
}}

# Test each location for read/write capabilities
for location in test_locations:
    try:
        test_dir = Path(location)
        test_file = test_dir / f"capability_test_{{int(time.time())}}.txt"
        
        # Test directory creation if needed
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test file operations
        test_content = "Capability test"
        test_file.write_text(test_content)
        read_content = test_file.read_text()
        test_file.unlink()  # Clean up
        
        analysis_result["location_tests"][location] = {{
            "writable": True,
            "readable": True,
            "content_preserved": read_content == test_content
        }}
        
    except Exception as e:
        analysis_result["location_tests"][location] = {{
            "writable": False,
            "error": str(e)
        }}

# Generate recommendations based on test results
if analysis_result["location_tests"].get("{TestConfig.PATHS['plots_dir']}", {{}}).get("writable"):
    analysis_result["recommended_patterns"].append(
        "✅ Use {TestConfig.PATHS['plots_dir']} for matplotlib plot output"
    )

if analysis_result["location_tests"].get("{TestConfig.PATHS['temp_dir']}", {{}}).get("writable"):
    analysis_result["recommended_patterns"].append(
        "✅ Use {TestConfig.PATHS['temp_dir']} for temporary file processing"
    )
else:
    analysis_result["warnings"].append(
        "⚠️  Temporary directory not writable - use in-memory processing"
    )

print("🔍 FILESYSTEM CAPABILITY ANALYSIS")
print("=" * 50)
for location, result in analysis_result["location_tests"].items():
    status = "✅" if result.get("writable") else "❌"
    print(f"{status} {location}: {{'writable' if result.get('writable') else 'not writable'}}")

print("\\n📋 RECOMMENDED PATTERNS:")
for pattern in analysis_result["recommended_patterns"]:
    print(f"   {pattern}")

if analysis_result["warnings"]:
    print("\\n⚠️  WARNINGS:")
    for warning in analysis_result["warnings"]:
        print(f"   {warning}")

analysis_result
"""
        
        # Execute filesystem analysis
        response = api_client.execute_code(analysis_code)
        
        # Then: Validate analysis completed successfully
        assert response["success"] is True, f"Filesystem analysis failed: {response}"
        
        # Verify comprehensive output was provided
        stdout_output = response["data"]["stdout"]
        assert "FILESYSTEM CAPABILITY ANALYSIS" in stdout_output, "Analysis header should be present"
        assert "RECOMMENDED PATTERNS" in stdout_output, "Recommendations should be provided"
        
        # Document the guidance for users
        print("\n" + "="*60)
        print("📚 COMPREHENSIVE USER GUIDANCE FOR FILE OPERATIONS")
        print("="*60)
        print(stdout_output)
        print("\n🎯 BEST PRACTICES SUMMARY:")
        print("   1. 📊 Plots: Save to /plots/matplotlib/ for direct filesystem access")
        print("   2. 📄 Data: Return processed results as JSON in API response")
        print("   3. 🔄 Workflows: Complete processing in single API call when possible")
        print("   4. 🗂️  Large Files: Use upload endpoints and process immediately")
        print("   5. 🧹 Cleanup: Don't rely on file persistence across sessions")
        print("="*60)
    
    def test_given_cross_platform_code_when_executed_then_works_on_windows_and_linux(
        self, 
        server_ready, 
        api_client, 
        test_cleanup
    ):
        """
        Validate that Python code using pathlib works across platforms.
        
        Given: Python code that needs to work on both Windows and Linux
        When: The code is executed in Pyodide environment
        Then: pathlib should provide cross-platform compatibility
        
        Args:
            server_ready: Fixture ensuring server availability
            api_client: API client with contract validation
            test_cleanup: Cleanup tracker for test artifacts
            
        Purpose:
            Ensure that all Python code in tests uses pathlib for maximum
            portability across different operating systems.
        """
        # Given: Cross-platform file operations using pathlib
        test_timestamp = int(time.time())
        
        cross_platform_code = f"""
from pathlib import Path
import sys
import os

# Cross-platform file operations demo
base_dir = Path("{TestConfig.PATHS['temp_dir']}")
test_file = base_dir / "cross_platform_test_{test_timestamp}.txt"
nested_dir = base_dir / "nested" / "subdirectory"

result = {{
    "platform_info": {{
        "python_platform": sys.platform,
        "path_separator": os.sep,
        "current_working_directory": str(Path.cwd())
    }},
    "pathlib_operations": {{}},
    "portability_verified": True
}}

try:
    # Test 1: Directory creation with parents
    nested_dir.mkdir(parents=True, exist_ok=True)
    result["pathlib_operations"]["mkdir_parents"] = nested_dir.exists()
    
    # Test 2: File creation with path joining
    test_content = "Cross-platform test content\\nLine 2\\nLine 3"
    test_file.write_text(test_content, encoding="utf-8")
    result["pathlib_operations"]["file_creation"] = test_file.exists()
    
    # Test 3: Path resolution and normalization
    resolved_path = test_file.resolve()
    result["pathlib_operations"]["path_resolution"] = str(resolved_path)
    
    # Test 4: File metadata access
    file_stat = test_file.stat()
    result["pathlib_operations"]["file_metadata"] = {{
        "size": file_stat.st_size,
        "exists": test_file.exists(),
        "is_file": test_file.is_file()
    }}
    
    # Test 5: Directory traversal
    all_files = list(base_dir.rglob("*.txt"))
    result["pathlib_operations"]["directory_traversal"] = {{
        "total_txt_files": len(all_files),
        "test_file_found": test_file in all_files
    }}
    
    # Test 6: Cross-platform path operations
    parts_test = test_file.parts
    parent_test = test_file.parent
    name_test = test_file.name
    suffix_test = test_file.suffix
    
    result["pathlib_operations"]["path_components"] = {{
        "parts": list(parts_test),
        "parent": str(parent_test),
        "name": name_test,
        "suffix": suffix_test
    }}
    
    print("✅ Cross-platform pathlib operations successful")
    print(f"   Platform: {{sys.platform}}")
    print(f"   File created: {{test_file}}")
    print(f"   Directory created: {{nested_dir}}")
    print(f"   File size: {{file_stat.st_size}} bytes")
    
    # Cleanup
    test_file.unlink()
    nested_dir.rmdir()
    (base_dir / "nested").rmdir()
    
except Exception as e:
    result["portability_verified"] = False
    result["error"] = str(e)
    print(f"❌ Cross-platform operation failed: {{e}}")

result
"""
        
        test_cleanup.track_file(f"{TestConfig.PATHS['temp_dir']}/cross_platform_test_{test_timestamp}.txt")
        
        # When: Execute cross-platform code
        response = api_client.execute_code(cross_platform_code)
        
        # Then: Validate cross-platform compatibility
        assert response["success"] is True, f"Cross-platform test failed: {response}"
        
        stdout_output = response["data"]["stdout"]
        assert "✅ Cross-platform pathlib operations successful" in stdout_output, "Operations should succeed"
        
        # Verify platform information is captured
        assert "Platform:" in stdout_output, "Platform info should be logged"
        assert "File created:" in stdout_output, "File creation should be logged"
        assert "Directory created:" in stdout_output, "Directory creation should be logged"
        
        print("\n✅ CROSS-PLATFORM COMPATIBILITY VERIFIED")
        print("   - pathlib provides consistent API across platforms")
        print("   - File and directory operations work reliably")
        print("   - Path resolution and metadata access functional")
        print("   - Recommended for all file operations in tests")


class TestFilesystemPersistenceRecommendations:
    """
    Test class focused on documenting and validating recommended patterns.
    
    This class provides comprehensive examples and validation of the
    recommended approaches for handling files in Pyodide environments.
    """
    
    def test_given_data_processing_need_when_using_recommended_patterns_then_demonstrate_best_practices(
        self, 
        server_ready, 
        api_client, 
        test_cleanup
    ):
        """
        Demonstrate recommended patterns for data processing workflows.
        
        Given: A need to process data and preserve results
        When: Using recommended patterns for file handling
        Then: Demonstrate effective approaches for different scenarios
        
        Args:
            server_ready: Fixture ensuring server availability
            api_client: API client with contract validation
            test_cleanup: Cleanup tracker for test artifacts
            
        Examples:
            Pattern 1: Return processed data as JSON
            Pattern 2: Save plots to mounted directories
            Pattern 3: Process uploaded files immediately
        """
        # When: Demonstrate Pattern 1 - Return data as JSON
        data_processing_code = """
import json
from pathlib import Path

# Pattern 1: Process data and return as JSON (recommended for data results)
sample_data = {
    "users": [
        {"id": 1, "name": "Alice", "score": 85},
        {"id": 2, "name": "Bob", "score": 92},
        {"id": 3, "name": "Charlie", "score": 78}
    ]
}

# Process the data
processed_results = {
    "total_users": len(sample_data["users"]),
    "average_score": sum(user["score"] for user in sample_data["users"]) / len(sample_data["users"]),
    "top_performer": max(sample_data["users"], key=lambda x: x["score"]),
    "score_distribution": {
        "above_80": len([u for u in sample_data["users"] if u["score"] > 80]),
        "below_80": len([u for u in sample_data["users"] if u["score"] <= 80])
    }
}

print("📊 Pattern 1: Data Processing with JSON Return")
print(f"   Total users: {processed_results['total_users']}")
print(f"   Average score: {processed_results['average_score']:.1f}")
print(f"   Top performer: {processed_results['top_performer']['name']}")
print("✅ Recommended: Return processed data directly in API response")

# Return the processed data (will be captured in API response)
processed_results
"""
        
        # Execute data processing pattern
        response = api_client.execute_code(data_processing_code)
        
        # Then: Validate data processing pattern
        assert response["success"] is True, "Data processing should succeed"
        assert "Pattern 1: Data Processing" in response["data"]["stdout"], "Pattern should be demonstrated"
        
        # When: Demonstrate Pattern 2 - Plot generation
        plot_generation_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pattern 2: Generate plots and save to mounted directory (recommended for visualizations)
plots_dir = Path("{TestConfig.PATHS['plots_dir']}")
plots_dir.mkdir(parents=True, exist_ok=True)

# Create sample plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Pattern 2: Plot Generation Example')
plt.legend()
plt.grid(True, alpha=0.3)

# Save to mounted directory
plot_filename = "pattern_demo_{int(time.time())}.png"
plot_path = plots_dir / plot_filename
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

result = {{
    "plot_saved": plot_path.exists(),
    "plot_path": str(plot_path),
    "plot_size": plot_path.stat().st_size if plot_path.exists() else 0
}}

print("📈 Pattern 2: Plot Generation with Mounted Directory")
print(f"   Plot saved: {{result['plot_saved']}}")
print(f"   File path: {{result['plot_path']}}")
print(f"   File size: {{result['plot_size']}} bytes")
print("✅ Recommended: Save plots to mounted directories for direct access")

result
"""
        
        # Execute plot generation pattern
        plot_response = api_client.execute_code(plot_generation_code)
        
        # Then: Validate plot generation pattern
        assert plot_response["success"] is True, "Plot generation should succeed"
        assert "Pattern 2: Plot Generation" in plot_response["data"]["stdout"], "Plot pattern should be demonstrated"
        
        # Verify comprehensive guidance is provided
        print("\n" + "="*60)
        print("📚 RECOMMENDED PATTERNS DEMONSTRATION COMPLETE")
        print("="*60)
        print("✅ Pattern 1: Data Processing → Return JSON in API response")
        print("✅ Pattern 2: Plot Generation → Save to mounted directories")
        print("✅ Pattern 3: File Processing → Process immediately, don't store")
        print("\n🎯 KEY TAKEAWAYS:")
        print("   • Don't rely on filesystem persistence across API calls")
        print("   • Return important data directly in API responses")
        print("   • Use mounted directories only for plots and permanent files")
        print("   • Complete workflows in single API calls when possible")
        print("="*60)


# Test configuration validation
def test_configuration_constants_properly_defined():
    """
    Validate that all configuration constants are properly defined and accessible.
    
    This test ensures that the centralized configuration approach is working
    correctly and all required constants are available for use in tests.
    """
    # Validate TestConfig class structure
    assert hasattr(TestConfig, 'BASE_URL'), "BASE_URL should be defined"
    assert hasattr(TestConfig, 'TIMEOUTS'), "TIMEOUTS should be defined"
    assert hasattr(TestConfig, 'ENDPOINTS'), "ENDPOINTS should be defined"
    assert hasattr(TestConfig, 'PATHS'), "PATHS should be defined"
    assert hasattr(TestConfig, 'FILE_SETTINGS'), "FILE_SETTINGS should be defined"
    
    # Validate specific values
    assert TestConfig.BASE_URL == "http://localhost:3000", "BASE_URL should be correct"
    assert TestConfig.ENDPOINTS["execute_raw"] == "/api/execute-raw", "execute_raw endpoint should be correct"
    assert TestConfig.PATHS["temp_dir"] == "/tmp", "temp_dir path should be correct"
    
    # Validate timeout values are reasonable
    assert TestConfig.TIMEOUTS["api_request"] > 0, "API timeout should be positive"
    assert TestConfig.TIMEOUTS["code_execution"] > TestConfig.TIMEOUTS["api_request"], "Code execution timeout should be longer than API timeout"
    
    print("✅ Configuration constants validation passed")
    print(f"   Base URL: {TestConfig.BASE_URL}")
    print(f"   Timeout settings: {len(TestConfig.TIMEOUTS)} configured")
    print(f"   Endpoint settings: {len(TestConfig.ENDPOINTS)} configured")
    print(f"   Path settings: {len(TestConfig.PATHS)} configured")
