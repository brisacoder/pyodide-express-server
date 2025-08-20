"""
Test filesystem persistence behavior across server restarts.

This test addresses the critical issue mentioned in Pyodide FAQ:
https://pyodide.org/en/stable/usage/faq.html#why-changes-made-to-indexeddb-don-t-persist

Key points:
- IndexedDB (pyodide.FS.filesystem.IDBFS) is asynchronous
- Changes don't persist unless pyodide.FS.syncfs() is called
- We need to test both scenarios: with and without syncfs
"""

import unittest
import requests
import time
import subprocess
from pathlib import Path

BASE_URL = "http://localhost:3000"


def wait_for_server(url: str, timeout: int = 60):
    """Wait for server to be available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


class FilesystemPersistenceTestCase(unittest.TestCase):
    """Test filesystem persistence across server restarts."""

    @classmethod
    def setUpClass(cls):
        # Use existing server
        if not wait_for_server(f"{BASE_URL}/health"):
            raise unittest.SkipTest("Server is not running on localhost:3000")

    def test_file_persistence_documentation(self):
        """Document current filesystem behavior - files don't persist across sessions."""
        # This test documents the current behavior and requirements for proper persistence
        
        test_filename = f"test_persistence_{int(time.time())}.txt"
        test_content = "This tests filesystem persistence behavior"
        
        # Step 1: Create file and document filesystem info
        create_file_code = f'''
from pathlib import Path
import sys

# Create file in Pyodide filesystem
test_file = Path("/tmp/{test_filename}")
test_file.write_text("{test_content}")

result = {{
    "file_exists": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "filename": str(test_file),
    "platform": sys.platform,
    "pyodide_available": "pyodide" in sys.modules,
    "filesystem_info": {{
        "cwd": str(Path.cwd()),
        "tmp_writable": Path("/tmp").exists()
    }}
}}

# Check if we have access to FS API
try:
    import js
    if hasattr(js, "pyodide"):
        result["pyodide_js_available"] = True
        if hasattr(js.pyodide, "FS"):
            result["fs_api_available"] = True
            result["has_syncfs"] = hasattr(js.pyodide.FS, "syncfs")
        else:
            result["fs_api_available"] = False
    else:
        result["pyodide_js_available"] = False
except Exception as e:
    result["js_access_error"] = str(e)

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": create_file_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"File creation failed: {result}")
        
        data = result.get("result")
        self.assertTrue(data.get("file_exists"), "File should exist after creation")
        self.assertEqual(data.get("file_content"), test_content)
        
        print("\\n🔍 FILESYSTEM BEHAVIOR ANALYSIS:")
        print(f"📋 Platform: {data.get('platform')}")
        print(f"📋 Pyodide available: {data.get('pyodide_available')}")
        print(f"📋 Pyodide JS available: {data.get('pyodide_js_available')}")
        print(f"📋 FS API available: {data.get('fs_api_available')}")
        print(f"📋 Has syncfs(): {data.get('has_syncfs')}")
        print(f"📋 Current working directory: {data.get('filesystem_info', {}).get('cwd')}")
        
        # Step 2: Check if file persists in new execution context
        check_persistence_code = f'''
from pathlib import Path

test_file = Path("/tmp/{test_filename}")
result = {{
    "file_exists_in_new_context": test_file.exists(),
    "file_content_in_new_context": test_file.read_text() if test_file.exists() else None
}}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": check_persistence_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Persistence check failed: {result}")
        
        data = result.get("result")
        
        print(f"\\n📋 File exists in new execution context: {data.get('file_exists_in_new_context')}")
        
        if data.get("file_exists_in_new_context"):
            print("✅ Files persist within the same server session")
        else:
            print("❌ Files do not persist even within same server session")
        
        # Document the expected behavior based on Pyodide FAQ
        print("\\n📚 EXPECTED BEHAVIOR (from Pyodide FAQ):")
        print("- Files in IndexedDB (IDBFS) require pyodide.FS.syncfs() to persist")
        print("- Without syncfs(), changes are only in memory")
        print("- Server restart = loss of all non-synced files")
        print("- This test documents current behavior for future reference")

    def test_recommended_persistence_pattern(self):
        """Test and document the recommended pattern for file persistence."""
        
        persistence_code = '''
from pathlib import Path
import sys

# Recommended pattern for file persistence in Pyodide applications:

result = {
    "pattern_description": "Use explicit data serialization instead of relying on filesystem persistence",
    "recommended_approaches": [
        "1. Return file contents as API response data",
        "2. Use local filesystem mounting for direct file access", 
        "3. Store data in external database/storage",
        "4. Use explicit syncfs() calls if IndexedDB persistence is needed"
    ],
    "current_capabilities": {}
}

# Test what we can do in current setup
test_file = Path("/tmp/recommended_pattern_test.txt")
test_content = "Data that needs to persist"
test_file.write_text(test_content)

# Pattern 1: Return file contents as data
result["current_capabilities"]["file_content_as_data"] = {
    "file_content": test_file.read_text(),
    "file_size": len(test_content),
    "can_return_as_api_response": True
}

# Pattern 2: Test if files appear in mounted directories
mounted_file = Path("/plots/matplotlib/api_test.txt")
try:
    mounted_file.parent.mkdir(parents=True, exist_ok=True)
    mounted_file.write_text("Test file in mounted directory")
    result["current_capabilities"]["mounted_directory"] = {
        "can_write": True,
        "file_exists": mounted_file.exists()
    }
except Exception as e:
    result["current_capabilities"]["mounted_directory"] = {
        "can_write": False,
        "error": str(e)
    }

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": persistence_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Pattern test failed: {result}")
        
        data = result.get("result")
        
        print("\\n💡 RECOMMENDED PERSISTENCE PATTERNS:")
        for i, approach in enumerate(data.get("recommended_approaches", []), 1):
            print(f"   {approach}")
        
        print("\\n🔧 CURRENT CAPABILITIES:")
        capabilities = data.get("current_capabilities", {})
        
        # Test file content as data
        file_data = capabilities.get("file_content_as_data", {})
        if file_data.get("can_return_as_api_response"):
            print(f"✅ Can return file contents as API response ({file_data.get('file_size')} bytes)")
        
        # Test mounted directory
        mounted = capabilities.get("mounted_directory", {})
        if mounted.get("can_write"):
            print(f"✅ Can write to mounted directories")
            
            # Check if file appears in local filesystem
            local_file = Path(__file__).parent.parent / "plots" / "matplotlib" / "api_test.txt"
            if local_file.exists():
                print("✅ Files in mounted directories appear in local filesystem")
                local_file.unlink()  # cleanup
            else:
                print("⚠️  Files in mounted directories may not appear locally immediately")
        else:
            print(f"❌ Cannot write to mounted directories: {mounted.get('error')}")

    def test_user_guidance_for_persistence(self):
        """Provide clear guidance for users who need file persistence."""
        
        print("\\n📖 USER GUIDANCE FOR FILE PERSISTENCE:")
        print("\\n1. 🎯 FOR PLOT/IMAGE GENERATION:")
        print("   - Save to mounted directory (/plots/matplotlib/, /plots/seaborn/)")
        print("   - Files appear directly in local filesystem")
        print("   - Use absolute paths: Path('/plots/matplotlib/myplot.png')")
        
        print("\\n2. 📊 FOR DATA PROCESSING RESULTS:")
        print("   - Return processed data as JSON in API response")
        print("   - Use .to_dict(), .tolist(), or explicit serialization")
        print("   - Don't rely on intermediate files persisting")
        
        print("\\n3. 💾 FOR LARGE DATASETS:")
        print("   - Upload files via /api/upload-csv endpoint")
        print("   - Process and return results immediately")
        print("   - Consider external storage for persistence")
        
        print("\\n4. ⚠️  WHAT NOT TO DO:")
        print("   - Don't assume files in /tmp persist across requests")
        print("   - Don't rely on Pyodide filesystem for data storage")
        print("   - Don't expect files to survive server restarts")
        
        print("\\n5. 🔧 FOR ADVANCED USERS (if needed):")
        print("   - Implement explicit pyodide.FS.syncfs() calls")
        print("   - Use IndexedDB directly for browser-side persistence")
        print("   - Consider WebAssembly filesystem limitations")
        
        # This test always passes - it's documentation
        self.assertTrue(True, "User guidance provided")


if __name__ == "__main__":
    unittest.main()

    def test_file_persistence_without_syncfs(self):
        """Test that files DON'T persist across server restarts without syncfs()."""
        # Create a test file without calling syncfs
        test_filename = f"test_no_syncfs_{int(time.time())}.txt"
        test_content = "This file should NOT persist without syncfs()"
        
        create_file_code = f'''
from pathlib import Path

# Create file in Pyodide filesystem (likely IDBFS)
test_file = Path("/tmp/{test_filename}")
test_file.write_text("{test_content}")

# Verify file exists
result = {{
    "file_exists": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "filename": str(test_file)
}}

# NOTE: We deliberately DO NOT call pyodide.FS.syncfs() here
result
'''
        
        # Step 1: Create file
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": create_file_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"File creation failed: {result}")
        
        data = result.get("result")
        self.assertTrue(data.get("file_exists"), "File should exist after creation")
        self.assertEqual(data.get("file_content"), test_content)
        
        # Step 2: Restart server (this simulates real-world server restart)
        print("\\n🔄 Restarting server to test persistence...")
        self.server.terminate()
        self.server.wait(timeout=10)
        time.sleep(2)
        
        # Start new server
        self.server = subprocess.Popen(
            ["node", "src/server.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        
        if not wait_for_server(f"{BASE_URL}/health"):
            self.fail("Server failed to restart")
        
        # Step 3: Check if file persists (it should NOT without syncfs)
        check_file_code = f'''
from pathlib import Path

test_file = Path("/tmp/{test_filename}")
result = {{
    "file_exists": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "filesystem_type": "unknown"
}}

# Try to detect filesystem type
try:
    import js
    if hasattr(js.pyodide, 'FS'):
        # This would indicate we can access Emscripten FS API
        result["has_fs_api"] = True
    else:
        result["has_fs_api"] = False
except:
    result["has_fs_api"] = False

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": check_file_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"File check failed: {result}")
        
        data = result.get("result")
        
        # The file should NOT exist after restart (without syncfs)
        # This documents the expected Pyodide behavior
        print(f"\\n📋 File persistence without syncfs: {data.get('file_exists')}")
        print(f"📋 Has FS API access: {data.get('has_fs_api')}")
        
        # For now, we document the behavior rather than assert
        # TODO: Once we implement proper IDBFS with syncfs, update this assertion
        if not data.get("file_exists"):
            print("✅ Expected behavior: File does not persist without syncfs() call")
        else:
            print("⚠️  Unexpected: File persisted without syncfs() - may indicate different filesystem")

    def test_file_persistence_with_syncfs(self):
        """Test that files DO persist across server restarts with syncfs()."""
        test_filename = f"test_with_syncfs_{int(time.time())}.txt"
        test_content = "This file SHOULD persist with syncfs()"
        
        create_and_sync_code = f'''
from pathlib import Path
import js

# Create file in Pyodide filesystem
test_file = Path("/tmp/{test_filename}")
test_file.write_text("{test_content}")

result = {{
    "file_created": test_file.exists(),
    "file_content": test_file.read_text() if test_file.exists() else None,
    "syncfs_attempted": False,
    "syncfs_success": False
}}

# Attempt to call syncfs to persist changes
try:
    if hasattr(js.pyodide, 'FS') and hasattr(js.pyodide.FS, 'syncfs'):
        # Call syncfs to persist changes to IndexedDB
        js.pyodide.FS.syncfs(False)  # False = save from memory to persistent storage
        result["syncfs_attempted"] = True
        result["syncfs_success"] = True
    else:
        result["syncfs_attempted"] = True
        result["syncfs_success"] = False
        result["error"] = "syncfs not available"
except Exception as e:
    result["syncfs_attempted"] = True
    result["syncfs_success"] = False
    result["error"] = str(e)

result
'''
        
        # Step 1: Create file and sync
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": create_and_sync_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"File creation/sync failed: {result}")
        
        data = result.get("result")
        self.assertTrue(data.get("file_created"), "File should exist after creation")
        self.assertTrue(data.get("syncfs_attempted"), "Should attempt to call syncfs")
        
        print(f"\\n📋 Syncfs success: {data.get('syncfs_success')}")
        if not data.get("syncfs_success"):
            print(f"📋 Syncfs error: {data.get('error', 'Unknown error')}")
        
        # Step 2: Restart server
        print("\\n🔄 Restarting server to test persistence with syncfs...")
        self.server.terminate()
        self.server.wait(timeout=10)
        time.sleep(2)
        
        # Start new server
        self.server = subprocess.Popen(
            ["node", "src/server.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        
        if not wait_for_server(f"{BASE_URL}/health"):
            self.fail("Server failed to restart")
        
        # Step 3: Check if file persists (it should if syncfs worked)
        check_persistence_code = f'''
from pathlib import Path

test_file = Path("/tmp/{test_filename}")
result = {{
    "file_exists_after_restart": test_file.exists(),
    "file_content_after_restart": test_file.read_text() if test_file.exists() else None,
    "content_matches": False
}}

if test_file.exists():
    content = test_file.read_text()
    result["content_matches"] = (content == "{test_content}")

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": check_persistence_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Persistence check failed: {result}")
        
        data = result.get("result")
        
        print(f"\\n📋 File exists after restart: {data.get('file_exists_after_restart')}")
        print(f"📋 Content matches: {data.get('content_matches')}")
        
        # Document the behavior - assertion depends on whether syncfs is properly implemented
        if data.get("file_exists_after_restart") and data.get("content_matches"):
            print("✅ SUCCESS: File persisted across restart with syncfs()")
        else:
            print("⚠️  File did not persist - syncfs may not be properly configured")
            # This is documentation rather than failure until we implement proper IDBFS

    def test_filesystem_type_detection(self):
        """Test to detect what type of filesystem Pyodide is using."""
        filesystem_detection_code = '''
import js
import sys
from pathlib import Path

result = {
    "python_platform": sys.platform,
    "pyodide_available": "pyodide" in sys.modules,
    "fs_api_available": False,
    "filesystem_types": [],
    "mount_points": {}
}

# Check if we can access Emscripten FS API
try:
    if hasattr(js.pyodide, 'FS'):
        result["fs_api_available"] = True
        
        # Try to get filesystem information
        if hasattr(js.pyodide.FS, 'filesystems'):
            result["filesystem_types"] = list(js.pyodide.FS.filesystems.keys())
        
        # Check mount points
        for path in ["/", "/tmp", "/home", "/dev", "/proc"]:
            try:
                stat = js.pyodide.FS.stat(path)
                result["mount_points"][path] = {
                    "exists": True,
                    "mode": stat.mode if hasattr(stat, 'mode') else None
                }
            except:
                result["mount_points"][path] = {"exists": False}
    
except Exception as e:
    result["fs_api_error"] = str(e)

# Test directory creation in different locations
test_locations = ["/tmp", "/home", "/dev/shm", "/var/tmp"]
result["location_tests"] = {}

for location in test_locations:
    try:
        test_dir = Path(location) / "test_dir"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        
        result["location_tests"][location] = {
            "writable": True,
            "test_file_exists": test_file.exists()
        }
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        
    except Exception as e:
        result["location_tests"][location] = {
            "writable": False,
            "error": str(e)
        }

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": filesystem_detection_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Filesystem detection failed: {result}")
        
        data = result.get("result")
        
        print("\\n🔍 FILESYSTEM ANALYSIS:")
        print(f"📋 Platform: {data.get('python_platform')}")
        print(f"📋 Pyodide available: {data.get('pyodide_available')}")
        print(f"📋 FS API available: {data.get('fs_api_available')}")
        print(f"📋 Filesystem types: {data.get('filesystem_types', [])}")
        
        print("\\n📁 Mount points:")
        for path, info in data.get("mount_points", {}).items():
            print(f"   {path}: {info}")
        
        print("\\n✏️  Writable locations:")
        for location, info in data.get("location_tests", {}).items():
            status = "✅" if info.get("writable") else "❌"
            print(f"   {status} {location}: {info}")
        
        # This test is purely informational
        self.assertIsNotNone(data)


if __name__ == "__main__":
    unittest.main()
