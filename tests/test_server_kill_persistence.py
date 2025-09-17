#!/usr/bin/env python3
"""
INSTRUCTIONS FOR MANUAL FILESYSTEM PERSISTENCE TESTING

This file documents how to manually test filesystem persistence across server kills.
The automated version had issues with terminal management, so manual testing is required.

MANUAL TEST PROCEDURE:
======================

1. START SERVER:
   node src/server.js
   (Wait for "Pyodide initialized" message)

2. CREATE TEST FILES:
   python tests/test_simple_filesystem.py
   (This will create files and verify they appear locally)

3. KILL SERVER:
   Stop-Process -Name node -Force
   (Or Ctrl+C in server terminal)

4. VERIFY LOCAL FILES STILL EXIST:
   Test-Path "plots/matplotlib/simple_test_*.txt"
   (Should return True)

5. RESTART SERVER:
   node src/server.js
   (Wait for initialization)

6. TEST IF PYODIDE CAN SEE PERSISTED FILES:
   Use API to check if files still exist in Pyodide filesystem

EXPECTED RESULTS:
================
- Files in /home/pyodide/plots/matplotlib/ should persist (mounted directory)
- Files in /tmp/ should NOT persist (virtual filesystem)
- Local filesystem files survive server kills
- Pyodide can access files in mounted directories after restart

CONCLUSION:
===========
For production: Use mounted directories (/plots, /uploads) for persistent storage.
Avoid /tmp for files that need to survive server restarts.
"""

import unittest
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:3000"

class FilesystemPersistenceDocumentation(unittest.TestCase):
    """Documentation and simple tests for filesystem persistence."""
    
    def test_create_persistence_test_files(self):
        """Create test files for manual persistence testing."""
        
        timestamp = int(time.time() * 1000)
        
        create_code = f'''
from pathlib import Path

# Create test files for persistence testing
tmp_file = Path("/tmp/persistence_test_{timestamp}.txt")
tmp_file.write_text("This file should NOT persist after server restart")

plots_file = Path('/home/pyodide/plots/matplotlib/persistence_test_{timestamp}.txt")
plots_file.parent.mkdir(parents=True, exist_ok=True)
plots_file.write_text("This file SHOULD persist after server restart")

{{
    "timestamp": {timestamp},
    "tmp_file_created": tmp_file.exists(),
    "plots_file_created": plots_file.exists(),
    "files_for_testing": [
        "/tmp/persistence_test_{timestamp}.txt",
        "/home/pyodide/plots/matplotlib/persistence_test_{timestamp}.txt"
    ]
}}
'''
        
        response = requests.post(f"{BASE_URL}/api/execute", json={"code": create_code})
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertTrue(result.get("success"))
        
        data = result["result"]
        print(f"\n📁 Created test files for timestamp {timestamp}:")
        print(f"   /tmp file: {data['tmp_file_created']}")
        print(f"   /plots file: {data['plots_file_created']}")
        
        # Check local file
        local_file = Path(__file__).parent.parent / "plots" / "matplotlib" / f"persistence_test_{timestamp}.txt"
        self.assertTrue(local_file.exists())
        print(f"   Local file exists: {local_file.exists()}")
        
        print(f"\n� MANUAL TEST INSTRUCTIONS:")
        print(f"1. Note the timestamp: {timestamp}")
        print(f"2. Kill the server: Stop-Process -Name node -Force")
        print(f"3. Check local file: Test-Path 'plots/matplotlib/persistence_test_{timestamp}.txt'")
        print(f"4. Restart server: node src/server.js")
        print(f"5. Test if Pyodide can still see the files")
        
        return timestamp

if __name__ == "__main__":
    print(__doc__)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("\n✅ Server is running - you can run the test")
            unittest.main(verbosity=2)
        else:
            print("\n❌ Server not healthy")
    except:
        print("\n❌ Server not running. Start with: node src/server.js")
