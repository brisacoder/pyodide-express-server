import unittest
import requests
from pathlib import Path

BASE_URL = "http://localhost:3000"

class TestLocalFSAccessViaPyodideAPI(unittest.TestCase):
    """
    Test that a file placed on the local filesystem is readable from Pyodide
    using the /api/execute-raw endpoint.
    """
    @classmethod
    def setUpClass(cls):
        cls.local_dir = Path(__file__).parent.parent / "plots" 
        cls.test_filename = "test_pyodide_local_access_api.txt"
        cls.local_file_path = cls.local_dir / cls.test_filename
        cls.local_dir.mkdir(parents=True, exist_ok=True)
        with open(cls.local_file_path, "w", encoding="utf-8") as f:
            f.write("Hello from local FS via API!")

    # Do not delete the file after the test so it remains for inspection

    def test_pyodide_can_read_local_file_via_api(self):
        # Python code to run inside Pyodide
        py_code = f"""
with open(r'/plots/{self.test_filename}', 'r', encoding='utf-8') as f:
    content = f.read()
print(content)
"""
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=py_code,
            headers={"Content-Type": "text/plain"},
            timeout=20
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        print(result)
        self.assertTrue(result.get("success"), f"API call failed: {result}")
        output = result.get("stdout", "")
        self.assertIn("Hello from local FS via API!", output)

if __name__ == "__main__":
    unittest.main()
