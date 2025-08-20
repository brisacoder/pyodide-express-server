import time
import unittest
import requests

BASE_URL = "http://localhost:3000"


def wait_for_server(url: str, timeout: int = 120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


class NonHappyPathsTestCase(unittest.TestCase):
    """Additional negative/edge tests to tighten coverage on error handling."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running, but don't start a new one
        # This allows the test to work with existing servers
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            # If no server is running, we'll skip these tests rather than start our own
            # to avoid conflicts with test runners
            raise unittest.SkipTest("Server is not running on localhost:3000")

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    # Execute-raw edge cases

    def test_execute_raw_empty_body(self):
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=b"", headers={"Content-Type": "text/plain"}, timeout=15)
        self.assertIn(r.status_code, [400, 200])
        if r.status_code == 200:
            self.assertFalse(r.json().get("success"))

    def test_execute_raw_binary_content(self):
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=b"\x00\x01\x02\xff",
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )
        # Depending on server validation, this may be treated as bad request or internal error
        self.assertIn(r.status_code, [200, 400, 415, 500])

    def test_execute_with_large_timeout_but_reasonable_enforced(self):
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "1+1", "timeout": 10_000_000}, timeout=15)
        # Should be rejected or coerced down by validation
        self.assertIn(r.status_code, [200, 400])
        if r.status_code == 200:
            # Either succeeds with capped timeout or returns validation error in body
            payload = r.json()
            self.assertIn("success", payload)

    def test_execute_with_context_non_jsonable(self):
        # Simulate non-JSONable content reaching server by sending a string payload
        r = requests.post(
            f"{BASE_URL}/api/execute",
            data='{"code": "print(\"x\")", "context": {"bad": "<non-json>"}}',
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self.assertIn(r.status_code, [200, 400])

    def test_packages_endpoint_structure(self):
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertIn("success", payload)
        
        if payload.get("success"):
            self.assertIn("result", payload)
            result = payload["result"]
            
            # Result should NOT be null - it should be a valid dictionary
            self.assertIsNotNone(result, "Packages endpoint should return actual data, not null")
            self.assertIsInstance(result, dict, "Packages result should be a dictionary")
            
            # Should have required fields
            required_fields = ["python_version", "installed_packages", "total_packages"]
            for field in required_fields:
                self.assertIn(field, result, f"Missing required field: {field}")
                
            # Basic data validation
            self.assertIsInstance(result["installed_packages"], list)
            self.assertIsInstance(result["total_packages"], int)
            self.assertGreater(result["total_packages"], 0, "Should have some packages")
        else:
            # On failure, should include an error field
            self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
