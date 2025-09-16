import time
import unittest

import requests

BASE_URL = "http://localhost:3000"


def wait_for_server(url: str, timeout: int = 30):
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


class PackageManagementTestCase(unittest.TestCase):
    """Comprehensive tests for package installation and listing functionality."""

    @classmethod
    def setUpClass(cls):
        wait_for_server(f"{BASE_URL}/health")

    def test_packages_endpoint_returns_valid_data(self):
        """Test that packages endpoint returns proper structure with actual data."""
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        
        payload = r.json()
        self.assertIn("success", payload)
        self.assertTrue(payload["success"], f"Packages endpoint failed: {payload}")
        
        # Result should not be null - it should be a dictionary
        self.assertIn("result", payload)
        result = payload["result"]
        self.assertIsNotNone(result, "Packages result should not be null")
        self.assertIsInstance(result, dict, "Packages result should be a dictionary")
        
        # Check required fields
        required_fields = ["python_version", "installed_packages", "total_packages"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Validate data types
        self.assertIsInstance(result["python_version"], str)
        self.assertIsInstance(result["installed_packages"], list)
        self.assertIsInstance(result["total_packages"], int)
        
        # Should have some packages installed by default
        self.assertGreater(result["total_packages"], 0, "Should have some packages installed")
        self.assertGreater(len(result["installed_packages"]), 0, "Should have package names")

    def test_install_package_and_verify_in_list(self):
        """Test installing a package and verifying it appears in the packages list."""
        test_package = "beautifulsoup4"
        
        # Install the package
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": test_package},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200)
        install_result = r.json()
        self.assertTrue(install_result.get("success"), f"Package installation failed: {install_result}")
        
        # Verify package appears in packages list
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        
        packages_result = r.json()
        self.assertTrue(packages_result.get("success"), f"Packages list failed: {packages_result}")
        
        installed_packages = packages_result["result"]["installed_packages"]
        self.assertIn(test_package, installed_packages, 
                     f"{test_package} not found in installed packages: {installed_packages}")

    def test_install_package_and_verify_usable(self):
        """Test that installed packages can actually be imported and used."""
        test_package = "beautifulsoup4"
        
        # Install the package
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": test_package},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200)
        
        # Test that the package can be imported and used
        test_code = '''
from bs4 import BeautifulSoup

# Test basic functionality
html = "<html><body><p>Hello World</p></body></html>"
soup = BeautifulSoup(html, 'html.parser')
result = soup.find('p').text
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        exec_result = r.json()
        self.assertTrue(exec_result.get("success"), f"Code execution failed: {exec_result}")
        self.assertEqual(exec_result.get("result"), "Hello World")

    def test_package_dependencies_included(self):
        """Test that package dependencies are also listed."""
        # beautifulsoup4 has dependencies like soupsieve
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        
        packages_result = r.json()
        installed_packages = packages_result["result"]["installed_packages"]
        
        # Check for known dependencies
        expected_packages = ["beautifulsoup4", "soupsieve"]
        for pkg in expected_packages:
            self.assertIn(pkg, installed_packages, 
                         f"Expected package {pkg} not found in: {installed_packages}")

    def test_loaded_modules_vs_installed_packages(self):
        """Test that loaded_modules and installed_packages are different concepts."""
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()["result"]
        
        # Should have both fields
        self.assertIn("loaded_modules", result)
        self.assertIn("installed_packages", result)
        
        loaded_modules = result["loaded_modules"]
        installed_packages = result["installed_packages"]
        
        # They should be different lists
        self.assertIsInstance(loaded_modules, list)
        self.assertIsInstance(installed_packages, list)
        
        # Loaded modules should include basic Python modules
        basic_modules = ["sys", "os", "json", "re"]
        for module in basic_modules:
            self.assertIn(module, loaded_modules, 
                         f"Basic module {module} should be loaded")

    def test_package_count_accuracy(self):
        """Test that total_packages count matches the actual list length."""
        r = requests.get(f"{BASE_URL}/api/packages", timeout=20)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()["result"]
        total_packages = result["total_packages"]
        installed_packages = result["installed_packages"]
        
        self.assertEqual(total_packages, len(installed_packages),
                        f"total_packages ({total_packages}) doesn't match list length ({len(installed_packages)})")


if __name__ == "__main__":
    unittest.main()
