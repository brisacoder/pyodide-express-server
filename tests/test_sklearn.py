import time
import unittest

import requests

BASE_URL = "http://localhost:3000"


def wait_for_server(url: str, timeout: int = 180):
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


class SklearnTestCase(unittest.TestCase):
    """Run scikit-learn workloads (CV and GridSearch) inside Pyodide."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running, but don't start a new one
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            # If no server is running, we'll skip these tests
            raise unittest.SkipTest("Server is not running on localhost:3000")

        # Ensure scikit-learn is available (Pyodide package). Give it ample time.
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "scikit-learn"},
            timeout=300,
        )
        # Installation may already be present; both should return 200
        assert r.status_code == 200, f"Failed to reach install endpoint: {r.status_code}"
        
        # Verify availability by attempting an import inside the runtime
        check = requests.post(
            f"{BASE_URL}/api/execute",
            json={"code": "import sklearn as sk; sk.__version__"},
            timeout=120,
        )
        cls.has_sklearn = False
        if check.status_code == 200:
            payload = check.json()
            cls.has_sklearn = bool(payload.get("success"))
            
        # IMPORTANT: Also verify that scikit-learn appears in the packages list
        if cls.has_sklearn:
            packages_r = requests.get(f"{BASE_URL}/api/packages", timeout=30)
            if packages_r.status_code == 200:
                packages_data = packages_r.json()
                if packages_data.get("success") and packages_data.get("result"):
                    installed_packages = packages_data["result"].get("installed_packages", [])
                    if "scikit-learn" not in installed_packages:
                        print(f"WARNING: scikit-learn not found in packages list: {installed_packages}")
                        # Still proceed with tests, but log the issue
            payload = check.json()
            cls.has_sklearn = bool(payload.get("success"))

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    def test_cross_validation_logistic_regression(self):
        """Small cross-validation using LogisticRegression to validate runtime path."""
        if not getattr(self.__class__, "has_sklearn", False):
            self.skipTest("scikit-learn not available in this Pyodide environment")
        code = r'''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = make_classification(n_samples=120, n_features=8, n_informative=5, n_redundant=1, random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf = LogisticRegression(solver="liblinear", max_iter=200, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
result = {"n_scores": len(scores), "scores": list(map(float, scores)), "mean": float(np.mean(scores))}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        result = data.get("result")
        self.assertEqual(result.get("n_scores"), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in result.get("scores", [])))
        self.assertGreaterEqual(result.get("mean", 0.0), 0.5)  # should be reasonably accurate on synthetic data

    def test_grid_search_logistic_regression(self):
        """Run a tiny GridSearchCV to ensure more involved sklearn paths work."""
        if not getattr(self.__class__, "has_sklearn", False):
            self.skipTest("scikit-learn not available in this Pyodide environment")
        code = r'''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

X, y = make_classification(n_samples=150, n_features=10, n_informative=6, n_redundant=2, random_state=0)
param_grid = {"C": [0.1, 1.0], "penalty": ["l1", "l2"]}
base = LogisticRegression(solver="liblinear", max_iter=300, random_state=0)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=1)
gs.fit(X, y)
result = {"best_params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in gs.best_params_.items()},
 "best_score": float(gs.best_score_),
 "n_candidates": len(gs.cv_results_["params"])}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=180)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        result = data.get("result")
        self.assertIsNotNone(result, f"Result is None. Full response: {data}")
        self.assertIn("best_params", result)
        self.assertIn("C", result["best_params"])
        self.assertIn("penalty", result["best_params"])
        self.assertIsInstance(result.get("best_score"), float)
        self.assertEqual(result.get("n_candidates"), 4)

    def test_grid_search_invalid_param_raises(self):
        """Non-happy path: invalid parameter in grid should produce an error response."""
        if not getattr(self.__class__, "has_sklearn", False):
            self.skipTest("scikit-learn not available in this Pyodide environment")
        code = r'''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X, y = make_classification(n_samples=50, n_features=5, random_state=0)
param_grid = {"not_a_param": [1, 10]}  # invalid for LogisticRegression
gs = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid=param_grid, cv=2)
gs.fit(X, y)
"should_not_reach_here"
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data.get("success"))
        self.assertIn("invalid parameter", data.get("error", "").lower())

    def test_sklearn_package_listed(self):
        """Test that scikit-learn appears in the packages list after installation."""
        if not getattr(self.__class__, "has_sklearn", False):
            self.skipTest("scikit-learn not available in this Pyodide environment")
            
        r = requests.get(f"{BASE_URL}/api/packages", timeout=30)
        self.assertEqual(r.status_code, 200)
        
        data = r.json()
        self.assertTrue(data.get("success"), f"Packages endpoint failed: {data}")
        
        result = data.get("result")
        self.assertIsNotNone(result, "Packages result should not be null")
        
        installed_packages = result.get("installed_packages", [])
        self.assertIn("scikit-learn", installed_packages, 
                     f"scikit-learn not found in installed packages: {installed_packages}")
        
        # Also check that it has a reasonable number of total packages
        total_packages = result.get("total_packages", 0)
        self.assertGreater(total_packages, 10, 
                          f"Expected more packages, got {total_packages}")


if __name__ == "__main__":
    unittest.main()
