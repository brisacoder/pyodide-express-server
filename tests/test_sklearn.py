"""
Comprehensive scikit-learn functionality tests for Pyodide Express Server.

This module tests machine learning workflows including cross-validation,
grid search, and error handling using scikit-learn within the Pyodide
environment. Uses only the /api/execute-raw endpoint as specified.
"""

import pytest
import requests
import re


class TestSklearnWorkflows:
    """Test scikit-learn machine learning workflows in Pyodide environment.

    This test class validates that complex machine learning operations
    can be executed successfully through the /api/execute-raw endpoint,
    including cross-validation, hyperparameter tuning, and error handling.
    """

    @pytest.fixture(autouse=True)
    def setup_sklearn_availability(self, server, base_url, api_timeout):
        """Verify scikit-learn is available in the Pyodide environment.

        This fixture runs before each test to ensure scikit-learn can be
        imported. Tests are skipped if scikit-learn is not available.

        Args:
            server: Server fixture from conftest.py
            base_url: Base URL fixture from conftest.py
            api_timeout: API timeout fixture from conftest.py

        Yields:
            bool: True if scikit-learn is available, False otherwise
        """
        # Test sklearn availability by attempting import
        check_code = """
import sklearn as sk
print(f"scikit-learn version: {sk.__version__}")
sk.__version__
"""

        try:
            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=check_code,
                headers={"Content-Type": "text/plain"},
                timeout=api_timeout * 4  # Give extra time for sklearn import
            )

            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                stdout = result.get("data", {}).get("stdout", "")
                has_sklearn = success and "scikit-learn version:" in stdout
            else:
                has_sklearn = False

        except Exception:
            has_sklearn = False

        if not has_sklearn:
            pytest.skip("scikit-learn not available in Pyodide environment")

        self.has_sklearn = has_sklearn
        yield has_sklearn

    def test_given_synthetic_classification_data_when_cross_validation_performed_then_returns_valid_scores(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test cross-validation with LogisticRegression on synthetic data.

        This test validates that scikit-learn's cross-validation
        functionality works correctly within the Pyodide environment.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Synthetic classification dataset (120 samples, 8 features)
            - LogisticRegression with liblinear solver
            - 3-fold stratified cross-validation

        Expected Output:
            - 3 cross-validation scores between 0.0 and 1.0
            - Mean accuracy >= 0.5 (reasonable for synthetic data)
            - Valid JSON response following API contract

        Example:
            API Response:
            {
                "success": true,
                "data": {
                    "result": {
                        "stdout": "...",
                        "stderr": "",
                        "executionTime": 1234
                    }
                },
                "error": null,
                "meta": {"timestamp": "..."}
            }
        """
        code = '''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from pathlib import Path

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=120,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42
)

# Configure cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf = LogisticRegression(solver="liblinear", max_iter=200, random_state=42)

# Perform cross-validation
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

# Format results for output
result = {
    "n_scores": len(scores),
    "scores": list(map(float, scores)),
    "mean": float(np.mean(scores))
}

print(f"Cross-validation completed with {result['n_scores']} folds")
print(f"Individual scores: {result['scores']}")
print(f"Mean accuracy: {result['mean']:.4f}")

# Return result for validation
result
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 4  # Give extra time for ML operations
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        assert data.get("success") is True, f"Expected success=True: {data}"
        assert "data" in data, f"Missing 'data' field: {data}"
        assert data.get("error") is None, f"Unexpected error: {data}"
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate execution results
        result_data = data["data"]
        assert "stdout" in result_data, f"Missing stdout: {result_data}"
        assert "Cross-validation completed" in result_data["stdout"]
        assert "Individual scores:" in result_data["stdout"]
        assert "Mean accuracy:" in result_data["stdout"]

        # Extract and validate numerical results from stdout
        stdout_lines = result_data["stdout"].strip().split('\n')
        mean_line = [line for line in stdout_lines if "Mean accuracy:" in line]
        assert len(mean_line) == 1, f"Expected 1 mean line: {mean_line}"

        mean_accuracy = float(mean_line[0].split(": ")[1])
        assert 0.0 <= mean_accuracy <= 1.0, f"Mean accuracy OOR: {mean_accuracy}"  # noqa: E501
        assert mean_accuracy >= 0.5, f"Mean accuracy low: {mean_accuracy}"

    def test_given_hyperparameter_grid_when_grid_search_performed_then_finds_optimal_parameters(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test GridSearchCV with hyperparameter optimization.

        This test validates that scikit-learn's grid search functionality
        works correctly for hyperparameter tuning within Pyodide.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Synthetic classification dataset (150 samples, 10 features)
            - Parameter grid with C values [0.1, 1.0] and penalties
            - 3-fold stratified cross-validation
            - Accuracy scoring metric

        Expected Output:
            - Best parameters selected from the grid
            - Best cross-validation score as float
            - Total of 4 parameter combinations tested
            - Valid JSON response following API contract
        """
        code = '''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
from pathlib import Path

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=150,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=0
)

# Define parameter grid for hyperparameter tuning
param_grid = {"C": [0.1, 1.0], "penalty": ["l1", "l2"]}
base_classifier = LogisticRegression(
    solver="liblinear",
    max_iter=300,
    random_state=0
)

# Configure cross-validation strategy
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Perform grid search
gs = GridSearchCV(
    base_classifier,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("Starting grid search hyperparameter optimization...")
gs.fit(X, y)

# Extract results
best_params = {
    k: (float(v) if isinstance(v, (int, float)) else v)
    for k, v in gs.best_params_.items()
}

result = {
    "best_params": best_params,
    "best_score": float(gs.best_score_),
    "n_candidates": len(gs.cv_results_["params"])
}

print(f"Grid search completed with {result['n_candidates']} combinations")
print(f"Best parameters: {result['best_params']}")
print(f"Best cross-validation score: {result['best_score']:.4f}")

# Return result for validation
result
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 6  # Grid search takes longer
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        assert data.get("success") is True, f"Expected success=True: {data}"
        assert "data" in data, f"Missing 'data' field: {data}"
        assert data.get("error") is None, f"Unexpected error: {data}"
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate execution results
        result_data = data["data"]
        assert "stdout" in result_data, f"Missing stdout: {result_data}"
        stdout = result_data["stdout"]

        # Validate grid search execution
        assert "Grid search completed with 4" in stdout
        assert "Best parameters:" in stdout
        assert "Best cross-validation score:" in stdout

        # Extract and validate numerical results
        score_lines = [
            line for line in stdout.split('\n')
            if "Best cross-validation score:" in line
        ]
        assert len(score_lines) == 1, f"Expected 1 score line: {score_lines}"

        best_score = float(score_lines[0].split(": ")[1])
        assert 0.0 <= best_score <= 1.0, f"Best score OOR: {best_score}"

        # Validate parameter extraction
        param_lines = [
            line for line in stdout.split('\n')
            if "Best parameters:" in line
        ]
        assert len(param_lines) == 1, f"Expected 1 param line: {param_lines}"

        # Verify that C and penalty are mentioned in the parameters
        param_line = param_lines[0]
        assert "'C'" in param_line, f"C parameter missing: {param_line}"
        assert "'penalty'" in param_line, f"penalty missing: {param_line}"

    def test_given_invalid_parameters_when_grid_search_executed_then_error_is_returned(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test error handling with invalid hyperparameters in GridSearchCV.

        This test validates that the system correctly handles and reports
        errors when invalid hyperparameter names are used in grid search.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Synthetic classification dataset (50 samples, 5 features)
            - Invalid parameter grid with non-existent parameter name
            - LogisticRegression as base estimator

        Expected Output:
            - API response with success=false
            - Error message containing "invalid parameter" or similar
            - No execution timeout or crash
            - Valid JSON response following API contract
        """
        code = '''
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pathlib import Path

print("Testing error handling with invalid hyperparameters...")

# Generate small synthetic dataset for error testing
X, y = make_classification(n_samples=50, n_features=5, random_state=0)

# Define invalid parameter grid (parameter name doesn't exist)
param_grid = {"not_a_param": [1, 10]}  # invalid for LogisticRegression

print(f"Creating grid search with invalid parameter: {list(param_grid.keys())}")  # noqa: E501

# This should raise an error during fit
gs = GridSearchCV(
    LogisticRegression(solver="liblinear"),
    param_grid=param_grid,
    cv=2
)

print("Attempting to fit with invalid parameters...")
gs.fit(X, y)

print("ERROR: This line should not be reached!")
"execution_should_have_failed"
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 2  # Error handling should be fast
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract for error case
        assert data.get("success") is False, f"Expected success=False: {data}"
        assert "error" in data, f"Missing 'error' field: {data}"
        assert data.get("error") is not None, f"Expected non-null error: {data}"  # noqa: E501
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate error message content
        error_message = data.get("error", "").lower()
        assert any(keyword in error_message for keyword in [
            "invalid parameter",
            "unknown parameter",
            "not_a_param",
            "valueerror",
            "execution failed",
            "pythonerror"
        ]), f"Expected parameter error message: {data.get('error')}"

    def test_given_sklearn_availability_when_import_attempted_then_version_information_returned(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test scikit-learn import and version information access.

        This test validates that scikit-learn is properly available in the
        Pyodide environment and can provide version and module information.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Import scikit-learn module
            - Access version information
            - Check basic submodules availability

        Expected Output:
            - Successful import of sklearn module
            - Version string in semantic version format (e.g., "1.2.3")
            - Access to common submodules (datasets, linear_model, etc.)
            - Valid JSON response following API contract
        """
        code = '''
import sklearn as sk
from sklearn import datasets, linear_model, model_selection
from pathlib import Path

print("Testing scikit-learn availability and version information...")

# Check version information
version = sk.__version__
print(f"scikit-learn version: {version}")

# Verify semantic version format (X.Y.Z or X.Y.Z.dev0 etc.)
import re
version_pattern = r'^\\d+\\.\\d+\\.\\d+.*$'
is_valid_version = bool(re.match(version_pattern, version))
print(f"Version format valid: {is_valid_version}")

# Test basic submodule accessibility
submodules_available = []

# Test datasets submodule
try:
    from sklearn.datasets import make_classification
    submodules_available.append("datasets")
    print("✓ datasets submodule accessible")
except ImportError as e:
    print(f"✗ datasets submodule failed: {e}")

# Test linear_model submodule
try:
    from sklearn.linear_model import LogisticRegression
    submodules_available.append("linear_model")
    print("✓ linear_model submodule accessible")
except ImportError as e:
    print(f"✗ linear_model submodule failed: {e}")

# Test model_selection submodule
try:
    from sklearn.model_selection import cross_val_score
    submodules_available.append("model_selection")
    print("✓ model_selection submodule accessible")
except ImportError as e:
    print(f"✗ model_selection submodule failed: {e}")

print(f"Available submodules: {submodules_available}")

result = {
    "version": version,
    "version_valid": is_valid_version,
    "submodules_available": submodules_available,
    "total_submodules": len(submodules_available)
}

print("scikit-learn successfully imported and accessible")
print(f"Summary: {result}")

# Return result for validation
result
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 2
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        assert data.get("success") is True, f"Expected success=True: {data}"
        assert "data" in data, f"Missing 'data' field: {data}"
        assert data.get("error") is None, f"Unexpected error: {data}"
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate execution results
        result_data = data["data"]
        assert "stdout" in result_data, f"Missing stdout: {result_data}"
        stdout = result_data["stdout"]

        # Validate version information
        assert "scikit-learn version:" in stdout
        assert "Version format valid: True" in stdout
        assert "scikit-learn successfully imported" in stdout

        # Validate submodule availability
        assert "✓ datasets submodule accessible" in stdout
        assert "✓ linear_model submodule accessible" in stdout
        assert "✓ model_selection submodule accessible" in stdout

        # Extract and validate version format
        version_lines = [
            line for line in stdout.split('\n')
            if "scikit-learn version:" in line
        ]
        assert len(version_lines) == 1, f"Expected 1 version line: {version_lines}"  # noqa: E501

        version_str = version_lines[0].split(": ")[1]
        # Basic version format check (should be like 1.2.3 or 1.2.3.dev0)
        version_pattern = r'^\d+\.\d+\.\d+.*$'
        assert re.match(version_pattern, version_str), f"Invalid version: {version_str}"  # noqa: E501

    def test_given_classification_algorithms_when_multiple_models_compared_then_all_execute_successfully(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test multiple classification algorithms for broad sklearn compatibility.  # noqa: E501

        This test validates that various classification algorithms from
        scikit-learn work correctly within Pyodide, demonstrating
        comprehensive machine learning library support.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Synthetic multiclass classification dataset
            - Multiple algorithms: LogisticRegression, SVM, Random Forest
            - Training and prediction pipeline for each algorithm

        Expected Output:
            - Successful training and prediction for all algorithms
            - Accuracy scores within reasonable ranges
            - Model comparison results
            - Valid JSON response following API contract
        """
        code = '''
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

print("Testing multiple classification algorithms for compatibility...")

# Generate multiclass classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")  # noqa: E501
print(f"Train/test split: {X_train.shape[0]}/{X_test.shape[0]} samples")

# Define algorithms to test
algorithms = {
    "LogisticRegression": LogisticRegression(
        random_state=42, max_iter=1000
    ),
    "SVC": SVC(random_state=42, probability=True),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=50, random_state=42
    )
}

results = {}
print(f"Testing {len(algorithms)} classification algorithms...")

# Train and evaluate each algorithm
for name, clf in algorithms.items():
    print(f"Training {name}...")

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = float(accuracy)

    print(f"{name} accuracy: {accuracy:.4f}")

print("\\nAlgorithm comparison completed")
print(f"Results summary: {results}")

# Find best performing algorithm
best_algorithm = max(results.items(), key=lambda x: x[1])
print(f"Best algorithm: {best_algorithm[0]} (accuracy: {best_algorithm[1]:.4f})")  # noqa: E501

# Return results for validation
results
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 8  # Multiple algorithms need more time
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        assert data.get("success") is True, f"Expected success=True: {data}"
        assert "data" in data, f"Missing 'data' field: {data}"
        assert data.get("error") is None, f"Unexpected error: {data}"
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate execution results
        result_data = data["data"]
        assert "stdout" in result_data, f"Missing stdout: {result_data}"
        stdout = result_data["stdout"]

        # Validate algorithm execution
        assert "Testing 3 classification algorithms" in stdout
        assert "LogisticRegression accuracy:" in stdout
        assert "SVC accuracy:" in stdout
        assert "RandomForestClassifier accuracy:" in stdout
        assert "Algorithm comparison completed" in stdout
        assert "Best algorithm:" in stdout

        # Extract and validate accuracy scores
        accuracy_lines = [
            line for line in stdout.split('\n')
            if " accuracy: " in line and "Best algorithm:" not in line
        ]
        assert len(accuracy_lines) == 3, f"Expected 3 accuracy lines: {accuracy_lines}"  # noqa: E501

        # Validate that all accuracies are reasonable
        for line in accuracy_lines:
            accuracy_str = line.split(": ")[1]
            accuracy = float(accuracy_str)
            assert 0.0 <= accuracy <= 1.0, f"Accuracy OOR: {accuracy} in {line}"  # noqa: E501
            assert accuracy >= 0.3, f"Accuracy too low: {accuracy} in {line}"

    def test_given_regression_problem_when_multiple_regressors_used_then_predictions_are_accurate(  # noqa: E501
        self, server, base_url, api_timeout
    ):
        """Test regression algorithms for broad sklearn regression support.

        This test validates that regression algorithms from scikit-learn
        work correctly within Pyodide, demonstrating ML versatility.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Synthetic regression dataset with noise
            - Multiple algorithms: Linear, Ridge, Random Forest
            - Training and prediction pipeline for each regressor

        Expected Output:
            - Successful training and prediction for all regressors
            - R² scores indicating model performance
            - Mean squared error metrics
            - Valid JSON response following API contract
        """
        code = '''
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from pathlib import Path

print("Testing multiple regression algorithms for compatibility...")

# Generate regression dataset with noise
X, y = make_regression(
    n_samples=200,
    n_features=15,
    n_informative=10,
    noise=10.0,
    random_state=42
)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Train/test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Define regression algorithms to test
algorithms = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=50, random_state=42
    )
}

results = {}
print(f"Testing {len(algorithms)} regression algorithms...")

# Train and evaluate each algorithm
for name, regressor in algorithms.items():
    print(f"Training {name}...")

    # Train the model
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results[name] = {
        "r2_score": float(r2),
        "mse": float(mse)
    }

    print(f"{name} R² score: {r2:.4f}")
    print(f"{name} MSE: {mse:.4f}")

print("\\nRegression comparison completed")
print(f"Results summary: {results}")

# Find best performing algorithm by R² score
best_algorithm = max(results.items(), key=lambda x: x[1]["r2_score"])
print(f"Best R² score: {best_algorithm[0]} ({best_algorithm[1]['r2_score']:.4f})")  # noqa: E501

# Return results for validation
results
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 6  # Regression algorithms need time
        )

        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        assert data.get("success") is True, f"Expected success=True: {data}"
        assert "data" in data, f"Missing 'data' field: {data}"
        assert data.get("error") is None, f"Unexpected error: {data}"
        assert "meta" in data, f"Missing 'meta' field: {data}"
        assert "timestamp" in data["meta"], f"Missing timestamp: {data}"

        # Validate execution results
        result_data = data["data"]
        assert "stdout" in result_data, f"Missing stdout: {result_data}"
        stdout = result_data["stdout"]

        # Validate algorithm execution
        assert "Testing 3 regression algorithms" in stdout
        assert "LinearRegression R² score:" in stdout
        assert "Ridge R² score:" in stdout
        assert "RandomForestRegressor R² score:" in stdout
        assert "Regression comparison completed" in stdout
        assert "Best R² score:" in stdout

        # Extract and validate R² scores
        r2_lines = [
            line for line in stdout.split('\n')
            if " R² score: " in line and "Best R² score:" not in line
        ]
        assert len(r2_lines) == 3, f"Expected 3 R² score lines: {r2_lines}"

        # Validate that all R² scores are reasonable
        for line in r2_lines:
            r2_str = line.split(": ")[1]
            r2_score = float(r2_str)
            # R² can be negative for very poor models, but should be
            # reasonable for synthetic data
            assert r2_score >= 0.3, f"R² score too low: {r2_score} in {line}"
            assert r2_score <= 1.0, f"R² score too high (max 1.0): {r2_score} in {line}"  # noqa: E501
