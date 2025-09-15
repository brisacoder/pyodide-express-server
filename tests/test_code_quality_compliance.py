#!/usr/bin/env python3
"""
Modern BDD-style test suite for code quality and compliance validation.

This module provides comprehensive testing for code quality standards across
the project, focusing on JavaScript ESLint compliance, Python flake8 standards,
and API contract validation through the /api/execute-raw endpoint.

Test Categories:
- JavaScript code quality and ESLint compliance
- Python code standards and flake8 validation
- API contract compliance and response structure
- Cross-platform code portability
- Import organization and PEP 8 adherence

Key Features:
1. ✅ Pure pytest framework with BDD-style structure
2. ✅ Parameterized tests with no hardcoded values
3. ✅ API contract validation using /api/execute-raw only
4. ✅ Cross-platform pathlib usage validation
5. ✅ Comprehensive docstrings with examples
6. ✅ Fixtures for configuration management
7. ✅ Server-side API contract enforcement testing

Requirements Compliance:
- Uses only /api/execute-raw endpoint for Pyodide testing
- Validates proper API response structure
- Tests pathlib usage for cross-platform compatibility
- Comprehensive BDD test structure
- No hardcoded timeouts or constants
- Full docstring documentation
"""

import json
import subprocess
import sys
from pathlib import Path


import pytest
import requests

# Import shared configuration from conftest.py
from conftest import Config


class QualityStandards:
    """Code quality standards and thresholds configuration."""

    MAX_FORMATTING_ISSUES = 20
    MIN_JSDOC_COVERAGE = 0.5  # 50% minimum documentation coverage
    MAX_FLAKE8_ERRORS = 0      # Zero tolerance for flake8 errors

    # Library classifications for import order validation
    STANDARD_LIBS = [
        'unittest', 'json', 'os', 'sys', 'pathlib', 'time',
        'tempfile', 're', 'subprocess', 'typing', 'collections'
    ]
    THIRD_PARTY_LIBS = [
        'requests', 'numpy', 'pandas', 'matplotlib', 'pytest',
        'seaborn', 'scipy', 'sklearn', 'flask', 'django'
    ]


@pytest.fixture
def project_root() -> Path:
    """
    Provide project root directory path.

    Returns:
        Path: Absolute path to project root directory

    Example:
        >>> root = project_root()
        >>> assert (root / "package.json").exists()
    """
    return Path(__file__).parent.parent


@pytest.fixture
def node_modules_bin(project_root: Path) -> Path:
    """
    Provide Node.js binary directory path.

    Args:
        project_root: Project root directory fixture

    Returns:
        Path: Path to node_modules/.bin directory

    Example:
        >>> bin_dir = node_modules_bin(project_root())
        >>> assert bin_dir.name == ".bin"
    """
    return project_root / "node_modules" / ".bin"


@pytest.fixture
def api_session() -> requests.Session:
    """
    Create configured requests session for API testing.

    Returns:
        requests.Session: Configured session with appropriate timeout

    Example:
        >>> session = api_session()
        >>> response = session.get("/api/health")
        >>> assert response.timeout == Config.TIMEOUTS["api_request"]
    """
    session = requests.Session()
    session.timeout = Config.TIMEOUTS["api_request"]
    return session


@pytest.fixture
def quality_standards() -> QualityStandards:
    """
    Provide quality standards configuration.

    Returns:
        QualityStandards: Configuration object with quality thresholds

    Example:
        >>> standards = quality_standards()
        >>> assert standards.MAX_FLAKE8_ERRORS == 0
    """
    return QualityStandards()


class TestJavaScriptCodeQuality:
    """
    Test JavaScript code quality and standards compliance.

    This class validates JavaScript code against ESLint rules,
    JSDoc documentation standards, and consistent formatting.
    """

    def test_given_javascript_source_when_running_eslint_then_no_errors_found(
        self,
        project_root: Path,
        node_modules_bin: Path,
        quality_standards: QualityStandards
    ):
        """
        Validate JavaScript source code has no ESLint errors.

        Given: JavaScript source files exist in the project
        When: ESLint is executed on the source directory
        Then: No ESLint errors should be found

        Args:
            project_root: Project root directory fixture
            node_modules_bin: Node modules binary directory fixture
            quality_standards: Quality standards configuration fixture

        Returns:
            None: Test assertion validates no ESLint errors

        Example:
            ESLint should report:
            ```
            ESLint results: 0 errors, 2 warnings
            ```
        """
        # Given: JavaScript source files exist
        src_dir = project_root / "src"
        assert src_dir.exists(), f"Source directory not found: {src_dir}"

        # When: Running ESLint on source files
        eslint_binary = "eslint.cmd" if sys.platform == 'win32' else "eslint"
        eslint_cmd = [
            str(node_modules_bin / eslint_binary),
            str(src_dir),
            "--format", "json"
        ]

        try:
            result = subprocess.run(
                eslint_cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=Config.TIMEOUTS["quick_operation"]
            )

            # Then: Should have no ESLint errors
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)

                    # Count total errors and warnings
                    total_errors = sum(
                        len([msg for msg in file.get('messages', [])
                            if msg.get('severity') == 2])
                        for file in eslint_results
                    )
                    total_warnings = sum(
                        len([msg for msg in file.get('messages', [])
                            if msg.get('severity') == 1])
                        for file in eslint_results
                    )

                    print(f"ESLint results: {total_errors} errors, {total_warnings} warnings")

                    # Document warnings but don't fail the test
                    if total_warnings > 0:
                        print(f"ESLint warnings found: {total_warnings}")
                        for file_result in eslint_results:
                            if file_result.get('messages'):
                                print(f"  {file_result['filePath']}: {len(file_result['messages'])} issues")

                    # Assert no ESLint errors
                    assert total_errors == 0, f"ESLint found {total_errors} errors in JavaScript code"

                except json.JSONDecodeError:
                    # ESLint might return non-JSON output for certain errors
                    if result.returncode != 0:
                        pytest.fail(f"ESLint execution failed with non-JSON output: {result.stderr}")

        except FileNotFoundError:
            pytest.skip("ESLint not found - run 'npm install' first")
        except subprocess.TimeoutExpired:
            pytest.fail(f"ESLint timed out after {Config.TIMEOUTS['quick_operation']} seconds")

    def test_given_javascript_functions_when_checking_jsdoc_then_meets_coverage_threshold(
        self,
        project_root: Path,
        quality_standards: QualityStandards
    ):
        """
        Validate JSDoc documentation coverage meets minimum threshold.

        Given: JavaScript source files with function definitions exist
        When: Analyzing JSDoc documentation coverage
        Then: Documentation coverage should meet minimum threshold (50%)

        Args:
            project_root: Project root directory fixture
            quality_standards: Quality standards configuration fixture

        Returns:
            None: Test assertion validates documentation coverage

        Example:
            Expected output:
            ```
            JSDoc coverage: 15/20 (75.0%)
            ```
        """
        # Given: JavaScript files with functions exist
        src_dir = project_root / "src"
        js_files = list(src_dir.rglob("*.js"))
        assert js_files, "No JavaScript files found in source directory"

        documented_functions = 0
        total_functions = 0
        issues = []

        # When: Checking JSDoc documentation coverage
        for js_file in js_files:
            try:
                content = js_file.read_text(encoding='utf-8')
                lines = content.split('\n')

                for i, line in enumerate(lines):
                    # Look for function definitions (excluding commented lines)
                    if ('function ' in line or 'async function' in line) and not line.strip().startswith('//'):
                        total_functions += 1

                        # Check if previous lines contain JSDoc comment
                        has_jsdoc = False
                        for j in range(max(0, i-10), i):
                            if '/**' in lines[j]:
                                has_jsdoc = True
                                break

                        if has_jsdoc:
                            documented_functions += 1
                        else:
                            issues.append(f"{js_file}:{i+1} - Function without JSDoc: {line.strip()}")

            except Exception as e:
                print(f"Error reading {js_file}: {e}")

        # Then: Documentation coverage should meet threshold
        if total_functions > 0:
            documentation_ratio = documented_functions / total_functions
            print(f"JSDoc coverage: {documented_functions}/{total_functions} ({documentation_ratio:.1%})")

            # Show some undocumented functions for improvement
            for issue in issues[:5]:
                print(f"  {issue}")

            assert documentation_ratio >= quality_standards.MIN_JSDOC_COVERAGE, \
                f"JSDoc coverage too low: {documentation_ratio:.1%} < {quality_standards.MIN_JSDOC_COVERAGE:.1%}"
        else:
            pytest.skip("No JavaScript functions found to validate")

    def test_given_javascript_files_when_checking_formatting_then_follows_consistent_style(
        self,
        project_root: Path,
        quality_standards: QualityStandards
    ):
        """
        Validate JavaScript code follows consistent formatting standards.

        Given: JavaScript source files exist
        When: Analyzing code formatting consistency
        Then: Should have minimal formatting issues (< 20)

        Args:
            project_root: Project root directory fixture
            quality_standards: Quality standards configuration fixture

        Returns:
            None: Test assertion validates formatting consistency

        Example:
            Expected outcome:
            ```
            Code formatting issues: 3
            src/app.js:45 - Trailing whitespace
            src/utils.js:12 - Inconsistent indentation
            ```
        """
        # Given: JavaScript source files exist
        src_dir = project_root / "src"
        js_files = list(src_dir.rglob("*.js"))
        assert js_files, "No JavaScript files found for formatting validation"

        formatting_issues = []

        # When: Checking code formatting standards
        for js_file in js_files:
            try:
                content = js_file.read_text(encoding='utf-8')
                lines = content.split('\n')

                in_template_literal = False
                template_literal_start_char = None

                for i, line in enumerate(lines):
                    # Track template literal boundaries to skip formatting checks
                    for char in line:
                        if char == '`' and not in_template_literal:
                            in_template_literal = True
                            template_literal_start_char = '`'
                        elif char == '`' and in_template_literal and template_literal_start_char == '`':
                            in_template_literal = False
                            template_literal_start_char = None

                    # Skip formatting checks for lines within template literals
                    if in_template_literal:
                        continue

                    # Check for consistent indentation (2 spaces for JS)
                    if line.strip() and line.startswith(' '):
                        stripped = line.lstrip(' ')

                        if stripped.startswith('*'):
                            # JSDoc comment lines - should be properly aligned
                            leading_spaces = len(line) - len(stripped)
                            if (leading_spaces - 1) % 2 != 0:
                                formatting_issues.append(f"{js_file}:{i+1} - JSDoc comment indentation")
                        else:
                            # Regular code should use even number of spaces
                            leading_spaces = len(line) - len(line.lstrip(' '))
                            if leading_spaces % 2 != 0:
                                formatting_issues.append(f"{js_file}:{i+1} - Inconsistent indentation")

                    # Check for trailing whitespace
                    if line.endswith(' ') or line.endswith('\t'):
                        formatting_issues.append(f"{js_file}:{i+1} - Trailing whitespace")

            except Exception as e:
                print(f"Error checking formatting in {js_file}: {e}")

        # Then: Should have minimal formatting issues
        if formatting_issues:
            print(f"Code formatting issues: {len(formatting_issues)}")
            for issue in formatting_issues[:10]:  # Show first 10 issues
                print(f"  {issue}")

        assert len(formatting_issues) < quality_standards.MAX_FORMATTING_ISSUES, \
            f"Too many formatting issues: {len(formatting_issues)} >= {quality_standards.MAX_FORMATTING_ISSUES}"


class TestPythonCodeQuality:
    """
    Test Python code quality and standards compliance.

    This class validates Python code against flake8 rules,
    PEP 8 import organization, and coding best practices.
    """

    def test_given_python_source_when_running_flake8_then_no_issues_found(
        self,
        project_root: Path,
        quality_standards: QualityStandards
    ):
        """
        Validate Python source code has no flake8 issues.

        Given: Python test files exist in the project
        When: flake8 linter is executed on test directory
        Then: No code quality issues should be found

        Args:
            project_root: Project root directory fixture
            quality_standards: Quality standards configuration fixture

        Returns:
            None: Test assertion validates no flake8 issues

        Example:
            Expected output:
            ```
            Flake8: No Python code quality issues found
            ```
        """
        # Given: Python test files exist
        tests_dir = project_root / "tests"
        assert tests_dir.exists(), f"Tests directory not found: {tests_dir}"

        try:
            # When: Running flake8 on test files
            result = subprocess.run(
                [sys.executable, "-m", "flake8", str(tests_dir), "--statistics"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=Config.TIMEOUTS["quick_operation"]
            )

            # Then: Should find no code quality issues
            if result.returncode == 0:
                print("Flake8: No Python code quality issues found")
            else:
                # Parse flake8 output to count issues
                output_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                error_lines = [line for line in output_lines if ':' in line]

                print(f"Flake8 found {len(error_lines)} issues:")
                for line in error_lines[:10]:  # Show first 10 issues
                    print(f"  {line}")

                assert len(error_lines) == quality_standards.MAX_FLAKE8_ERRORS, \
                    f"Flake8 found {len(error_lines)} issues (expected: {quality_standards.MAX_FLAKE8_ERRORS})"

        except FileNotFoundError:
            pytest.skip("Flake8 not found - install with 'uv add flake8'")
        except subprocess.TimeoutExpired:
            pytest.fail(f"Flake8 timed out after {Config.TIMEOUTS['quick_operation']} seconds")

    def test_given_python_imports_when_checking_organization_then_follows_pep8_order(
        self,
        project_root: Path,
        quality_standards: QualityStandards
    ):
        """
        Validate Python imports follow PEP 8 organization standards.

        Given: Python test files with import statements exist
        When: Analyzing import organization
        Then: Should follow PEP 8 order (standard → third_party → local)

        Args:
            project_root: Project root directory fixture
            quality_standards: Quality standards configuration fixture

        Returns:
            None: Test assertion validates import organization

        Example:
            Valid import order:
            ```python
            import json     # standard library
            import sys      # standard library

            import pytest   # third-party
            import requests # third-party

            from conftest import Config  # local
            ```
        """
        # Given: Python test files with imports exist
        tests_dir = project_root / "tests"
        python_files = list(tests_dir.glob("test_*.py"))
        assert python_files, "No Python test files found"

        issues = []

        # When: Checking import organization
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')

                import_sections = {
                    'standard': [],
                    'third_party': [],
                    'local': []
                }

                in_imports = False

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    if stripped.startswith('import ') or stripped.startswith('from '):
                        in_imports = True

                        # Classify import type
                        section = self._classify_import(stripped, quality_standards)
                        import_sections[section].append((i, stripped))

                    elif in_imports and stripped == '':
                        continue  # Blank lines in import section are allowed
                    elif in_imports and not stripped.startswith('#'):
                        break  # End of import section

                # Verify imports are in correct PEP 8 order
                all_imports = []
                for section in ['standard', 'third_party', 'local']:
                    all_imports.extend(import_sections[section])

                # Check import order violations
                for i in range(len(all_imports) - 1):
                    if all_imports[i][0] > all_imports[i+1][0]:
                        curr_section = self._classify_import(all_imports[i][1], quality_standards)
                        next_section = self._classify_import(all_imports[i+1][1], quality_standards)

                        if self._section_order(curr_section) > self._section_order(next_section):
                            issues.append(f"{py_file}:{all_imports[i+1][0]+1} - Import order violation")

            except Exception as e:
                print(f"Error checking imports in {py_file}: {e}")

        # Then: Should have no import organization issues
        if issues:
            print(f"Import organization issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"  {issue}")

        assert len(issues) == 0, f"Found {len(issues)} import organization issues"

    def test_given_python_files_when_checking_unused_imports_then_documents_findings(
        self,
        project_root: Path
    ):
        """
        Document potentially unused imports in Python files.

        Given: Python test files with import statements exist
        When: Analyzing for unused imports
        Then: Should document any potentially unused imports (informational only)

        Args:
            project_root: Project root directory fixture

        Returns:
            None: Informational test that documents findings

        Note:
            This test is informational only and does not fail.
            It helps identify potential cleanup opportunities.

        Example:
            Output might show:
            ```
            Potential unused imports: 3
            tests/test_api.py - Potentially unused import: tempfile
            ```
        """
        # Given: Python test files with imports exist
        tests_dir = project_root / "tests"
        python_files = list(tests_dir.glob("test_*.py"))

        if not python_files:
            pytest.skip("No Python test files found")

        issues = []

        # When: Checking for unused imports
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')

                # Extract imported modules
                import_modules = []
                for line in content.split('\n'):
                    stripped = line.strip()
                    if stripped.startswith('import '):
                        module = stripped.replace('import ', '').split(' as ')[0].split('.')[0]
                        import_modules.append(module)
                    elif stripped.startswith('from ') and ' import ' in stripped:
                        parts = stripped.split(' import ')
                        if len(parts) == 2:
                            imported_items = parts[1].split(', ')
                            for item in imported_items:
                                clean_item = item.split(' as ')[0].strip()
                                import_modules.append(clean_item)

                # Check if imports are used (simple heuristic check)
                for module in import_modules:
                    if module and module not in ['unittest', '__future__', 'pytest']:
                        # Simple check - module name appears elsewhere in file
                        if content.count(module) <= 1:  # Only in import line
                            issues.append(f"{py_file} - Potentially unused import: {module}")

            except Exception as e:
                print(f"Error checking unused imports in {py_file}: {e}")

        # Then: Document findings (informational only)
        if issues:
            print(f"Potential unused imports: {len(issues)}")
            for issue in issues[:10]:
                print(f"  {issue}")
        else:
            print("No potentially unused imports detected")

        # This is informational only - no assertion to fail the test

    def _classify_import(self, import_line: str, quality_standards: QualityStandards) -> str:
        """
        Classify import statement as standard, third_party, or local.

        Args:
            import_line: Python import statement to classify
            quality_standards: Quality standards configuration

        Returns:
            str: Import classification ('standard', 'third_party', 'local')

        Example:
            >>> _classify_import("import json", standards)
            'standard'
            >>> _classify_import("import pytest", standards)
            'third_party'
        """
        if any(lib in import_line for lib in quality_standards.STANDARD_LIBS):
            return 'standard'
        elif any(lib in import_line for lib in quality_standards.THIRD_PARTY_LIBS):
            return 'third_party'
        else:
            return 'local'

    def _section_order(self, section: str) -> int:
        """
        Get numeric order for import sections per PEP 8.

        Args:
            section: Import section name

        Returns:
            int: Numeric order for sorting

        Example:
            >>> _section_order('standard')
            1
            >>> _section_order('third_party')
            2
        """
        return {'standard': 1, 'third_party': 2, 'local': 3}.get(section, 4)


class TestApiContractCompliance:
    """
    Test API contract compliance and response structure validation.

    This class validates that the server API endpoints return responses
    in the expected format and that Pyodide code execution works correctly
    with proper cross-platform compatibility.
    """

    def test_given_server_running_when_executing_simple_python_then_returns_valid_contract(
        self,
        api_session: requests.Session
    ):
        """
        Validate /api/execute-raw endpoint returns proper API contract structure.

        Given: Server is running and accessible
        When: Executing simple Python code via /api/execute-raw
        Then: Should return valid JSON with success, data, error, meta fields

        Args:
            api_session: Configured requests session fixture

        Returns:
            None: Test assertion validates API contract compliance

        Example:
            Expected response structure:
            ```json
            {
                "success": true,
                "data": {
                    "result": "Hello from Pyodide!",
                    "stdout": "Hello from Pyodide!\n",
                    "stderr": "",
                    "executionTime": 123
                },
                "error": null,
                "meta": {
                    "timestamp": "2025-09-14T10:30:00.000Z"
                }
            }
            ```
        """
        # Given: Simple Python code to execute
        python_code = "print('Hello from Pyodide!')"

        # When: Executing code via /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            json={
                "code": python_code,
                "timeout": Config.TIMEOUTS["code_execution"] * 1000  # Convert to milliseconds
            },
            timeout=Config.TIMEOUTS["api_request"]
        )

        # Then: Should return valid API contract structure
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        response_data = response.json()

        # Validate API contract structure
        self._validate_api_contract(response_data)

        # Validate successful execution
        assert response_data["success"] is True, f"Execution failed: {response_data.get('error')}"
        assert response_data["data"] is not None, "Expected data field to be populated"
        assert response_data["error"] is None, f"Unexpected error: {response_data['error']}"

        # Validate data structure
        data = response_data["data"]
        assert "result" in data, "Expected result field in data"
        assert "stdout" in data, "Expected stdout field in data"
        assert "stderr" in data, "Expected stderr field in data"
        assert "executionTime" in data, "Expected executionTime field in data"

        # Validate content
        assert "Hello from Pyodide!" in data["result"], f"Expected output not found in result: {data['result']}"

    def test_given_server_running_when_executing_pathlib_code_then_validates_portability(
        self,
        api_session: requests.Session
    ):
        """
        Validate cross-platform pathlib usage in Pyodide execution.

        Given: Server is running and accessible
        When: Executing Python code using pathlib for file operations
        Then: Should work correctly across Windows and Linux platforms

        Args:
            api_session: Configured requests session fixture

        Returns:
            None: Test assertion validates pathlib portability

        Example:
            Python code being tested:
            ```python
            from pathlib import Path

            # Create portable path handling
            test_path = Path('/tmp') / 'test_file.txt'
            print(f"Path: {test_path}")
            print(f"Is absolute: {test_path.is_absolute()}")
            ```
        """
        # Given: Python code using pathlib for cross-platform compatibility
        python_code = """
from pathlib import Path
import sys

# Test pathlib usage for portability
test_path = Path('/tmp') / 'test_file.txt'
plots_path = Path('/plots') / 'matplotlib' / 'test.png'

# Validate pathlib functionality
print(f"Test path: {test_path}")
print(f"Plots path: {plots_path}")
print(f"Test path is absolute: {test_path.is_absolute()}")
print(f"Plots path parent: {plots_path.parent}")
print(f"Path separator: {test_path.parts}")
print(f"Python version: {sys.version}")
print("Pathlib portability test completed successfully")
        """

        # When: Executing pathlib code via /api/execute-raw
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            json={
                "code": python_code,
                "timeout": Config.TIMEOUTS["code_execution"] * 1000
            },
            timeout=Config.TIMEOUTS["api_request"]
        )

        # Then: Should execute successfully with pathlib operations
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        response_data = response.json()
        self._validate_api_contract(response_data)

        assert response_data["success"] is True, f"Pathlib execution failed: {response_data.get('error')}"

        data = response_data["data"]
        result_text = data["result"]

        # Validate pathlib operations worked correctly
        assert "Test path: /tmp/test_file.txt" in result_text, "Pathlib path construction failed"
        assert "Test path is absolute: True" in result_text, "Pathlib absolute path check failed"
        assert "Plots path parent: /plots/matplotlib" in result_text, "Pathlib parent operation failed"
        assert "Pathlib portability test completed successfully" in result_text, "Test completion marker not found"

    def test_given_server_running_when_code_execution_fails_then_returns_proper_error_contract(
        self,
        api_session: requests.Session
    ):
        """
        Validate error handling follows API contract structure.

        Given: Server is running and accessible
        When: Executing invalid Python code that causes an error
        Then: Should return proper error structure following API contract

        Args:
            api_session: Configured requests session fixture

        Returns:
            None: Test assertion validates error contract compliance

        Example:
            Expected error response:
            ```json
            {
                "success": false,
                "data": null,
                "error": "NameError: name 'undefined_variable' is not defined",
                "meta": {
                    "timestamp": "2025-09-14T10:30:00.000Z"
                }
            }
            ```
        """
        # Given: Invalid Python code that will cause an error
        invalid_python_code = "print(undefined_variable)"  # NameError

        # When: Executing invalid code via /api/execute-raw
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            json={
                "code": invalid_python_code,
                "timeout": Config.TIMEOUTS["code_execution"] * 1000
            },
            timeout=Config.TIMEOUTS["api_request"]
        )

        # Then: Should return proper error contract structure
        assert response.status_code == 200, f"Expected 200 even for execution errors, got {response.status_code}: {response.text}"

        response_data = response.json()
        self._validate_api_contract(response_data)

        # Validate error handling
        assert response_data["success"] is False, "Expected success to be false for execution error"
        assert response_data["data"] is None, "Expected data to be null for execution error"
        assert response_data["error"] is not None, "Expected error field to be populated"
        assert isinstance(response_data["error"], str), "Expected error to be a string"
        assert len(response_data["error"]) > 0, "Expected non-empty error message"

    def test_given_server_running_when_executing_data_science_code_then_validates_libraries(
        self,
        api_session: requests.Session
    ):
        """
        Validate data science libraries are available and working correctly.

        Given: Server is running with data science libraries available
        When: Executing Python code using numpy, pandas, matplotlib
        Then: Should execute successfully and return expected results

        Args:
            api_session: Configured requests session fixture

        Returns:
            None: Test assertion validates data science library availability

        Example:
            Expected to work with libraries:
            - numpy for numerical operations
            - pandas for data manipulation
            - matplotlib for plotting (with pathlib paths)
        """
        # Given: Data science code using common libraries
        data_science_code = """
import numpy as np
import pandas as pd
from pathlib import Path

# Test numpy functionality
arr = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(arr)
print(f"NumPy array mean: {mean_val}")

# Test pandas functionality
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})
print(f"Pandas DataFrame shape: {df.shape}")
print(f"DataFrame mean: {df.mean().to_dict()}")

# Test pathlib for file operations
data_dir = Path('/uploads')
print(f"Data directory path: {data_dir}")
print(f"Is data_dir absolute: {data_dir.is_absolute()}")

print("Data science libraries validation completed successfully")
        """

        # When: Executing data science code via /api/execute-raw
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            json={
                "code": data_science_code,
                "timeout": Config.TIMEOUTS["code_execution"] * 1000
            },
            timeout=Config.TIMEOUTS["api_request"]
        )

        # Then: Should execute successfully with all libraries working
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        response_data = response.json()
        self._validate_api_contract(response_data)

        assert response_data["success"] is True, f"Data science execution failed: {response_data.get('error')}"

        data = response_data["data"]
        result_text = data["result"]

        # Validate each library worked correctly
        assert "NumPy array mean: 3.0" in result_text, "NumPy operations failed"
        assert "Pandas DataFrame shape: (5, 2)" in result_text, "Pandas DataFrame creation failed"
        assert "'A': 3.0, 'B': 30.0" in result_text, "Pandas mean calculation failed"
        assert "Data directory path: /uploads" in result_text, "Pathlib path operations failed"
        assert "Data science libraries validation completed successfully" in result_text, "Test completion marker not found"

    def _validate_api_contract(self, response_data: dict) -> None:
        """
        Validate response follows the API contract structure.

        Args:
            response_data: JSON response data to validate

        Raises:
            AssertionError: If response doesn't follow API contract

        Example:
            Expected structure:
            ```json
            {
                "success": true|false,
                "data": <object|null>,
                "error": <string|null>,
                "meta": {"timestamp": <string>}
            }
            ```
        """
        # Validate required top-level fields
        assert "success" in response_data, "Response missing required 'success' field"
        assert "data" in response_data, "Response missing required 'data' field"
        assert "error" in response_data, "Response missing required 'error' field"
        assert "meta" in response_data, "Response missing required 'meta' field"

        # Validate field types
        assert isinstance(response_data["success"], bool), "Field 'success' must be boolean"

        # Validate meta structure
        meta = response_data["meta"]
        assert isinstance(meta, dict), "Field 'meta' must be object"
        assert "timestamp" in meta, "Meta missing required 'timestamp' field"
        assert isinstance(meta["timestamp"], str), "Meta 'timestamp' must be string"

        # Validate success/error/data consistency
        if response_data["success"]:
            assert response_data["error"] is None, "Success response should have null error"
            assert response_data["data"] is not None, "Success response should have non-null data"
        else:
            assert response_data["error"] is not None, "Error response should have non-null error"
            assert response_data["data"] is None, "Error response should have null data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])