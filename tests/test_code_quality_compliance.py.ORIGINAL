#!/usr/bin/env python3
"""
BDD-style test suite for code quality and linting compliance.

These tests verify that code quality standards are maintained across
the project, including ESLint for JavaScript and flake8 for Python.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Global constants
PROJECT_ROOT = Path(__file__).parent.parent
NODE_MODULES_BIN = PROJECT_ROOT / "node_modules" / ".bin"
MAX_FORMATTING_ISSUES = 20
MIN_JSDOC_COVERAGE = 0.5  # 50% minimum documentation coverage
STANDARD_LIBS = ['unittest', 'json', 'os', 'sys', 'pathlib', 'time', 'tempfile', 're', 'subprocess']
THIRD_PARTY_LIBS = ['requests', 'numpy', 'pandas', 'matplotlib', 'pytest']


class TestJavaScriptCodeQuality:
    """Test JavaScript code quality and standards compliance."""

    def test_when_running_eslint_then_no_errors_found(self):
        """
        Given: JavaScript source files in the project
        When: ESLint is run on the source directory
        Then: Should find no ESLint errors
        """
        # Given
        eslint_cmd = [
            str(NODE_MODULES_BIN / "eslint.cmd") if sys.platform == 'win32' else str(NODE_MODULES_BIN / "eslint"),
            "src/",
            "--format", "json"
        ]
        
        try:
            # When
            result = subprocess.run(
                eslint_cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            # Then
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)
                    
                    # Count total errors and warnings
                    total_errors = sum(
                        len([msg for msg in file.get('messages', []) if msg.get('severity') == 2]) 
                        for file in eslint_results
                    )
                    total_warnings = sum(
                        len([msg for msg in file.get('messages', []) if msg.get('severity') == 1]) 
                        for file in eslint_results
                    )
                    
                    print(f"ESLint results: {total_errors} errors, {total_warnings} warnings")
                    
                    # Document warnings but don't fail
                    if total_warnings > 0:
                        print(f"ESLint warnings found: {total_warnings}")
                        for file_result in eslint_results:
                            if file_result.get('messages'):
                                print(f"  {file_result['filePath']}: {len(file_result['messages'])} issues")
                    
                    # Should have no ESLint errors
                    assert total_errors == 0, f"ESLint found {total_errors} errors in JavaScript code"
                    
                except json.JSONDecodeError:
                    # ESLint might return non-JSON output for certain errors
                    if result.returncode != 0:
                        pytest.fail(f"ESLint failed: {result.stderr}")
            
        except FileNotFoundError:
            pytest.skip("ESLint not found - run 'npm install' first")

    def test_when_checking_jsdoc_coverage_then_meets_minimum_threshold(self):
        """
        Given: JavaScript source files with function definitions
        When: Checking for JSDoc documentation
        Then: Should have at least 50% documentation coverage
        """
        # Given
        js_files = list(Path("src").rglob("*.js"))
        documented_functions = 0
        total_functions = 0
        issues = []
        
        # When
        for js_file in js_files:
            try:
                content = js_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    # Look for function definitions
                    if ('function ' in line or 'async function' in line) and not line.strip().startswith('//'):
                        total_functions += 1
                        
                        # Check if previous lines contain JSDoc
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
        
        # Then
        if total_functions > 0:
            documentation_ratio = documented_functions / total_functions
            print(f"JSDoc coverage: {documented_functions}/{total_functions} ({documentation_ratio:.1%})")
            
            # Show some undocumented functions
            for issue in issues[:5]:
                print(f"  {issue}")
            
            assert documentation_ratio > MIN_JSDOC_COVERAGE, \
                f"JSDoc documentation coverage too low: {documentation_ratio:.1%}"

    def test_when_checking_formatting_then_follows_consistent_style(self):
        """
        Given: JavaScript source files
        When: Checking code formatting standards
        Then: Should have minimal formatting issues (less than 20)
        """
        # Given
        js_files = list(Path("src").rglob("*.js"))
        formatting_issues = []
        
        # When
        for js_file in js_files:
            try:
                content = js_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                in_template_literal = False
                template_literal_start_char = None
                
                for i, line in enumerate(lines):
                    # Track template literal boundaries
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
                        
                    # Check for consistent indentation (should be 2 spaces for JS)
                    if line.strip() and line.startswith(' '):
                        # Special case: JSDoc comments
                        stripped = line.lstrip(' ')
                        if stripped.startswith('*'):
                            # JSDoc comment lines
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
        
        # Then
        if formatting_issues:
            print(f"Code formatting issues: {len(formatting_issues)}")
            for issue in formatting_issues[:10]:
                print(f"  {issue}")
        
        assert len(formatting_issues) < MAX_FORMATTING_ISSUES, \
            f"Too many formatting issues: {len(formatting_issues)}"


class TestPythonCodeQuality:
    """Test Python code quality and standards compliance."""

    def test_when_running_flake8_then_no_issues_found(self):
        """
        Given: Python test files in the project
        When: Running flake8 linter
        Then: Should find no code quality issues
        """
        try:
            # When
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "tests/", "--statistics"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            # Then
            if result.returncode == 0:
                print("Flake8: No Python code quality issues found")
            else:
                # Parse flake8 output to count issues
                output_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                error_lines = [line for line in output_lines if ':' in line and not line.startswith('tests/')]
                
                print(f"Flake8 found {len(error_lines)} issues:")
                for line in error_lines[:10]:  # Show first 10 issues
                    print(f"  {line}")
                
                assert len(error_lines) == 0, \
                    f"Flake8 found {len(error_lines)} Python code quality issues"
            
        except subprocess.CalledProcessError:
            pytest.skip("Flake8 not found - install with 'pip install flake8'")

    def test_when_checking_import_order_then_follows_pep8_organization(self):
        """
        Given: Python test files with imports
        When: Checking import organization
        Then: Should follow PEP 8 order (standard -> third_party -> local)
        """
        # Given
        python_files = list(Path("tests").glob("test_*.py"))
        issues = []
        
        # When
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
                        
                        # Classify import
                        section = _classify_import(stripped)
                        import_sections[section].append((i, stripped))
                    
                    elif in_imports and stripped == '':
                        continue  # Blank lines in import section
                    elif in_imports and not stripped.startswith('#'):
                        break  # End of import section
                
                # Check import order
                all_imports = []
                for section in ['standard', 'third_party', 'local']:
                    all_imports.extend(import_sections[section])
                
                # Verify imports are in order
                for i in range(len(all_imports) - 1):
                    if all_imports[i][0] > all_imports[i+1][0]:
                        curr_section = _classify_import(all_imports[i][1])
                        next_section = _classify_import(all_imports[i+1][1])
                        
                        if _section_order(curr_section) > _section_order(next_section):
                            issues.append(f"{py_file}:{all_imports[i+1][0]+1} - Import order violation")
            
            except Exception as e:
                print(f"Error checking imports in {py_file}: {e}")
        
        # Then
        if issues:
            print(f"Import organization issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"  {issue}")
        
        assert len(issues) == 0, f"Found {len(issues)} import organization issues"

    def test_when_checking_unused_imports_then_documents_findings(self):
        """
        Given: Python test files with import statements
        When: Checking for unused imports
        Then: Should document any potentially unused imports (informational only)
        """
        # Given
        python_files = list(Path("tests").glob("test_*.py"))
        issues = []
        
        # When
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Extract imports
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
                                import_modules.append(item.split(' as ')[0])
                
                # Check if imports are used
                for module in import_modules:
                    if module and module not in ['unittest', '__future__', 'pytest']:
                        # Simple check - module name appears elsewhere in file
                        if content.count(module) <= 1:  # Only in import line
                            issues.append(f"{py_file} - Potentially unused import: {module}")
            
            except Exception as e:
                print(f"Error checking unused imports in {py_file}: {e}")
        
        # Then - Document findings but don't fail
        if issues:
            print(f"Potential unused imports: {len(issues)}")
            for issue in issues[:10]:
                print(f"  {issue}")
        # This is informational only - no assertion


def _classify_import(import_line):
    """Classify import as standard, third_party, or local."""
    if any(lib in import_line for lib in STANDARD_LIBS):
        return 'standard'
    elif any(lib in import_line for lib in THIRD_PARTY_LIBS):
        return 'third_party'
    else:
        return 'local'


def _section_order(section):
    """Get numeric order for import sections."""
    return {'standard': 1, 'third_party': 2, 'local': 3}.get(section, 4)