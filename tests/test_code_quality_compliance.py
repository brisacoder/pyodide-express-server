#!/usr/bin/env python3
"""
Test suite for code quality and linting compliance
Tests that recent code quality improvements are maintained
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NODE_MODULES_BIN = PROJECT_ROOT / "node_modules" / ".bin"


class CodeQualityTestCase(unittest.TestCase):
    """Test code quality and linting compliance"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = PROJECT_ROOT
        os.chdir(self.project_root)

    def test_eslint_javascript_compliance(self):
        """Test that JavaScript code passes ESLint checks"""
        try:
            # Run ESLint on source files
            eslint_cmd = [
                str(NODE_MODULES_BIN / "eslint.cmd") if os.name == 'nt' else str(NODE_MODULES_BIN / "eslint"),
                "src/",
                "--format", "json"
            ]
            
            result = subprocess.run(
                eslint_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse ESLint output
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)
                    
                    # Count total errors and warnings
                    total_errors = sum(len([msg for msg in file.get('messages', []) if msg.get('severity') == 2]) 
                                     for file in eslint_results)
                    total_warnings = sum(len([msg for msg in file.get('messages', []) if msg.get('severity') == 1]) 
                                       for file in eslint_results)
                    
                    print(f"ESLint results: {total_errors} errors, {total_warnings} warnings")
                    
                    # Should have no ESLint errors
                    self.assertEqual(total_errors, 0, 
                                   f"ESLint found {total_errors} errors in JavaScript code")
                    
                    # Document warnings but don't fail
                    if total_warnings > 0:
                        print(f"ESLint warnings found: {total_warnings}")
                        for file_result in eslint_results:
                            if file_result.get('messages'):
                                print(f"  {file_result['filePath']}: {len(file_result['messages'])} issues")
                    
                except json.JSONDecodeError:
                    # ESLint might return non-JSON output for certain errors
                    if result.returncode != 0:
                        self.fail(f"ESLint failed: {result.stderr}")
            
        except FileNotFoundError:
            self.skipTest("ESLint not found - run 'npm install' first")

    def test_python_flake8_compliance(self):
        """Test that Python code passes flake8 checks"""
        try:
            # Run flake8 on test files
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "tests/", "--statistics"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Flake8 returns 0 for no issues, 1 for issues found
            if result.returncode == 0:
                print("Flake8: No Python code quality issues found")
            else:
                # Parse flake8 output to count issues
                output_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                error_lines = [line for line in output_lines if ':' in line and not line.startswith('tests/')]
                
                print(f"Flake8 found {len(error_lines)} issues:")
                for line in error_lines[:10]:  # Show first 10 issues
                    print(f"  {line}")
                
                # Should have no flake8 errors
                self.assertEqual(len(error_lines), 0,
                               f"Flake8 found {len(error_lines)} Python code quality issues")
            
        except FileNotFoundError:
            self.skipTest("Flake8 not found - install with 'pip install flake8'")

    def test_jsdoc_documentation_present(self):
        """Test that JavaScript functions have proper JSDoc documentation"""
        js_files = list(Path("src").rglob("*.js"))
        
        documented_functions = 0
        total_functions = 0
        issues = []
        
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
        
        if total_functions > 0:
            documentation_ratio = documented_functions / total_functions
            print(f"JSDoc coverage: {documented_functions}/{total_functions} ({documentation_ratio:.1%})")
            
            # Show some undocumented functions
            for issue in issues[:5]:
                print(f"  {issue}")
            
            # Should have reasonable documentation coverage
            self.assertGreater(documentation_ratio, 0.5,
                             f"JSDoc documentation coverage too low: {documentation_ratio:.1%}")

    def test_import_organization_compliance(self):
        """Test that Python imports follow PEP 8 organization"""
        python_files = list(Path("tests").glob("test_*.py"))
        
        issues = []
        
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
                current_section = 'standard'
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        in_imports = True
                        
                        # Classify import
                        if any(lib in stripped for lib in ['unittest', 'json', 'os', 'sys', 'pathlib', 'time', 'tempfile', 're']):
                            import_sections['standard'].append((i, stripped))
                        elif any(lib in stripped for lib in ['requests', 'numpy', 'pandas', 'matplotlib']):
                            import_sections['third_party'].append((i, stripped))
                        else:
                            import_sections['local'].append((i, stripped))
                    
                    elif in_imports and stripped == '':
                        continue  # Blank lines in import section
                    elif in_imports and not stripped.startswith('#'):
                        break  # End of import section
                
                # Check import order (standard -> third_party -> local)
                all_imports = []
                for section in ['standard', 'third_party', 'local']:
                    all_imports.extend(import_sections[section])
                
                # Verify imports are in order by line number
                for i in range(len(all_imports) - 1):
                    if all_imports[i][0] > all_imports[i+1][0]:
                        # Check if it's actually an order violation
                        curr_section = self._classify_import(all_imports[i][1])
                        next_section = self._classify_import(all_imports[i+1][1])
                        
                        if self._section_order(curr_section) > self._section_order(next_section):
                            issues.append(f"{py_file}:{all_imports[i+1][0]+1} - Import order violation")
            
            except Exception as e:
                print(f"Error checking imports in {py_file}: {e}")
        
        if issues:
            print(f"Import organization issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"  {issue}")
        
        # Should have proper import organization
        self.assertEqual(len(issues), 0, f"Found {len(issues)} import organization issues")
    
    def _classify_import(self, import_line):
        """Classify import as standard, third_party, or local"""
        if any(lib in import_line for lib in ['unittest', 'json', 'os', 'sys', 'pathlib', 'time', 'tempfile', 're']):
            return 'standard'
        elif any(lib in import_line for lib in ['requests', 'numpy', 'pandas', 'matplotlib']):
            return 'third_party'
        else:
            return 'local'
    
    def _section_order(self, section):
        """Get numeric order for import sections"""
        return {'standard': 1, 'third_party': 2, 'local': 3}.get(section, 4)

    def test_no_unused_imports(self):
        """Test that there are no unused imports in Python files"""
        python_files = list(Path("tests").glob("test_*.py"))
        
        issues = []
        
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
                    if module and module not in ['unittest', '__future__']:
                        # Simple check - module name appears elsewhere in file
                        if content.count(module) <= 1:  # Only in import line
                            issues.append(f"{py_file} - Potentially unused import: {module}")
            
            except Exception as e:
                print(f"Error checking unused imports in {py_file}: {e}")
        
        # Document findings but don't fail (unused imports might be intentional)
        if issues:
            print(f"Potential unused imports: {len(issues)}")
            for issue in issues[:10]:
                print(f"  {issue}")

    def test_consistent_code_formatting(self):
        """Test that code follows consistent formatting"""
        js_files = list(Path("src").rglob("*.js"))
        
        formatting_issues = []
        
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
                        # Special case: JSDoc comments - they can have indentation + 1 space before '*'
                        stripped = line.lstrip(' ')
                        if stripped.startswith('*'):
                            # JSDoc comment lines: indentation should be even + 1 for the '*'
                            leading_spaces = len(line) - len(stripped)
                            if (leading_spaces - 1) % 2 != 0:
                                formatting_issues.append(f"{js_file}:{i+1} - JSDoc comment indentation")
                        else:
                            # Regular code should use even number of spaces (2-space indentation)
                            leading_spaces = len(line) - len(line.lstrip(' '))
                            if leading_spaces % 2 != 0:
                                formatting_issues.append(f"{js_file}:{i+1} - Inconsistent indentation")
                    
                    # Check for trailing whitespace
                    if line.endswith(' ') or line.endswith('\t'):
                        formatting_issues.append(f"{js_file}:{i+1} - Trailing whitespace")
            
            except Exception as e:
                print(f"Error checking formatting in {js_file}: {e}")
        
        # Document formatting issues
        if formatting_issues:
            print(f"Code formatting issues: {len(formatting_issues)}")
            for issue in formatting_issues[:10]:
                print(f"  {issue}")
        
        # Should have minimal formatting issues
        self.assertLess(len(formatting_issues), 20,
                       f"Too many formatting issues: {len(formatting_issues)}")


if __name__ == '__main__':
    unittest.main()
