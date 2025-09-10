#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner for Pyodide Express Server

This script runs all test categories and provides detailed reporting.
Use this to validate the entire system after changes.
"""

import argparse
import os
import subprocess
import sys
import time
import unittest
from io import StringIO

import requests

# Test modules to import
test_modules = [
    'tests.test_api',
    'tests.test_basic_error',
    'tests.test_code_quality_compliance',
    'tests.test_container_filesystem',
    'tests.test_crash_reporting_system',
    'tests.test_csp_compliance',
    'tests.test_debug_filesystem_plots',
    'tests.test_direct_filesystem_mount',
    'tests.test_dynamic_modules_and_execution_robustness',
    'tests.test_error_handling',
    'tests.test_extract_plots',
    'tests.test_extract_virtual_file_implementation',
    'tests.test_file_management_enhancements',
    'tests.test_file_upload_enhancements',
    'tests.test_filesystem_mount',
    'tests.test_filesystem_persistence',
    'tests.test_function_return_patterns',
    'tests.test_integration',
    'tests.test_javascript_module',
    'tests.test_matplotlib_base64',
    'tests.test_matplotlib.py',
    'tests.test_matplotlib_filesystem',
    'tests.test_nissan_leaf_telemetry',
    'tests.test_non_happy_paths',
    'tests.test_package_management',
    'tests.test_performance',
    'tests.test_reset',
    'tests.test_return_data_types',
    'tests.test_seaborn_base64',
    'tests.test_seaborn_filesystem',
    'tests.test_seaborn',
    'tests.test_security',
    'tests.test_security_info_disclosure',
    'tests.test_security_logging',
    'tests.test_simple_file_creation',
    'tests.test_simple_filesystem',
    'tests.test_sklearn',
    'tests.test_user_isolation',
    'tests.test_virtual_filesystem'
]


class ServerManager:
    """Manages the test server lifecycle with crash recovery."""
    
    def __init__(self, base_url="http://localhost:3000", timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        self.server_process = None
        
    def start_server(self):
        """Start the server and wait for it to be ready."""
        if self.server_process:
            self.stop_server()
            
        print("Starting server...")
        try:
            # Start server in subprocess with no pipes to avoid hanging
            self.server_process = subprocess.Popen(
                ["node", "src/server.js"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            
            print("Server process started, waiting for it to be ready...")
            
            # Wait for server to be ready
            start_time = time.time()
            dots = 0
            while time.time() - start_time < self.timeout:
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    if response.status_code == 200:
                        print("\n✅ Server is ready")
                        return True
                except Exception:
                    pass
                
                # Show progress dots
                if dots % 5 == 0:
                    print(".", end="", flush=True)
                dots += 1
                time.sleep(1)
                
            print(f"\n❌ Server failed to start within {self.timeout} seconds")
            self.stop_server()
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            try:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait()
            except Exception as e:
                print(f"Warning: Error stopping server: {e}")
            finally:
                self.server_process = None
    
    def is_server_running(self):
        """Check if server is running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def restart_server_if_needed(self):
        """Restart server if it's not running."""
        if not self.is_server_running():
            print("🔄 Server appears to be down, restarting...")
            return self.start_server()
        return True


class DetailedTestResult(unittest.TextTestResult):
    """Enhanced test result with timing and detailed reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.test_details = {}
        
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        if hasattr(self, 'test_start_time'):
            test_time = time.time() - self.test_start_time
            self.test_times[str(test)] = test_time
        else:
            # Fallback if startTest wasn't called properly
            self.test_times[str(test)] = 0.0
        
    def addSuccess(self, test):
        super().addSuccess(test)
        test_time = self.test_times.get(str(test), 0.0)
        self.test_details[str(test)] = {'status': 'PASS', 'time': test_time}
        
    def addError(self, test, err):
        super().addError(test, err)
        test_time = self.test_times.get(str(test), 0.0)
        self.test_details[str(test)] = {'status': 'ERROR', 'time': test_time, 'error': str(err[1])}
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_time = self.test_times.get(str(test), 0.0)
        self.test_details[str(test)] = {'status': 'FAIL', 'time': test_time, 'error': str(err[1])}
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_details[str(test)] = {'status': 'SKIP', 'time': 0, 'reason': reason}


class ComprehensiveTestRunner:
    """Main test runner for all test categories."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.results = {}
        self.server_manager = ServerManager()
        
    def run_test_category(self, module_name, category_name):
        """Run tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"Running {category_name} Tests")
        print(f"{'='*60}")
        
        # Security tests can crash the server, so ensure it's running before each category
        if category_name == 'Security':
            print("⚠️  Security tests may cause server instability - ensuring server is ready...")
            if not self.server_manager.restart_server_if_needed():
                print("❌ Could not start server for security tests")
                self.results[category_name] = {
                    'error': 'Server startup failed',
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'skipped': 0
                }
                return False
        
        try:
            # Import the test module
            test_module = __import__(module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with detailed results
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=self.verbosity,
                resultclass=DetailedTestResult
            )
            
            start_time = time.time()
            result = runner.run(suite)
            total_time = time.time() - start_time
            
            # Store results
            self.results[category_name] = {
                'result': result,
                'total_time': total_time,
                'output': stream.getvalue(),
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped)
            }
            
            # Print summary for this category
            self.print_category_summary(category_name)
            
            # Check if server is still running after security tests
            if category_name == 'Security' and not self.server_manager.is_server_running():
                print("⚠️  Server crashed during security tests (this may be expected behavior)")
            
            return result.wasSuccessful()
            
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            self.results[category_name] = {
                'error': str(e),
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0
            }
            return False


    def print_category_summary(self, category_name):
        """Print summary for a test category."""
        result_data = self.results[category_name]
        
        if 'error' in result_data:
            print(f"❌ {category_name}: Import Error - {result_data['error']}")
            return
            
        result = result_data['result']
        total_time = result_data['total_time']
        
        status = "✅ PASSED" if result.wasSuccessful() else "❌ FAILED"
        print(f"{status} {category_name}:")
        print(f"  Tests Run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Skipped: {len(result.skipped)}")
        print(f"  Time: {total_time:.2f}s")
        
        # Show slowest tests
        if hasattr(result, 'test_details'):
            slow_tests = sorted(
                [(test, details['time']) for test, details in result.test_details.items()
                 if details['status'] == 'PASS'],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            if slow_tests:
                print("  Slowest tests:")
                for test_name, test_time in slow_tests:
                    test_short = test_name.split('.')[-1]
                    print(f"    {test_short}: {test_time:.2f}s")
                    
        # Show failures/errors
        if result.failures:
            print("  Failures:")
            for test, trace in result.failures:
                print(f"    {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
                
        if result.errors:
            print("  Errors:")
            for test, trace in result.errors:
                print(f"    {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
    
    def print_overall_summary(self):
        """Print overall test suite summary."""
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = sum(r.get('tests_run', 0) for r in self.results.values())
        total_failures = sum(r.get('failures', 0) for r in self.results.values())
        total_errors = sum(r.get('errors', 0) for r in self.results.values())
        total_skipped = sum(r.get('skipped', 0) for r in self.results.values())
        total_time = sum(r.get('total_time', 0) for r in self.results.values() if 'total_time' in r)
        
        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_tests - total_failures - total_errors}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        
        # Category breakdown
        print(f"\nCategory Breakdown:")
        for category, result_data in self.results.items():
            if 'error' in result_data:
                print(f"  {category}: ❌ Import Error")
            else:
                tests = result_data['tests_run']
                failures = result_data['failures']
                errors = result_data['errors']
                status = "✅" if failures == 0 and errors == 0 else "❌"
                print(f"  {category}: {status} {tests - failures - errors}/{tests} passed")
                
        # Overall status
        overall_success = total_failures == 0 and total_errors == 0 and total_tests > 0
        if overall_success:
            print(f"\n🎉 ALL TESTS PASSED! System is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Please review the failures above.")
            
        return overall_success
    
    def run_all_tests(self, selected_categories=None):
        """Run all test categories."""
        categories = [
            ('tests.test_api', 'Basic API'),
            ('tests.test_basic_errors', 'Basic Error'),
            ('tests.test_code_quality_compliance', 'Code Quality Compliance'),
            ('tests.test_container_filesystem', 'Container Filesystem'),
            ('tests.test_crash_reporting_system', 'Crash Reporting System'),
            ('tests.test_csp_compliance', 'CSP Compliance'),
            ('tests.test_debug_filesystem_plots', 'Debug Filesystem Plots'),
            ('tests.test_direct_filesystem_mount', 'Direct Filesystem Mount'),
            ('tests.test_dynamic_modules_and_execution_robustness', 'Dynamic Modules & Execution Robustness'),
            ('tests.test_error_handling', 'Error Handling'),
            ('tests.test_extract_plots_api', 'Extract Plots'),
            ('tests.test_extract_virtual_file_implementation', 'Extract Virtual File Implementation'),
            ('tests.test_file_management_enhancements', 'File Management Enhancements'),
            ('tests.test_file_upload_enhancements', 'File Upload Enhancements'),
            ('tests.test_filesystem_mount', 'Filesystem Mount'),
            ('tests.test_filesystem_persistence', 'Filesystem Persistence'),
            ('tests.test_function_return_patterns', 'Function Return Patterns'),
            ('tests.test_integration', 'Integration'),
            ('tests.test_javascript_module', 'JavaScript Module'),
            ('tests.test_matplotlib', 'Matplotlib'),
            ('tests.test_matplotlib_base64', 'Matplotlib Base64 Plots'),
            ('tests.test_matplotlib_filesystem', 'Matplotlib VFS Plots'),
            ('tests.test_nissan_leaf_telemetry', 'Nissan Leaf Telemetry'),
            ('tests.test_non_happy_paths', 'Extra Non-Happy Paths'),
            ('tests.test_package_management', 'Package Management'),
            ('tests.test_performance', 'Performance'),
            ('tests.test_reset', 'Reset'),
            ('tests.test_return_data_types', 'Return Data Types'),
            ('tests.test_seaborn', 'Seaborn'),
            ('tests.test_seaborn_base64', 'Seaborn Base64 Plots'),
            ('tests.test_seaborn_filesystem', 'Seaborn VFS Plots'),
            ('tests.test_security', 'Security'),
            ('tests.test_security_info_disclosure', 'Security Info Disclosure'),
            ('tests.test_security_logging', 'Security Logging'),
            ('tests.test_simple_file_creation', 'Simple File Creation'),
            ('tests.test_simple_filesystem', 'Simple Filesystem'),
            ('tests.test_sklearn', 'Scikit-Learn'),
            ('tests.test_user_isolation', 'User Isolation'),
            ('tests.test_virtual_filesystem', 'Virtual Filesystem')
        ]
        categories.sort(key=lambda x: x[0])
        
        if selected_categories:
            categories = [(module, name) for module, name in categories 
                         if name.lower() in [c.lower() for c in selected_categories]]
        
        print("Starting Comprehensive Test Suite")
        print(f"Running {len(categories)} test categories...")
        
        # Start server for testing
        if not self.server_manager.start_server():
            print("❌ Failed to start server. Cannot run tests.")
            return False
        
        try:
            all_successful = True
            for module_name, category_name in categories:
                success = self.run_test_category(module_name, category_name)
                all_successful = all_successful and success
                
                # Restart server after security tests if it crashed
                if category_name == 'Security' and not self.server_manager.is_server_running():
                    print("🔄 Restarting server after security tests...")
                    self.server_manager.start_server()
            
            return self.print_overall_summary()
            
        finally:
            # Always stop the server when done
            print("\n🛑 Stopping test server...")
            self.server_manager.stop_server()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for Pyodide Express Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run native server tests (DEFAULT - fast for development)
  python run_comprehensive_tests.py
  
  # Run ALL tests including container tests (slower)
  python run_comprehensive_tests.py --compose
  
  # Run specific categories
  python run_comprehensive_tests.py --categories basic security function_patterns
  
  # Run container tests only (requires containerized server)
  python run_comprehensive_tests.py --categories container
        """
    )
    parser.add_argument('--categories', nargs='*',
                        choices=[
                            'basic',
                            'basic_errors',
                            'code_quality_compliance',
                            'container',
                            'crash_reporting',
                            'csp',
                            'debug_filesystem_plots',
                            'direct_mount',
                            'dynamic',
                            'error',
                            'extract_plots',
                            'extract_virtual_file_implementation',
                            'file_management',
                            'function_patterns',
                            'filesystem_mount',
                            'filesystem_persistence',
                            'file_upload_enhancements',
                            'integration',
                            'javascript',
                            'matplotlib',
                            'matplotlib_base64',
                            'matplotlib_vfs',
                            'nissan_leaf_telemetry',
                            'non_happy_paths',
                            'package_management',
                            'performance',
                            'reset',
                            'return_data_types',
                            'seaborn',
                            'seaborn_base64',
                            'seaborn_vfs',
                            'security',
                            'security_info_disclosure',
                            'security_logging',
                            'simple_file_creation',
                            'simple_filesystem',
                            'sklearn',
                            'isolation',
                            'user_isolation',
                            'vfs',
                            'virtual_filesystem'
                        ],
                        help='Specific test categories to run')
    parser.add_argument('--compose', action='store_true',
                       help='Run ALL tests including container tests (slower, requires containerized server)')
    parser.add_argument('--native', action='store_true',
                       help='Run only native server tests (DEFAULT behavior - kept for backward compatibility)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet output')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 2  # Default
    if args.verbose:
        verbosity = 3
    elif args.quiet:
        verbosity = 0
    
    # Map category names
    category_map = {
        'basic': 'Basic API',
        'basic_errors': 'Basic Error',
        'code_quality_compliance': 'Code Quality Compliance',
        'container': 'Container Filesystem',
        'crash_reporting': 'Crash Reporting System',
        'csp': 'CSP Compliance',
        'debug_filesystem_plots': 'Debug Filesystem Plots',
        'direct_mount': 'Direct Filesystem Mount',
        'dynamic': 'Dynamic Modules & Execution Robustness',
        'error': 'Error Handling',
        'extract_plots': 'Extract Plots',
        'extract_virtual_file_implementation': 'Extract Virtual File Implementation',
        'file_management': 'File Management Enhancements',
        'file_upload_enhancements': 'File Upload Enhancements',
        'filesystem_mount': 'Filesystem Mount',
        'filesystem_persistence': 'Filesystem Persistence',
        'function_patterns': 'Function Return Patterns',
        'integration': 'Integration',
        'javascript': 'JavaScript Module',
        'matplotlib': 'Matplotlib',
        'matplotlib_base64': 'Matplotlib Base64 Plots',
        'matplotlib_vfs': 'Matplotlib VFS Plots',
        'nissan_leaf_telemetry': 'Nissan Leaf Telemetry',
        'non_happy_paths': 'Extra Non-Happy Paths',
        'package_management': 'Package Management',
        'performance': 'Performance',
        'reset': 'Reset',
        'return_data_types': 'Return Data Types',
        'seaborn': 'Seaborn',
        'seaborn_base64': 'Seaborn Base64 Plots',
        'seaborn_vfs': 'Seaborn VFS Plots',
        'security': 'Security',
        'security_info_disclosure': 'Security Info Disclosure',
        'security_logging': 'Security Logging',
        'simple_file_creation': 'Simple File Creation',
        'simple_filesystem': 'Simple Filesystem',
        'sklearn': 'Scikit-Learn',
        'isolation': 'User Isolation',
        'user_isolation': 'User Isolation',
        'vfs': 'Virtual Filesystem',
        'virtual_filesystem': 'Virtual Filesystem'
    }
    
    selected_categories = None
    if args.categories:
        # Explicit categories specified
        selected_categories = [category_map[cat] for cat in args.categories]
    elif args.compose:
        # Compose mode: run ALL tests including container tests (slower)
        selected_categories = None  # Run all categories
        print("🐳 Running in COMPOSE mode (all tests including container tests)")
        print("   This includes slower container filesystem tests")
        print("   Running: ALL categories including container tests")
    else:
        # Default: Native mode - exclude container tests for faster development
        native_categories = [cat for cat in category_map.keys() if cat != 'container']
        selected_categories = [category_map[cat] for cat in native_categories]
        print("🚀 Running in NATIVE mode (DEFAULT - container tests excluded for faster development)")
        print(f"   Use --compose to include container tests")
        print(f"   Running: {len(selected_categories)} categories (container tests excluded)")
    
    # Run tests
    runner = ComprehensiveTestRunner(verbosity=verbosity)
    success = runner.run_all_tests(selected_categories)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
