#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner for Pyodide Express Server

This script runs all test categories and provides detailed reporting.
Use this to validate the entire system after changes.
"""

import os
import sys
import unittest
import argparse
import time
from io import StringIO

# Test modules to import
test_modules = [
    'tests.test_api',
    'tests.test_error_handling', 
    'tests.test_integration',
    'tests.test_security',
    'tests.test_performance'
]

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
        test_time = time.time() - self.test_start_time
        self.test_times[str(test)] = test_time
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_details[str(test)] = {'status': 'PASS', 'time': self.test_times[str(test)]}
        
    def addError(self, test, err):
        super().addError(test, err)
        self.test_details[str(test)] = {'status': 'ERROR', 'time': self.test_times[str(test)], 'error': str(err[1])}
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_details[str(test)] = {'status': 'FAIL', 'time': self.test_times[str(test)], 'error': str(err[1])}
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_details[str(test)] = {'status': 'SKIP', 'time': 0, 'reason': reason}

class ComprehensiveTestRunner:
    """Main test runner for all test categories."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.results = {}
        
    def run_test_category(self, module_name, category_name):
        """Run tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"Running {category_name} Tests")
        print(f"{'='*60}")
        
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
            ('tests.test_error_handling', 'Error Handling'),
            ('tests.test_integration', 'Integration'),
            ('tests.test_security', 'Security'),
            ('tests.test_performance', 'Performance')
        ]
        
        if selected_categories:
            categories = [(module, name) for module, name in categories 
                         if name.lower() in [c.lower() for c in selected_categories]]
        
        print("Starting Comprehensive Test Suite")
        print(f"Running {len(categories)} test categories...")
        
        all_successful = True
        for module_name, category_name in categories:
            success = self.run_test_category(module_name, category_name)
            all_successful = all_successful and success
            
        return self.print_overall_summary()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for Pyodide Express Server')
    parser.add_argument('--categories', nargs='*', 
                       choices=['basic', 'error', 'integration', 'security', 'performance'],
                       help='Specific test categories to run (default: all)')
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
        'error': 'Error Handling', 
        'integration': 'Integration',
        'security': 'Security',
        'performance': 'Performance'
    }
    
    selected_categories = None
    if args.categories:
        selected_categories = [category_map[cat] for cat in args.categories]
    
    # Run tests
    runner = ComprehensiveTestRunner(verbosity=verbosity)
    success = runner.run_all_tests(selected_categories)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
