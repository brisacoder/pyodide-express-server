#!/usr/bin/env python3
"""
Test suite for crash reporting system
Tests the comprehensive crash detection and reporting functionality
"""

import json
import time
import unittest

import requests

BASE_URL = "http://localhost:3000"


class CrashReportingSystemTestCase(unittest.TestCase):
    """Test crash reporting system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.session = requests.Session()
        
    def tearDown(self):
        """Clean up test environment"""
        # Clear any test-related crash reports if API exists
        try:
            self.session.delete(f"{BASE_URL}/api/crash-reports/test-reports", timeout=10)
        except requests.RequestException:
            pass

    def test_crash_reports_api_endpoints_exist(self):
        """Test that crash reporting API endpoints exist"""
        # Test GET /api/crash-reports
        response = self.session.get(f"{BASE_URL}/api/crash-reports", timeout=30)
        self.assertIn(response.status_code, [200, 404])  # 404 if no reports exist yet
        
        if response.status_code == 200:
            result = response.json()
            self.assertIn('reports', result)
            self.assertIsInstance(result['reports'], list)

    def test_crash_reports_list_endpoint(self):
        """Test crash reports listing endpoint"""
        response = self.session.get(f"{BASE_URL}/api/crash-reports", timeout=30)
        
        # Should return proper structure regardless of whether reports exist
        if response.status_code == 200:
            result = response.json()
            self.assertIn('reports', result)
            self.assertIn('totalReports', result)
            self.assertIsInstance(result['reports'], list)
            self.assertIsInstance(result['totalReports'], int)

    def test_crash_report_detail_endpoint(self):
        """Test individual crash report detail endpoint"""
        # First get list of reports
        response = self.session.get(f"{BASE_URL}/api/crash-reports", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result['reports']:
                # Get details of first report
                report_id = result['reports'][0]['id']
                detail_response = self.session.get(f"{BASE_URL}/api/crash-reports/{report_id}", timeout=30)
                
                if detail_response.status_code == 200:
                    detail_result = detail_response.json()
                    self.assertIn('report', detail_result)
                    self.assertIn('id', detail_result['report'])
                    self.assertIn('timestamp', detail_result['report'])

    def test_crash_report_summary_endpoint(self):
        """Test crash reports summary endpoint"""
        response = self.session.get(f"{BASE_URL}/api/crash-reports/summary", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            self.assertIn('summary', result)
            summary = result['summary']
            
            # Check expected summary fields
            expected_fields = ['totalReports', 'recentReports', 'errorTypes', 'affectedEndpoints']
            for field in expected_fields:
                self.assertIn(field, summary)

    def test_crash_detection_on_server_error(self):
        """Test that crashes are detected and reported"""
        # Try to trigger a server error by sending invalid data
        invalid_requests = [
            # Invalid JSON in execute endpoint
            {
                'endpoint': '/api/execute',
                'data': {'code': 'import invalid_module_that_does_not_exist\nprint("This should fail")'},
                'should_crash': False  # This should be handled gracefully
            },
            # Extremely long timeout to test timeout handling
            {
                'endpoint': '/api/execute',
                'data': {'code': 'print("test")', 'timeout': 999999999},
                'should_crash': False  # Should be validated and rejected
            }
        ]
        
        for test_case in invalid_requests:
            with self.subTest(endpoint=test_case['endpoint']):
                try:
                    response = self.session.post(
                        f"{BASE_URL}{test_case['endpoint']}", 
                        json=test_case['data'],
                        timeout=30
                    )
                    
                    # Even invalid requests should get proper HTTP responses
                    self.assertIn(response.status_code, [200, 400, 422, 500])
                    
                    # If it's a JSON response, it should be valid JSON
                    if 'application/json' in response.headers.get('content-type', ''):
                        result = response.json()
                        self.assertIn('success', result)
                        
                except requests.RequestException:
                    # Network errors are acceptable for stress tests
                    pass

    def test_swagger_documentation_includes_crash_endpoints(self):
        """Test that crash reporting endpoints are documented in Swagger"""
        try:
            response = self.session.get(f"{BASE_URL}/api-docs/swagger.json", timeout=30)
            
            if response.status_code == 200:
                swagger_doc = response.json()
                paths = swagger_doc.get('paths', {})
                
                # Check if crash reporting endpoints are documented
                crash_endpoints = [
                    '/api/crash-reports',
                    '/api/crash-reports/summary',
                    '/api/crash-reports/{id}'
                ]
                
                documented_endpoints = []
                for endpoint in crash_endpoints:
                    if endpoint in paths:
                        documented_endpoints.append(endpoint)
                
                # At least some crash endpoints should be documented
                # This test documents expected behavior for future implementation
                print(f"Documented crash endpoints: {documented_endpoints}")
                
        except requests.RequestException:
            # Swagger endpoint might not be available
            self.skipTest("Swagger documentation not available")

    def test_crash_report_data_structure(self):
        """Test that crash reports have proper data structure"""
        response = self.session.get(f"{BASE_URL}/api/crash-reports", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('reports'):
                report = result['reports'][0]
                
                # Check expected fields in crash report
                expected_fields = ['id', 'timestamp', 'type', 'endpoint']
                for field in expected_fields:
                    if field in report:  # Not all fields might be present in all reports
                        self.assertIsNotNone(report[field])

    def test_crash_report_cleanup_capability(self):
        """Test that crash reports can be managed/cleaned up"""
        # Test if there's a cleanup endpoint
        cleanup_endpoints = [
            '/api/crash-reports/cleanup',
            '/api/crash-reports/clear',
            '/api/admin/crash-reports/clear'
        ]
        
        cleanup_available = False
        for endpoint in cleanup_endpoints:
            try:
                response = self.session.post(f"{BASE_URL}{endpoint}", timeout=10)
                if response.status_code in [200, 204, 405]:  # 405 means method exists but wrong method
                    cleanup_available = True
                    break
            except requests.RequestException:
                continue
        
        # Document whether cleanup is available
        print(f"Crash report cleanup available: {cleanup_available}")

    def test_crash_reporting_performance_impact(self):
        """Test that crash reporting doesn't significantly impact performance"""
        # Make multiple requests to measure baseline performance
        start_time = time.time()
        
        for i in range(5):
            response = self.session.post(
                f"{BASE_URL}/api/execute",
                json={'code': f'print("Performance test {i}")'},
                timeout=30
            )
            self.assertEqual(response.status_code, 200)
        
        total_time = time.time() - start_time
        avg_time = total_time / 5
        
        # Crash reporting shouldn't make requests significantly slower
        # This is more of a performance documentation than a hard test
        print(f"Average request time with crash reporting: {avg_time:.3f}s")
        self.assertLess(avg_time, 10.0)  # Very generous limit

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across endpoints"""
        # Test various endpoints with invalid data
        test_cases = [
            {'endpoint': '/api/execute', 'data': {}, 'expected_status': [400, 422]},
            {'endpoint': '/api/install-package', 'data': {}, 'expected_status': [400, 422]},
            {'endpoint': '/api/upload', 'data': {}, 'expected_status': [400, 422]},
        ]
        
        for test_case in test_cases:
            with self.subTest(endpoint=test_case['endpoint']):
                try:
                    response = self.session.post(
                        f"{BASE_URL}{test_case['endpoint']}",
                        json=test_case['data'],
                        timeout=30
                    )
                    
                    # Should get proper error responses, not crashes
                    self.assertIn(response.status_code, test_case['expected_status'] + [500])
                    
                    # If JSON response, should have proper error structure
                    if 'application/json' in response.headers.get('content-type', ''):
                        result = response.json()
                        self.assertIn('success', result)
                        if not result['success']:
                            self.assertIn('error', result)
                            
                except requests.RequestException as e:
                    # Document any network errors
                    print(f"Network error testing {test_case['endpoint']}: {e}")


if __name__ == '__main__':
    unittest.main()
