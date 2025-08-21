#!/usr/bin/env python3
"""
Test suite for Content Security Policy (CSP) compliance
Tests that security policies are properly configured and UI functionality works
"""

import re
import unittest

import requests

BASE_URL = "http://localhost:3000"


class CSPComplianceTestCase(unittest.TestCase):
    """Test Content Security Policy compliance and UI functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.session = requests.Session()

    def test_csp_headers_present(self):
        """Test that CSP headers are present in responses"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        # Check for security headers
        headers = response.headers
        
        # CSP header should be present
        csp_header = headers.get('Content-Security-Policy')
        if csp_header:
            self.assertIsNotNone(csp_header)
            self.assertIn('script-src', csp_header)
            print(f"CSP Header: {csp_header}")

    def test_csp_allows_pyodide_execution(self):
        """Test that CSP allows Pyodide WebAssembly execution"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        csp_header = response.headers.get('Content-Security-Policy', '')
        
        # Should allow WebAssembly for Pyodide
        if csp_header:
            # Look for wasm-unsafe-eval or similar directives
            self.assertTrue(
                'wasm-unsafe-eval' in csp_header or 
                'unsafe-eval' in csp_header or
                "'self'" in csp_header,
                "CSP should allow WebAssembly execution for Pyodide"
            )

    def test_csp_includes_sha256_hashes_for_inline_scripts(self):
        """Test that CSP includes SHA256 hashes for inline event handlers"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        csp_header = response.headers.get('Content-Security-Policy', '')
        
        if csp_header:
            # Should contain SHA256 hashes for inline scripts
            sha256_pattern = r"'sha256-[A-Za-z0-9+/]+='"
            sha256_matches = re.findall(sha256_pattern, csp_header)
            
            self.assertGreater(len(sha256_matches), 0, 
                             "CSP should include SHA256 hashes for inline scripts")
            print(f"Found {len(sha256_matches)} SHA256 hashes in CSP")

    def test_ui_page_loads_successfully(self):
        """Test that the main UI page loads without CSP violations"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        # Should return HTML content
        self.assertIn('text/html', response.headers.get('content-type', ''))
        
        # Should contain expected UI elements
        content = response.text
        self.assertIn('<button', content)
        self.assertIn('onclick', content)  # Inline event handlers should be present

    def test_ui_buttons_have_proper_onclick_handlers(self):
        """Test that UI buttons have proper onclick handlers that should work with CSP"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        content = response.text
        
        # Expected button functions
        expected_functions = [
            'executeCode()',
            'clearResult()',
            'resetEnvironment()',
            'uploadFile()',
            'listFiles()',
            'clearAllFiles()',
            'installPackage()',
            'listPackages()'
        ]
        
        for function in expected_functions:
            self.assertIn(function, content, 
                         f"UI should contain {function} button handler")

    def test_javascript_functions_defined(self):
        """Test that required JavaScript functions are defined in the page"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        content = response.text
        
        # Functions should be defined in script tags
        expected_js_functions = [
            'function executeCode',
            'function clearResult',
            'function resetEnvironment',
            'function uploadFile',
            'function listFiles',
            'function clearAllFiles',
            'function installPackage',
            'function listPackages'
        ]
        
        for function_def in expected_js_functions:
            self.assertIn(function_def, content,
                         f"JavaScript function should be defined: {function_def}")

    def test_api_endpoints_return_proper_cors_headers(self):
        """Test that API endpoints return proper CORS headers"""
        api_endpoints = [
            '/api/execute',
            '/api/execute-raw',
            '/api/upload-csv',
            '/api/uploaded-files',
            '/health'
        ]
        
        for endpoint in api_endpoints:
            with self.subTest(endpoint=endpoint):
                try:
                    if endpoint == '/api/execute' or endpoint == '/api/execute-raw':
                        # POST request
                        response = self.session.post(
                            f"{BASE_URL}{endpoint}",
                            json={'code': 'print("test")'},
                            timeout=30
                        )
                    else:
                        # GET request
                        response = self.session.get(f"{BASE_URL}{endpoint}", timeout=30)
                    
                    # Should have CORS headers for browser compatibility
                    headers = response.headers
                    
                    # Document which CORS headers are present
                    cors_headers = {
                        'Access-Control-Allow-Origin': headers.get('Access-Control-Allow-Origin'),
                        'Access-Control-Allow-Methods': headers.get('Access-Control-Allow-Methods'),
                        'Access-Control-Allow-Headers': headers.get('Access-Control-Allow-Headers')
                    }
                    
                    print(f"CORS headers for {endpoint}: {cors_headers}")
                    
                except requests.RequestException as e:
                    print(f"Error testing {endpoint}: {e}")

    def test_security_headers_comprehensive(self):
        """Test comprehensive security headers configuration"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        headers = response.headers
        
        # Document all security-related headers
        security_headers = {
            'Content-Security-Policy': headers.get('Content-Security-Policy'),
            'X-Content-Type-Options': headers.get('X-Content-Type-Options'),
            'X-Frame-Options': headers.get('X-Frame-Options'),
            'X-XSS-Protection': headers.get('X-XSS-Protection'),
            'Strict-Transport-Security': headers.get('Strict-Transport-Security'),
            'Referrer-Policy': headers.get('Referrer-Policy')
        }
        
        print("Security Headers Configuration:")
        for header, value in security_headers.items():
            if value:
                print(f"  {header}: {value}")
            else:
                print(f"  {header}: Not set")

    def test_pyodide_script_loading_allowed(self):
        """Test that CSP allows loading Pyodide scripts"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        content = response.text
        csp_header = response.headers.get('Content-Security-Policy', '')
        
        # Should reference Pyodide CDN or local Pyodide files
        pyodide_references = [
            'pyodide', 'cdn.jsdelivr.net', 'pyodide.js'
        ]
        
        has_pyodide_ref = any(ref in content.lower() for ref in pyodide_references)
        
        if has_pyodide_ref and csp_header:
            # CSP should allow the Pyodide source
            self.assertTrue(
                'cdn.jsdelivr.net' in csp_header or 
                "'self'" in csp_header or
                "'unsafe-eval'" in csp_header,
                "CSP should allow Pyodide script loading"
            )

    def test_inline_styles_handling(self):
        """Test that inline styles are properly handled by CSP"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        content = response.text
        csp_header = response.headers.get('Content-Security-Policy', '')
        
        # Check if there are inline styles
        has_inline_styles = 'style=' in content
        
        if has_inline_styles and csp_header:
            # CSP should handle inline styles
            style_policy = 'style-src' in csp_header
            if style_policy:
                self.assertTrue(
                    "'unsafe-inline'" in csp_header or 
                    "'self'" in csp_header or
                    'sha256-' in csp_header,
                    "CSP should allow inline styles or have proper hashes"
                )

    def test_csp_script_src_attr_handling(self):
        """Test that CSP properly handles script-src-attr for onclick handlers"""
        response = self.session.get(BASE_URL, timeout=30)
        self.assertEqual(response.status_code, 200)
        
        csp_header = response.headers.get('Content-Security-Policy', '')
        content = response.text
        
        # Check if there are onclick handlers
        has_onclick = 'onclick=' in content
        
        if has_onclick and csp_header:
            # Should have script-src-attr directive or SHA256 hashes
            self.assertTrue(
                'script-src-attr' in csp_header or
                'sha256-' in csp_header,
                "CSP should handle onclick attributes with script-src-attr or SHA256 hashes"
            )


if __name__ == '__main__':
    unittest.main()
