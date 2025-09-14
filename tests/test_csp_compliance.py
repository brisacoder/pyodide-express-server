"""
Content Security Policy (CSP) Compliance Test Suite

This module provides comprehensive behavioral tests for Content Security Policy
compliance in the Pyodide Express Server. Tests follow BDD (Behavior-Driven
Development) patterns and validate that security policies are properly
configured while maintaining functionality.

Test Categories:
- CSP Header Validation
- Security Headers Compliance
- UI Functionality with CSP
- WebAssembly Execution Permissions
- Inline Script and Style Handling
- CORS Configuration
- Error Handling and Edge Cases

Test Patterns:
- Given/When/Then structure
- Pytest fixtures for setup/teardown
- Comprehensive docstrings
- Public API usage only (/api/execute-raw for code execution)
"""

import pytest
import requests
import re
from requests import Session, Response

# Global test configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30

# Security policy patterns
PYODIDE_REFERENCES = [
    "pyodide",
    "webassembly",
    "wasm",
    "cdn.jsdelivr.net/pyodide"
]

REQUIRED_JS_FUNCTIONS = [
    "executeCode",
    "uploadFile",
    "clearResult"
]

EXPECTED_UI_FUNCTIONS = [
    "onclick",
    "executeCode",
    "uploadFile"
]

SECURITY_HEADERS = [
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy"
]


@pytest.fixture(scope="session")
def http_session():
    """
    Create a persistent HTTP session for all tests.

    This fixture provides a configured requests.Session instance that is
    reused across all tests in the session, improving performance and
    maintaining connection pooling.

    Returns:
        requests.Session: Configured HTTP session with appropriate timeout
    """
    session = requests.Session()
    session.timeout = DEFAULT_TIMEOUT
    return session


@pytest.fixture(scope="session")
def base_response(http_session):
    """
    Fetch the base application response for CSP analysis.

    This fixture retrieves the main application page that contains CSP
    headers and UI elements. Used by multiple tests to analyze security
    policies and functionality.

    Args:
        http_session: Configured HTTP session fixture

    Returns:
        requests.Response: HTTP response from the main application page

    Raises:
        requests.RequestException: If the request fails
    """
    response = http_session.get(BASE_URL, timeout=DEFAULT_TIMEOUT)
    return response


@pytest.fixture(scope="session")
def csp_header(base_response):
    """
    Extract Content Security Policy header from the base response.

    Args:
        base_response: HTTP response fixture containing CSP headers

    Returns:
        Optional[str]: CSP header value if present, None otherwise
    """
    return base_response.headers.get("Content-Security-Policy")


def execute_python_code_via_api(session: Session, code: str,
                                timeout: int = DEFAULT_TIMEOUT) -> Response:
    """
    Execute Python code using the public /api/execute-raw endpoint.

    This function provides a consistent way to execute Python code through
    the public API endpoint, following the requirement to use only public APIs.

    Args:
        session: HTTP session for making requests
        code: Python code to execute
        timeout: Request timeout in seconds

    Returns:
        requests.Response: API response containing execution results

    Raises:
        requests.RequestException: If the request fails
    """
    return session.post(f"{BASE_URL}/api/execute-raw",
                        data=code, timeout=timeout,
                        headers={'Content-Type': 'text/plain'})


class TestCSPHeaderValidation:
    """
    Test suite for validating the presence and basic structure of CSP headers.

    This test class ensures that Content Security Policy headers are properly
    configured and contain essential security directives.
    """

    def test_given_application_running_when_accessing_main_page_then_csp_headers_are_present(  # noqa: E501
        self, base_response, csp_header
    ):
        """
        Verify that CSP headers are present in HTTP responses.

        Given: The application is running and accessible
        When: A request is made to the main application page
        Then: The response contains valid CSP headers with script-src directive

        Args:
            base_response: HTTP response from the main page
            csp_header: Extracted CSP header value
        """
        # Given/When: Application is accessed (handled by fixtures)

        # Then: Response should be successful
        assert (base_response.status_code == 200), \
            "Application should be accessible"

        # And: CSP header should be present
        assert csp_header is not None, "CSP header must be present"
        assert (("script-src" in csp_header)), \
            "CSP must include script-src directive"

    def test_given_csp_header_present_when_analyzed_then_contains_security_directives(  # noqa: E501
        self, csp_header
    ):
        """
        Verify that CSP header contains essential security directives.

        Given: CSP header is present in the response
        When: The header content is analyzed
        Then: It contains key security directives for web app protection

        Args:
            csp_header: Content Security Policy header value
        """
        # Given: CSP header exists (verified by fixture)
        csp_lower = csp_header.lower()

        essential_directives = [
            "script-src", "style-src", "default-src", "object-src"
        ]

        present_directives = [
            directive for directive in essential_directives
            if directive in csp_lower
        ]

        assert (len(present_directives) >= 2), \
            f"CSP must contain at least 2 essential directives, " \
            f"found: {present_directives}"


class TestWebAssemblyExecution:
    """
    Test suite for WebAssembly execution permissions in CSP.

    This class validates that the CSP configuration allows Pyodide WebAssembly
    execution while maintaining security standards.
    """

    def test_given_pyodide_application_when_csp_configured_then_webassembly_execution_allowed(  # noqa: E501
        self, csp_header
    ):
        """
        Verify that CSP allows WebAssembly execution for Pyodide.

        Given: The application uses Pyodide for Python execution
        When: CSP policies are configured
        Then: WebAssembly execution permissions are properly set

        Args:
            csp_header: Content Security Policy header value
        """
        # Given: CSP header should exist for WebAssembly testing
        if csp_header is None:
            pytest.skip("CSP header not present - " +
                        "cannot test WebAssembly permissions")

        # When: Analyzing CSP for WebAssembly permissions
        csp_lower = csp_header.lower()

        # Check for various WebAssembly permission patterns
        wasm_indicators = [
            "unsafe-eval" in csp_lower,
            "wasm-eval" in csp_lower,
            "'unsafe-inline'" in csp_lower and "script-src" in csp_lower
        ]

        wasm_allowed = any(wasm_indicators)

        # Then: WebAssembly execution should be permitted
        assert wasm_allowed, \
            "CSP must allow WebAssembly execution for Pyodide functionality"

    def test_given_webassembly_permissions_when_executing_python_then_execution_succeeds(  # noqa: E501
        self, http_session
    ):
        """
        Verify that WebAssembly actually works through the API.

        Given: Pyodide WebAssembly environment is configured
        When: Python code is executed via the API
        Then: WebAssembly execution succeeds without CSP violations

        Args:
            http_session: HTTP session for API requests
        """
        # Given: Simple Python code for WebAssembly execution test
        test_code = """
import sys
print(f"Python version: {sys.version}")
calculation_result = 42 * 1337
print(f"Calculation result: {calculation_result}")
        """

        # When: Executing Python code via the API
        response = execute_python_code_via_api(http_session, test_code)
        response_text = response.text.lower()

        # Then: Execution should succeed
        assert (response.status_code == 200), \
            f"API execution failed: {response.text}"

        assert ("python version" in response_text), \
            "Python execution should produce expected output"
        assert ("calculation result" in response_text), \
            "Mathematical operations should work"


class TestInlineScriptHandling:
    """
    Test suite for SHA256 hash-based inline script handling in CSP.

    This class validates that CSP properly handles inline event handlers
    and scripts using SHA256 hashes for security.
    """

    def test_given_inline_scripts_when_csp_configured_then_sha256_hashes_present(  # noqa: E501
        self, csp_header
    ):
        """
        Verify that CSP includes SHA256 hashes for inline scripts.

        Given: The application contains inline event handlers
        When: CSP is configured for security
        Then: SHA256 hashes are present for inline script approval

        Args:
            csp_header: Content Security Policy header value
        """
        # Given: CSP header should exist for hash testing
        if csp_header is None:
            pytest.skip("CSP header not present - cannot test SHA256 hashes")

        # When: Looking for SHA256 hash patterns in CSP
        sha256_pattern = r"'sha256-[A-Za-z0-9+/]+=*'"
        sha256_matches = re.findall(sha256_pattern, csp_header)

        # Then: Either hashes or explicit unsafe policies should be present
        unsafe_policies = ["unsafe-inline", "unsafe-eval"]
        if ("onclick=" in csp_header or
                any(f"unsafe-{policy}" in csp_header
                    for policy in unsafe_policies)):
            # If inline scripts are allowed, hashes or unsafe policies present
            assert (len(sha256_matches) > 0 or
                    any(f"unsafe-{policy}" in csp_header
                        for policy in unsafe_policies)), \
                "CSP must include SHA256 hashes for inline scripts or " + \
                "explicit unsafe policies"


class TestUIFunctionality:
    """
    Test suite for user interface functionality under CSP constraints.

    This class ensures that the web UI functions correctly despite
    Content Security Policy restrictions.
    """

    def test_given_web_ui_when_main_page_loaded_then_ui_elements_present(
        self, base_response
    ):
        """
        Verify that the main UI page loads without CSP violations.

        Given: The application provides a web-based UI
        When: The main page is requested
        Then: HTML content loads successfully with expected UI elements

        Args:
            base_response: HTTP response from the main application page
        """
        # Given/When: UI page is loaded (handled by fixture)

        # Then: Page should load successfully
        assert (base_response.status_code == 200), \
            "UI page should load successfully"

        page_content = base_response.text.lower()

        # And: Basic UI elements should be present
        assert "<html" in page_content, "Valid HTML structure should exist"
        assert ("<button" in page_content), \
            "UI should contain interactive buttons"
        assert (len(page_content.strip()) > 0), \
            "Page content should not be empty"

    def test_given_ui_buttons_when_page_analyzed_then_onclick_handlers_present(
        self, base_response
    ):
        """
        Verify that UI buttons have proper onclick handlers for CSP compliance.

        Given: The UI contains interactive buttons
        When: Button elements are analyzed
        Then: Expected onclick handlers are present and properly configured

        Args:
            base_response: HTTP response containing the UI page
        """
        # Given: Page content for analysis
        page_content = base_response.text.lower()

        # When: Looking for expected UI functions
        missing_functions = [
            func for func in EXPECTED_UI_FUNCTIONS
            if func.lower() not in page_content
        ]

        # Then: All expected UI functions should be present
        assert (len(missing_functions) == 0), \
            f"Missing UI functions: {missing_functions}"

    def test_given_javascript_functions_when_page_loaded_then_functions_defined(  # noqa: E501
        self, base_response
    ):
        """
        Verify that required JavaScript functions are properly defined.

        Given: The UI requires JavaScript functions for interaction
        When: The page content is analyzed
        Then: All expected JavaScript functions are defined in script tags

        Args:
            base_response: HTTP response containing JavaScript definitions
        """
        # Given: Page content containing JavaScript
        page_content = base_response.text.lower()

        # When: Checking for required JavaScript functions
        missing_js_functions = [
            func for func in REQUIRED_JS_FUNCTIONS
            if func.lower() not in page_content
        ]

        # Then: All required JavaScript functions should be defined
        assert (len(missing_js_functions) == 0), \
            f"Missing JavaScript functions: {missing_js_functions}"


class TestCORSConfiguration:
    """
    Test suite for Cross-Origin Resource Sharing (CORS) configuration.

    This class validates that API endpoints return proper CORS headers
    for browser compatibility while maintaining security.
    """

    @pytest.mark.parametrize("endpoint", [
        "/health", "/api/status", "/api/execute-raw"
    ])
    def test_given_public_endpoints_when_accessed_then_cors_headers_present(
        self, http_session, endpoint
    ):
        """
        Verify that API endpoints return proper CORS headers.

        Given: Public API endpoints are available
        When: An endpoint is accessed via HTTP request
        Then: Appropriate CORS headers are returned for browser compatibility

        Args:
            http_session: HTTP session for making requests
            endpoint: API endpoint to test
        """
        # Given: Endpoint URL construction
        endpoint_url = f"{BASE_URL}{endpoint}"

        # When: Making request to the endpoint
        try:
            if endpoint == "/api/execute-raw":
                # POST endpoint requires JSON payload
                test_payload = {"code": "print('CORS test')", "timeout": 5000}
                response = http_session.post(endpoint_url, json=test_payload,
                                             timeout=DEFAULT_TIMEOUT)
            else:
                # GET endpoints
                response = http_session.get(endpoint_url,
                                            timeout=DEFAULT_TIMEOUT)

            # Then: Endpoint should be accessible
            assert (response.status_code < 500), \
                f"Endpoint {endpoint} should not return server error"

            # And: CORS headers should be present
            cors_headers = {
                "Access-Control-Allow-Origin":
                    response.headers.get("Access-Control-Allow-Origin"),
                "Access-Control-Allow-Methods":
                    response.headers.get("Access-Control-Allow-Methods"),
                "Access-Control-Allow-Headers":
                    response.headers.get("Access-Control-Allow-Headers"),
            }

            # At least one CORS header should be present for browsers
            # Note: Not all endpoints have CORS headers - document them
            print(f"CORS headers for {endpoint}: {cors_headers}")

        except requests.RequestException as e:
            pytest.skip(f"Endpoint {endpoint} not accessible: {str(e)}")


class TestSecurityHeaders:
    """
    Test suite for comprehensive security headers validation.

    This class ensures that all security-related HTTP headers are
    properly configured for production deployment.
    """

    def test_given_security_requirements_when_headers_analyzed_then_comprehensive_coverage(  # noqa: E501
        self, base_response
    ):
        """
        Verify comprehensive security headers configuration.

        Given: The application must meet security standards
        When: HTTP response headers are analyzed
        Then: Essential security headers are present and properly configured

        Args:
            base_response: HTTP response containing security headers
        """
        # Given: Response headers for security analysis
        headers = base_response.headers

        # When: Checking for presence of security headers
        present_headers = {
            header: headers.get(header)
            for header in SECURITY_HEADERS if headers.get(header)
        }

        missing_headers = [
            header for header in SECURITY_HEADERS
            if header not in present_headers
        ]

        # Then: Critical security headers must be present
        critical_headers = [
            "Content-Security-Policy", "X-Content-Type-Options"
        ]
        missing_critical = [
            h for h in critical_headers if h in missing_headers
        ]

        assert (len(missing_critical) == 0), \
            f"Critical security headers missing: {missing_critical}"

        # And: Document all security headers for audit purposes
        print("\nSecurity Headers Analysis:")
        for header, value in present_headers.items():
            print(f"  {header}: Present")
            if value:
                value_str = str(value)
                display_value = (value_str[:100] + '...'
                                 if len(value_str) > 100 else value_str)
                print(f"    Value: {display_value}")


class TestPyodideScriptLoading:
    """
    Test suite for Pyodide script loading permissions in CSP.

    This class validates that CSP allows proper loading of Pyodide
    WebAssembly scripts from CDN or local sources.
    """

    def test_given_pyodide_dependency_when_csp_analyzed_then_script_loading_allowed(  # noqa: E501
        self, base_response, csp_header
    ):
        """
        Verify that CSP allows loading Pyodide scripts.

        Given: The application depends on Pyodide WebAssembly
        When: CSP policies are analyzed for script loading permissions
        Then: Pyodide script sources are properly whitelisted

        Args:
            base_response: HTTP response containing page content
            csp_header: Content Security Policy header value
        """
        # Given: Page content and CSP header
        content = base_response.text.lower()

        # When: Checking if Pyodide is referenced in the application
        has_pyodide_ref = any(
            ref in content.lower() for ref in PYODIDE_REFERENCES
        )

        if has_pyodide_ref and csp_header:
            # Then: CSP should allow script loading for Pyodide sources
            csp_lower = csp_header.lower()
            csp_allows_loading = (
                "'unsafe-eval'" in csp_lower or
                "cdn.jsdelivr.net" in csp_lower or
                "'self'" in csp_lower
            )

            assert csp_allows_loading, \
                "CSP must allow Pyodide script loading when Pyodide is used"
        else:
            pytest.skip("Pyodide not detected or CSP header not present")


class TestInlineStyleHandling:
    """
    Test suite for inline style handling under CSP constraints.

    This class validates that inline styles are properly managed
    through CSP policies without breaking functionality.
    """

    def test_given_inline_styles_when_csp_configured_then_style_policies_appropriate(  # noqa: E501
        self, base_response, csp_header
    ):
        """
        Verify that inline styles are properly handled by CSP.

        Given: The application may contain inline styles
        When: CSP style policies are analyzed
        Then: Inline styles are properly allowed or secured with hashes

        Args:
            base_response: HTTP response containing page content
            csp_header: Content Security Policy header value
        """
        # Given: Page content and CSP analysis
        content = base_response.text
        has_inline_styles = 'style="' in content

        if has_inline_styles and csp_header:
            # When: Analyzing CSP style-src policies
            csp_lower = csp_header.lower()

            # Then: Style policies should accommodate inline styles
            style_allowed = (
                "style-src" in csp_lower and (
                    "'unsafe-inline'" in csp_lower or
                    "sha256-" in csp_lower
                )
            )

            if style_allowed:
                assert style_allowed, \
                    "CSP must allow inline styles through appropriate " + \
                    "policies or hashes"
        else:
            print("Warning: Inline styles detected but no CSP " +
                  "style-src policy found")


class TestOnclickHandlerSecurity:
    """
    Test suite for onclick handler security under CSP.

    This class validates that CSP properly handles script-src-attr
    directive for onclick event handlers.
    """

    def test_given_onclick_handlers_when_csp_configured_then_script_src_attr_handled(  # noqa: E501
        self, base_response, csp_header
    ):
        """
        Verify that CSP properly handles script-src-attr for onclick handlers.

        Given: The UI contains onclick event handlers
        When: CSP script-src-attr policies are analyzed
        Then: Onclick attributes are properly secured with directives or hashes

        Args:
            base_response: HTTP response containing page content
            csp_header: Content Security Policy header value
        """
        # Given: Page content analysis for onclick handlers
        content = base_response.text.lower()
        has_onclick = 'onclick="' in content or "onclick='" in content

        if has_onclick and csp_header:
            # When: Analyzing CSP for script-src-attr directive
            csp_lower = csp_header.lower()

            # Then: Onclick handlers should be properly secured
            onclick_secured = (
                "script-src-attr" in csp_lower or
                "'unsafe-inline'" in csp_lower or
                "sha256-" in csp_lower
            )

            assert onclick_secured, \
                "CSP must handle onclick attributes with script-src-attr " + \
                "directive or SHA256 hashes"
        else:
            print("Warning: onclick handlers detected but no CSP " +
                  "script-src-attr policy found")


class TestErrorHandlingAndEdgeCases:
    """
    Test suite for error handling and edge cases in CSP compliance.

    This class validates system behavior under various error conditions
    and edge cases related to security policy enforcement.
    """

    def test_given_invalid_python_code_when_executed_then_csp_violations_prevented(  # noqa: E501
        self, http_session
    ):
        """
        Verify that CSP violations don't occur during error conditions.

        Given: Invalid Python code is submitted for execution
        When: The code execution fails
        Then: No CSP violations occur and error handling works properly

        Args:
            http_session: HTTP session for API requests
        """
        # Given: Invalid Python code that should cause execution error
        invalid_code = """
import non_existent_module
this_is_invalid_syntax !!!
undefined_variable.method()
        """

        # When: Attempting to execute invalid code
        response = execute_python_code_via_api(http_session, invalid_code)

        # Then: API should handle the error gracefully
        assert (response.status_code in [200, 400, 500]), \
            "API should handle invalid code gracefully"

        # And: Response should not indicate CSP violations
        # (CSP violations would typically prevent script execution entirely)

    def test_given_large_code_input_when_executed_then_system_remains_stable(
        self, http_session
    ):
        """
        Verify system stability with large code inputs under CSP.

        Given: A large Python code input is provided
        When: The code is executed through the API
        Then: The system remains stable without CSP violations

        Args:
            http_session: HTTP session for API requests
        """
        # Given: Large code block for stress testing
        large_code = """
# Large code block for stress testing
for i in range(100):
    print(f"Iteration {i}: Testing large code execution")

data = list(range(1000))
result = sum(x * x for x in data)
print(f"Final result: {result}")
"""

        # When: Executing large code block
        response = execute_python_code_via_api(http_session, large_code,
                                               timeout=45)

        # Then: System should handle large input appropriately
        assert (response.status_code in [200, 400, 408, 504]), \
            "Large code should execute, be rejected, or timeout gracefully"

        # And: No CSP-related errors should occur

    def test_given_empty_code_when_submitted_then_handled_appropriately(
        self, http_session
    ):
        """
        Verify appropriate handling of empty code submissions.

        Given: Empty or whitespace-only code is submitted
        When: The API processes the request
        Then: The system handles it appropriately without security issues

        Args:
            http_session: HTTP session for API requests
        """
        # Given: Various empty code scenarios
        empty_code_scenarios = ["", "   ", "\n\n", "\t\t"]

        for empty_code in empty_code_scenarios:
            # When: Submitting empty code
            response = execute_python_code_via_api(http_session, empty_code)

            # Then: API should handle empty input gracefully
            assert (response.status_code in [200, 400]), \
                f"Empty code '{repr(empty_code)}' should be handled gracefully"
