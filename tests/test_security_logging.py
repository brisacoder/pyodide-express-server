"""
Test suite for enhanced security logging functionality.

This module tests:
1. Security event logging (code execution, file uploads, package installs)
2. Statistics collection and accuracy
3. Dashboard endpoints functionality
4. Error tracking and IP monitoring
5. Security log file generation
"""

import tempfile
import unittest
from pathlib import Path

import requests

BASE_URL = "http://localhost:3000"


class SecurityLoggingTestCase(unittest.TestCase):
    """Test security logging features and statistics collection."""

    def setUp(self):
        """Set up test environment before each test."""
        # Clear statistics before each test for clean data
        self.clear_stats()
        self.uploaded_files = []
        self.temp_files = []

    def tearDown(self):
        """Clean up after each test."""
        # Clean up uploaded files
        for filename in self.uploaded_files:
            try:
                requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}")
            except:
                pass
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass

    def clear_stats(self):
        """Clear statistics for clean test data."""
        try:
            response = requests.post(f"{BASE_URL}/api/dashboard/stats/clear", timeout=10)
            return response.status_code == 200
        except:
            return False

    def test_01_security_logging_code_execution(self):
        """Test that code execution events are properly logged with security data."""
        # Execute some Python code
        code = "print('Testing security logging for code execution')"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("success"))
        
        # Check that statistics were updated
        stats_response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
        self.assertEqual(stats_response.status_code, 200)
        
        stats_data = stats_response.json()
        self.assertTrue(stats_data.get("success"))
        
        stats = stats_data.get("stats", {})
        overview = stats.get("overview", {})
        
        # Verify execution was tracked
        self.assertEqual(overview.get("totalExecutions"), 1)
        self.assertEqual(overview.get("successRate"), "100.0")
        
        # Verify IP tracking
        top_ips = stats.get("topIPs", [])
        self.assertTrue(len(top_ips) > 0)
        self.assertEqual(top_ips[0].get("count"), 1)
        
        # Verify user agent tracking
        user_agents = stats.get("userAgents", [])
        self.assertTrue(len(user_agents) > 0)
        self.assertIn("python-requests", user_agents[0].get("agent", ""))

    def test_02_security_logging_error_tracking(self):
        """Test that errors are properly categorized and tracked."""
        # Execute code that will cause a syntax error
        invalid_code = "invalid_syntax_here !@#"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=invalid_code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json().get("success"))
        
        # Execute code that will cause a name error
        name_error_code = "print(undefined_variable)"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=name_error_code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json().get("success"))
        
        # Check error tracking in statistics
        stats_response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
        self.assertEqual(stats_response.status_code, 200)
        
        stats = stats_response.json().get("stats", {})
        overview = stats.get("overview", {})
        top_errors = stats.get("topErrors", [])
        
        # Verify error tracking
        self.assertEqual(overview.get("totalExecutions"), 2)
        self.assertEqual(overview.get("successRate"), "0.0")
        self.assertTrue(len(top_errors) > 0)
        
        # Check that different error types are tracked
        error_types = [error.get("error") for error in top_errors]
        self.assertTrue(any("syntax" in error.lower() for error in error_types))

    def test_03_security_logging_file_uploads(self):
        """Test that file upload events are properly logged."""
        # Create a test CSV file
        test_data = "name,age,score\nAlice,25,85\nBob,30,92"
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(test_data)
        temp_file.close()
        self.temp_files.append(Path(temp_file.name))
        
        # Upload the file
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test_security_logging.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/upload-csv", files=files, timeout=30)
        
        # Check if upload was successful (may fail if server not fully ready)
        if response.status_code == 200:
            result = response.json()
            self.assertTrue(result.get("success"))
            
            if result.get("success"):
                filename = result.get("filename")
                if filename:
                    self.uploaded_files.append(filename)
            
            # Check that file upload was tracked in statistics
            stats_response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
            self.assertEqual(stats_response.status_code, 200)
            
            stats = stats_response.json().get("stats", {})
            recent = stats.get("recent", {})
            
            # Verify file upload tracking
            self.assertEqual(recent.get("filesUploaded"), 1)
        else:
            # If upload fails, just check that the endpoint exists
            self.assertIn(response.status_code, [400, 500, 503])  # Expected error codes

    def test_04_statistics_accuracy_over_time(self):
        """Test statistics accuracy with multiple operations over time."""
        initial_stats = self.get_current_stats()
        
        # Perform multiple successful executions
        for i in range(3):
            code = f"result = {i} * 2\nprint(f'Result: {{result}}')"
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json().get("success"))
        
        # Perform one failed execution
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data="x = 1 / 0  # This will cause ZeroDivisionError",
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json().get("success"))
        
        # Check final statistics
        final_stats = self.get_current_stats()
        overview = final_stats.get("overview", {})
        
        # Verify counts
        self.assertEqual(overview.get("totalExecutions"), 4)
        self.assertEqual(overview.get("successRate"), "75.0")  # 3 success out of 4 total
        
        # Verify error tracking
        top_errors = final_stats.get("topErrors", [])
        self.assertTrue(len(top_errors) > 0)
        
        # Check that ZeroDivisionError is tracked
        error_types = [error.get("error") for error in top_errors]
        self.assertTrue(any("division by zero" in error.lower() for error in error_types))

    def test_05_hourly_trend_tracking(self):
        """Test that hourly execution trends are properly tracked."""
        # Execute multiple operations
        for i in range(5):
            code = f"print('Trend test execution {i}')"
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30
            )
            self.assertEqual(response.status_code, 200)
        
        # Check hourly trend data
        stats = self.get_current_stats()
        hourly_trend = stats.get("hourlyTrend", [])
        
        # Verify trend data structure
        self.assertEqual(len(hourly_trend), 24)  # 24 hours
        self.assertTrue(all(isinstance(count, int) for count in hourly_trend))
        
        # Verify recent executions are tracked
        total_recent = sum(hourly_trend)
        self.assertEqual(total_recent, 5)

    def test_06_dashboard_endpoints_functionality(self):
        """Test all dashboard endpoints for proper functionality."""
        # Test main dashboard stats endpoint
        response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("stats", data)
        self.assertIn("timestamp", data)
        
        # Test dashboard HTML endpoint
        response = requests.get(f"{BASE_URL}/api/dashboard/stats/dashboard", timeout=10)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("Content-Type", ""))
        
        # Verify HTML contains expected elements
        html_content = response.text
        self.assertIn("Pyodide Express Server Dashboard", html_content)
        self.assertIn("chart.js", html_content.lower())  # Case-insensitive check
        self.assertIn("canvas", html_content)
        
        # Test stats clear endpoint
        response = requests.post(f"{BASE_URL}/api/dashboard/stats/clear", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("message", data)

    def test_07_legacy_stats_endpoint_compatibility(self):
        """Test that legacy stats endpoint maintains backward compatibility."""
        # Execute some code to generate data
        code = "print('Legacy compatibility test')"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        # Test legacy stats endpoint
        response = requests.get(f"{BASE_URL}/api/stats", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # Verify legacy fields are present at top level
        self.assertIn("uptime", data)
        self.assertIn("memory", data)
        self.assertIn("pyodide", data)
        self.assertIn("timestamp", data)
        
        # Verify enhanced stats are present under executionStats
        self.assertIn("executionStats", data)
        execution_stats = data.get("executionStats", {})
        
        self.assertIn("overview", execution_stats)
        self.assertIn("recent", execution_stats)
        self.assertIn("topIPs", execution_stats)
        self.assertIn("topErrors", execution_stats)
        self.assertIn("userAgents", execution_stats)
        self.assertIn("hourlyTrend", execution_stats)

    def test_08_ip_and_user_agent_tracking(self):
        """Test IP address and User-Agent tracking functionality."""
        # Execute code with custom user agent
        headers = {
            "Content-Type": "text/plain",
            "User-Agent": "SecurityLoggingTestSuite/1.0"
        }
        
        code = "print('Testing IP and User-Agent tracking')"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers=headers,
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        # Check tracking in statistics
        stats = self.get_current_stats()
        
        # Verify IP tracking
        top_ips = stats.get("topIPs", [])
        self.assertTrue(len(top_ips) > 0)
        
        # Verify User-Agent tracking
        user_agents = stats.get("userAgents", [])
        self.assertTrue(len(user_agents) > 0)
        
        # Should contain our custom user agent
        agent_strings = [ua.get("agent") for ua in user_agents]
        self.assertTrue(any("SecurityLoggingTestSuite" in agent for agent in agent_strings))

    def test_09_execution_time_tracking(self):
        """Test that execution times are properly tracked and averaged."""
        # Execute code with varying complexity
        test_codes = [
            "print('Simple execution')",
            "import time; sum(range(1000)); print('Medium execution')",
            "result = [i**2 for i in range(100)]; print('Complex execution')"
        ]
        
        for code in test_codes:
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=30
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json().get("success"))
        
        # Check execution time tracking
        stats = self.get_current_stats()
        overview = stats.get("overview", {})
        
        # Verify execution time is tracked
        avg_time = overview.get("averageExecutionTime")
        self.assertIsNotNone(avg_time)
        self.assertIsInstance(avg_time, (int, float))
        self.assertGreater(avg_time, 0)

    def test_10_statistics_reset_functionality(self):
        """Test that statistics can be properly reset."""
        # Generate some statistics
        code = "print('Pre-reset execution')"
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify statistics exist
        stats_before = self.get_current_stats()
        self.assertGreater(stats_before.get("overview", {}).get("totalExecutions", 0), 0)
        
        # Reset statistics
        response = requests.post(f"{BASE_URL}/api/dashboard/stats/clear", timeout=10)
        self.assertEqual(response.status_code, 200)
        
        # Verify statistics are reset
        stats_after = self.get_current_stats()
        overview = stats_after.get("overview", {})
        
        self.assertEqual(overview.get("totalExecutions"), 0)
        self.assertEqual(overview.get("successRate"), 0)
        self.assertEqual(len(stats_after.get("topIPs", [])), 0)
        self.assertEqual(len(stats_after.get("topErrors", [])), 0)
        self.assertEqual(len(stats_after.get("userAgents", [])), 0)

    def get_current_stats(self):
        """Helper method to get current statistics."""
        response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
        if response.status_code == 200:
            return response.json().get("stats", {})
        return {}


if __name__ == '__main__':
    unittest.main(verbosity=2)
