"""
Comprehensive Performance Testing Suite for Pyodide Express Server
"""

import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv

import pytest
import requests


@dataclass
class PerformanceConfig:
    """Configuration for performance testing."""

    BASE_URL: str = "http://localhost:3000"
    EXECUTE_ENDPOINT: str = "/api/execute-raw"

    TIMEOUTS = {
        "health_check": 10,
        "short_operation": 15,
        "medium_operation": 45,
        "long_operation": 120,
    }


@dataclass
class PerformanceMetrics:
    """Performance metrics collection with persistence."""

    test_name: str
    execution_times: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize performance tracking directory."""
        self.performance_dir = Path("performance_tracking")
        self.performance_dir.mkdir(exist_ok=True)

    def add_measurement(
        self,
        execution_time: float,
        response_time: float,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Add performance measurement."""
        if success:
            self.execution_times.append(execution_time)
            self.response_times.append(response_time)
            self.success_count += 1
        else:
            self.failure_count += 1
            if error:
                self.errors.append(error)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        stats = {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_count": self.success_count + self.failure_count,
            "success_rate": self.success_count / max(1, self.success_count + self.failure_count),
        }
        
        if self.execution_times:
            stats.update({
                "execution_time_mean": statistics.mean(self.execution_times),
                "execution_time_median": statistics.median(self.execution_times),
                "execution_time_max": max(self.execution_times),
                "execution_time_min": min(self.execution_times),
                "execution_time_stdev": statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0,
            })
        
        if self.response_times:
            stats.update({
                "response_time_mean": statistics.mean(self.response_times),
                "response_time_median": statistics.median(self.response_times),
                "response_time_max": max(self.response_times),
                "response_time_min": min(self.response_times),
            })
        
        return stats

    def save_to_history(self):
        """Save performance metrics to persistent storage."""
        # Save detailed JSON history
        json_file = self.performance_dir / f"{self.test_name}_history.json"
        
        history = []
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        
        history.append(self.get_summary_stats())
        
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save CSV summary for easy analysis
        csv_file = self.performance_dir / "performance_summary.csv"
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'test_name', 'success_count', 'failure_count', 'success_rate',
                'execution_time_mean', 'execution_time_max', 'response_time_mean'
            ])
            
            if not file_exists:
                writer.writeheader()
            
            stats = self.get_summary_stats()
            writer.writerow({
                'timestamp': stats['timestamp'],
                'test_name': stats['test_name'],
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'success_rate': f"{stats['success_rate']:.3f}",
                'execution_time_mean': f"{stats.get('execution_time_mean', 0):.2f}",
                'execution_time_max': f"{stats.get('execution_time_max', 0):.2f}",
                'response_time_mean': f"{stats.get('response_time_mean', 0):.2f}",
            })

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        json_file = self.performance_dir / f"{self.test_name}_history.json"
        
        if not json_file.exists():
            return {"trend_analysis": "No historical data available"}
        
        try:
            with open(json_file, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"trend_analysis": "Error reading historical data"}
        
        if len(history) < 2:
            return {"trend_analysis": "Insufficient data for trend analysis (need at least 2 runs)"}
        
        # Analyze recent vs historical performance
        recent_runs = history[-3:]  # Last 3 runs
        historical_runs = history[:-3] if len(history) > 3 else []
        
        trend_analysis = {
            "total_runs": len(history),
            "recent_runs_count": len(recent_runs),
            "historical_runs_count": len(historical_runs),
        }
        
        if recent_runs and historical_runs:
            # Compare recent performance to historical
            recent_exec_times = [run.get('execution_time_mean', 0) for run in recent_runs if 'execution_time_mean' in run]
            historical_exec_times = [run.get('execution_time_mean', 0) for run in historical_runs if 'execution_time_mean' in run]
            
            if recent_exec_times and historical_exec_times:
                recent_avg = statistics.mean(recent_exec_times)
                historical_avg = statistics.mean(historical_exec_times)
                performance_change = ((recent_avg - historical_avg) / historical_avg) * 100
                
                trend_analysis.update({
                    "recent_avg_execution_time": recent_avg,
                    "historical_avg_execution_time": historical_avg,
                    "performance_change_percent": performance_change,
                    "trend": "REGRESSION" if performance_change > 10 else "IMPROVEMENT" if performance_change < -10 else "STABLE"
                })
        
        # Success rate trend
        recent_success_rates = [run.get('success_rate', 0) for run in recent_runs]
        if recent_success_rates:
            avg_success_rate = statistics.mean(recent_success_rates)
            trend_analysis["recent_success_rate"] = avg_success_rate
            trend_analysis["reliability"] = "HIGH" if avg_success_rate > 0.95 else "MEDIUM" if avg_success_rate > 0.8 else "LOW"
        
        return trend_analysis

    def print_summary(self):
        """Print performance summary with trend analysis."""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SUMMARY: {self.test_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*60}")
        print(
            f"Success Rate: {self.success_count}/{self.success_count + self.failure_count}"
        )

        if self.execution_times:
            print(
                f"Execution Times - Mean: {statistics.mean(self.execution_times):.2f}ms"
            )
            print(f"                  Median: {statistics.median(self.execution_times):.2f}ms")
            print(f"                  Max: {max(self.execution_times):.2f}ms")
            print(f"                  Min: {min(self.execution_times):.2f}ms")
            if len(self.execution_times) > 1:
                print(f"                  StdDev: {statistics.stdev(self.execution_times):.2f}ms")

        if self.response_times:
            print(
                f"Response Times - Mean: {statistics.mean(self.response_times):.2f}ms"
            )

        if self.errors:
            print(f"Errors: {len(self.errors)}")
            for error in self.errors[:3]:
                print(f"  - {error}")
        
        # Show trend analysis
        trends = self.analyze_trends()
        if "trend" in trends:
            print(f"\n📈 TREND ANALYSIS:")
            print(f"  Total Runs: {trends['total_runs']}")
            print(f"  Performance Trend: {trends['trend']}")
            if trends['trend'] != 'STABLE':
                print(f"  Change: {trends['performance_change_percent']:+.1f}%")
            print(f"  Reliability: {trends.get('reliability', 'N/A')}")
        
        print(f"{'='*60}")
        print(f"📁 Data saved to: performance_tracking/{self.test_name}_history.json")
        print(f"📊 Summary: performance_tracking/performance_summary.csv")
        print(f"{'='*60}\n")


class PerformanceTestManager:
    """Performance test manager."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.session = requests.Session()

    def validate_api_contract(
        self, response: requests.Response
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate API contract."""
        try:
            data = response.json()
            required_fields = ["success", "data", "error", "meta"]
            for field in required_fields:
                if field not in data:
                    return False, {"error": f"Missing field: {field}"}

            if data["success"] and data["data"]:
                required_data_fields = ["result", "stdout", "stderr", "executionTime"]
                for field in required_data_fields:
                    if field not in data["data"]:
                        return False, {"error": f"Missing data field: {field}"}

            return True, data
        except json.JSONDecodeError:
            return False, {"error": "Invalid JSON"}

    def execute_code_with_metrics(
        self, code: str, timeout: int = None
    ) -> Tuple[bool, Dict[str, Any], float, float]:
        """Execute code and collect metrics."""
        if timeout is None:
            timeout = self.config.TIMEOUTS["medium_operation"]

        start_time = time.time()

        try:
            response = self.session.post(
                f"{self.config.BASE_URL}{self.config.EXECUTE_ENDPOINT}",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )

            response_time = (time.time() - start_time) * 1000
            is_valid, data = self.validate_api_contract(response)

            if not is_valid:
                return False, data, 0, response_time

            execution_time = (
                data.get("data", {}).get("executionTime", 0) if data.get("data") else 0
            )
            return data["success"], data, execution_time, response_time

        except requests.exceptions.Timeout:
            response_time = (time.time() - start_time) * 1000
            return (
                False,
                {"success": False, "error": f"Timeout after {timeout}s"},
                0,
                response_time,
            )
        except requests.exceptions.RequestException as e:
            response_time = (time.time() - start_time) * 1000
            return False, {"success": False, "error": str(e)}, 0, response_time

    def wait_for_server_ready(self) -> bool:
        """Wait for server readiness."""
        for _ in range(30):
            try:
                success, _, _, _ = self.execute_code_with_metrics(
                    "print('health_check')", timeout=5
                )
                if success:
                    return True
            except:
                pass
            time.sleep(1)
        return False


@pytest.fixture(scope="session")
def performance_config():
    """Performance configuration fixture."""
    return PerformanceConfig()


@pytest.fixture(scope="session")
def performance_manager(performance_config):
    """Performance manager fixture."""
    return PerformanceTestManager(performance_config)


@pytest.fixture(scope="session")
def server_ready(performance_manager):
    """Server readiness fixture."""
    if not performance_manager.wait_for_server_ready():
        pytest.skip("Server not ready")
    return True


@pytest.fixture
def performance_metrics_collector():
    """Metrics collector fixture with automatic persistence."""
    metrics = {}
    yield metrics
    
    # Print all collected metrics and save to history
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PERFORMANCE TEST SUMMARY") 
    print(f"{'='*80}")
    for name, metric in metrics.items():
        if isinstance(metric, PerformanceMetrics):
            # Save to persistent storage
            metric.save_to_history()
            # Print summary with trends
            metric.print_summary()


def generate_performance_report():
    """Generate comprehensive performance report from historical data."""
    performance_dir = Path("performance_tracking")
    if not performance_dir.exists():
        print("❌ No performance tracking data found")
        return
    
    csv_file = performance_dir / "performance_summary.csv"
    if not csv_file.exists():
        print("❌ No performance summary found")
        return
    
    print(f"\n{'='*80}")
    print("📊 HISTORICAL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("❌ No historical data available")
            return
        
        # Group by test name
        test_groups = {}
        for row in data:
            test_name = row['test_name']
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(row)
        
        for test_name, runs in test_groups.items():
            print(f"\n🔍 {test_name}:")
            print(f"  Total Runs: {len(runs)}")
            
            if len(runs) >= 2:
                # Show first and last run comparison
                first_run = runs[0]
                last_run = runs[-1]
                
                first_time = float(first_run.get('execution_time_mean', 0))
                last_time = float(last_run.get('execution_time_mean', 0))
                
                if first_time > 0:
                    change = ((last_time - first_time) / first_time) * 100
                    print(f"  Performance Change: {change:+.1f}%")
                    trend = "📈 IMPROVED" if change < -5 else "📉 DEGRADED" if change > 5 else "➡️ STABLE"
                    print(f"  Trend: {trend}")
                
                print(f"  Latest Success Rate: {float(last_run.get('success_rate', 0)):.1%}")
                print(f"  Latest Execution Time: {last_run.get('execution_time_mean', 'N/A')}ms")
        
        print(f"\n📁 Full data available in: {csv_file}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error reading performance data: {e}")
class TestCPUIntensivePerformance:
    """BDD tests for CPU-intensive performance scenarios."""

    def test_given_prime_calculation_when_executed_then_cpu_limits_discovered(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Prime number calculation with increasing complexity
        When: Executed with performance monitoring
        Then: CPU computation limits are discovered and reported
        """
        # Given: Progressive prime calculation operations
        prime_tests = [
            {"limit": 1000, "expected_max_ms": 5000},
            {"limit": 5000, "expected_max_ms": 15000},
            {"limit": 10000, "expected_max_ms": 30000},
        ]
        
        metrics = PerformanceMetrics("prime_calculation_limits")
        
        for test in prime_tests:
            # When: Execute prime calculation
            code = f"""
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Find all primes up to {test['limit']}
primes = [n for n in range(2, {test['limit']}) if is_prime(n)]
result = len(primes)
print(f"Found {{result}} primes up to {test['limit']}")
result
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["expected_max_ms"] // 1000 + 10
            )
            
            # Then: Measure and report performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ Prime calculation (limit={test['limit']}): {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ Prime calculation failed (limit={test['limit']}): {error}")
        
        performance_metrics_collector["prime_calculation_limits"] = metrics

    def test_given_matrix_operations_when_executed_then_computation_limits_found(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Matrix multiplication with increasing sizes
        When: Executed with performance monitoring
        Then: Matrix computation limits are discovered
        """
        # Given: Progressive matrix sizes
        matrix_tests = [
            {"size": 50, "expected_max_ms": 5000},
            {"size": 100, "expected_max_ms": 15000},
            {"size": 150, "expected_max_ms": 30000},
        ]
        
        metrics = PerformanceMetrics("matrix_operations_limits")
        
        for test in matrix_tests:
            # When: Execute matrix multiplication
            code = f"""
import random

# Create random matrices
size = {test['size']}
A = [[random.random() for _ in range(size)] for _ in range(size)]
B = [[random.random() for _ in range(size)] for _ in range(size)]

# Matrix multiplication
C = [[0 for _ in range(size)] for _ in range(size)]
for i in range(size):
    for j in range(size):
        for k in range(size):
            C[i][j] += A[i][k] * B[k][j]

# Calculate some stats
total = sum(sum(row) for row in C)
print(f"Matrix {{size}}x{{size}} multiplication completed, sum: {{total:.2f}}")
size
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["expected_max_ms"] // 1000 + 10
            )
            
            # Then: Measure and report performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ Matrix {test['size']}x{test['size']}: {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ Matrix {test['size']}x{test['size']} failed: {error}")
        
        performance_metrics_collector["matrix_operations_limits"] = metrics

    def test_given_recursive_algorithms_when_executed_then_recursion_limits_measured(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Recursive algorithms with increasing depth/complexity
        When: Executed with performance monitoring
        Then: Recursion and algorithm limits are discovered
        """
        # Given: Progressive recursive tests
        recursive_tests = [
            {"name": "fibonacci", "param": 30, "expected_max_ms": 5000},
            {"name": "fibonacci", "param": 35, "expected_max_ms": 15000},
            {"name": "factorial", "param": 1000, "expected_max_ms": 3000},
            {"name": "towers_hanoi", "param": 20, "expected_max_ms": 10000},
        ]
        
        metrics = PerformanceMetrics("recursive_algorithms_limits")
        
        for test in recursive_tests:
            # When: Execute recursive algorithm
            if test["name"] == "fibonacci":
                code = f"""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci({test['param']})
print(f"Fibonacci({test['param']}) = {{result}}")
result
"""
            elif test["name"] == "factorial":
                code = f"""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial({test['param']})
print(f"Factorial({test['param']}) calculated (digits: {{len(str(result))}})")
len(str(result))
"""
            elif test["name"] == "towers_hanoi":
                code = f"""
def hanoi_moves(n):
    if n == 1:
        return 1
    return 2 * hanoi_moves(n-1) + 1

result = hanoi_moves({test['param']})
print(f"Towers of Hanoi({test['param']}) requires {{result}} moves")
result
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["expected_max_ms"] // 1000 + 5
            )
            
            # Then: Measure and report performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ {test['name']}({test['param']}): {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ {test['name']}({test['param']}) failed: {error}")
        
        performance_metrics_collector["recursive_algorithms_limits"] = metrics


class TestMemoryIntensivePerformance:
    """BDD tests for memory-intensive performance scenarios."""

    def test_given_large_data_structures_when_created_then_memory_limits_discovered(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Large data structure creation with increasing sizes
        When: Executed with performance monitoring
        Then: Memory allocation limits are discovered
        """
        # Given: Progressive memory allocation tests
        memory_tests = [
            {"name": "list_int", "size": 100000, "expected_max_ms": 5000},
            {"name": "list_int", "size": 500000, "expected_max_ms": 10000},
            {"name": "list_int", "size": 1000000, "expected_max_ms": 15000},
            {"name": "dict_large", "size": 50000, "expected_max_ms": 10000},
            {"name": "nested_list", "size": 1000, "expected_max_ms": 15000},
        ]
        
        metrics = PerformanceMetrics("memory_allocation_limits")
        
        for test in memory_tests:
            # When: Execute memory allocation
            if test["name"] == "list_int":
                code = f"""
# Create large list of integers
size = {test['size']}
data = list(range(size))
total = sum(data)
print(f"Created list of {{len(data)}} integers, sum: {{total}}")
len(data)
"""
            elif test["name"] == "dict_large":
                code = f"""
# Create large dictionary
size = {test['size']}
data = {{i: f"value_{{i}}" for i in range(size)}}
print(f"Created dictionary with {{len(data)}} entries")
len(data)
"""
            elif test["name"] == "nested_list":
                code = f"""
# Create nested list structure
size = {test['size']}
data = [[i + j for j in range(size)] for i in range(size)]
total = sum(sum(row) for row in data)
print(f"Created {{len(data)}}x{{len(data[0])}} nested list, sum: {{total}}")
len(data)
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["expected_max_ms"] // 1000 + 10
            )
            
            # Then: Measure and report performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ {test['name']}({test['size']}): {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ {test['name']}({test['size']}) failed: {error}")
        
        performance_metrics_collector["memory_allocation_limits"] = metrics

    def test_given_string_processing_when_executed_then_string_limits_measured(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Large string processing operations
        When: Executed with performance monitoring
        Then: String processing limits are discovered
        """
        # Given: Progressive string processing tests
        string_tests = [
            {"operation": "concatenation", "size": 10000, "expected_max_ms": 5000},
            {"operation": "concatenation", "size": 50000, "expected_max_ms": 15000},
            {"operation": "join", "size": 100000, "expected_max_ms": 10000},
            {"operation": "regex", "size": 10000, "expected_max_ms": 8000},
        ]
        
        metrics = PerformanceMetrics("string_processing_limits")
        
        for test in string_tests:
            # When: Execute string processing
            if test["operation"] == "concatenation":
                code = f"""
# String concatenation test
size = {test['size']}
result = ""
for i in range(size):
    result += f"item_{{i}}_"
print(f"Concatenated string length: {{len(result)}}")
len(result)
"""
            elif test["operation"] == "join":
                code = f"""
# String join test
size = {test['size']}
items = [f"item_{{i}}" for i in range(size)]
result = "_".join(items)
print(f"Joined string length: {{len(result)}}")
len(result)
"""
            elif test["operation"] == "regex":
                code = f"""
import re
# Regex processing test
size = {test['size']}
text = " ".join([f"word{{i}}" for i in range(size)])
pattern = r"word\\d+"
matches = re.findall(pattern, text)
print(f"Found {{len(matches)}} regex matches in text of length {{len(text)}}")
len(matches)
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["expected_max_ms"] // 1000 + 10
            )
            
            # Then: Measure and report performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ {test['operation']}({test['size']}): {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ {test['operation']}({test['size']}) failed: {error}")
        
        performance_metrics_collector["string_processing_limits"] = metrics


class TestTimeoutAndLimitDiscovery:
    """BDD tests for discovering system timeout and execution limits."""

    def test_given_progressive_timeouts_when_executed_then_timeout_limits_discovered(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Operations with progressively longer execution times
        When: Executed with timeout monitoring
        Then: System timeout limits are discovered
        """
        # Given: Progressive timeout tests
        timeout_tests = [
            {"sleep_time": 5, "timeout": 10, "should_succeed": True},
            {"sleep_time": 15, "timeout": 20, "should_succeed": True},
            {"sleep_time": 30, "timeout": 35, "should_succeed": True},
            {"sleep_time": 60, "timeout": 65, "should_succeed": True},
            {"sleep_time": 120, "timeout": 125, "should_succeed": False},  # This might timeout
        ]
        
        metrics = PerformanceMetrics("timeout_limit_discovery")
        
        for test in timeout_tests:
            # When: Execute operation with specific sleep time
            code = f"""
import time
print(f"Starting {test['sleep_time']} second operation...")
time.sleep({test['sleep_time']})
print(f"Completed {test['sleep_time']} second operation")
"success"
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=test["timeout"]
            )
            
            # Then: Record timeout behavior
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ {test['sleep_time']}s operation: {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ {test['sleep_time']}s operation failed: {error}")
        
        performance_metrics_collector["timeout_limit_discovery"] = metrics

    def test_given_infinite_loops_when_executed_then_termination_behavior_measured(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Operations that could run indefinitely
        When: Executed with timeout controls
        Then: System termination behavior is measured
        """
        # Given: Potentially infinite operations
        infinite_tests = [
            {
                "name": "controlled_loop",
                "code": """
import time
count = 0
start_time = time.time()
while time.time() - start_time < 10:  # Run for 10 seconds max
    count += 1
    if count % 1000000 == 0:
        print(f"Iteration {count}")
print(f"Completed {count} iterations")
count
""",
                "timeout": 15,
                "should_succeed": True,
            },
            {
                "name": "cpu_bound_loop",
                "code": """
count = 0
max_iterations = 10000000
for i in range(max_iterations):
    count += i * i
    if i % 1000000 == 0:
        print(f"Progress: {i/max_iterations*100:.1f}%")
print(f"Final count: {count}")
count
""",
                "timeout": 30,
                "should_succeed": True,
            },
        ]
        
        metrics = PerformanceMetrics("termination_behavior")
        
        for test in infinite_tests:
            # When: Execute potentially long-running operation
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                test["code"], timeout=test["timeout"]
            )
            
            # Then: Measure termination behavior
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"\n✅ {test['name']}: {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"\n❌ {test['name']} terminated: {error}")
        
        performance_metrics_collector["termination_behavior"] = metrics


class TestProgressiveLoadTesting:
    """BDD tests for progressive load testing to find system limits."""

    def test_given_increasing_concurrent_load_when_executed_then_concurrency_limits_found(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Increasing numbers of concurrent requests
        When: Executed simultaneously
        Then: System concurrency limits are discovered
        """
        # Given: Progressive concurrency levels
        concurrency_levels = [2, 5, 10, 15, 20]
        
        metrics = PerformanceMetrics("concurrency_limits")
        
        for level in concurrency_levels:
            print(f"\n🔄 Testing concurrency level: {level}")
            
            # When: Execute concurrent operations
            code = """
import time
import random
# Simulate some work
data = []
for i in range(1000):
    data.append(i * random.random())
result = sum(data)
print(f"Processed {len(data)} items, sum: {result:.2f}")
result
"""
            
            def execute_single():
                return performance_manager.execute_code_with_metrics(code, timeout=15)
            
            level_success_count = 0
            level_times = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(execute_single) for _ in range(level)]
                
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        success, data, exec_time, resp_time = future.result()
                        if success:
                            level_success_count += 1
                            level_times.append(exec_time)
                            metrics.add_measurement(exec_time, resp_time, success=True)
                        else:
                            error = data.get("error", "Unknown error")
                            metrics.add_measurement(0, resp_time, success=False, error=error)
                    except Exception as e:
                        metrics.add_measurement(0, 0, success=False, error=str(e))
            
            # Then: Report concurrency results
            success_rate = level_success_count / level
            avg_time = statistics.mean(level_times) if level_times else 0
            print(f"   Success rate: {success_rate:.2%} ({level_success_count}/{level})")
            print(f"   Average time: {avg_time:.2f}ms")
            
            # Stop if success rate drops below 50%
            if success_rate < 0.5:
                print(f"   ⚠️  Concurrency limit reached at level {level}")
                break
        
        performance_metrics_collector["concurrency_limits"] = metrics

    def test_given_increasing_data_sizes_when_processed_then_data_limits_discovered(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Data processing tasks with increasing data sizes
        When: Executed with performance monitoring
        Then: Data processing limits are discovered
        """
        # Given: Progressive data sizes
        data_sizes = [1000, 10000, 50000, 100000, 500000]
        
        metrics = PerformanceMetrics("data_processing_limits")
        
        for size in data_sizes:
            print(f"\n📊 Testing data size: {size:,}")
            
            # When: Process increasingly large datasets
            code = f"""
import random
import statistics

# Generate large dataset
size = {size}
data = [random.random() * 1000 for _ in range(size)]

# Perform various operations
mean_val = statistics.mean(data)
sorted_data = sorted(data)
filtered_data = [x for x in data if x > mean_val]

# Calculate results
result = {{
    'size': len(data),
    'mean': mean_val,
    'sorted_first': sorted_data[0],
    'sorted_last': sorted_data[-1],
    'filtered_count': len(filtered_data)
}}

print(f"Processed {{size:,}} items: mean={{mean_val:.2f}}, filtered={{len(filtered_data):,}}")
result['size']
"""
            
            success, data, exec_time, resp_time = performance_manager.execute_code_with_metrics(
                code, timeout=60
            )
            
            # Then: Measure data processing performance
            if success:
                metrics.add_measurement(exec_time, resp_time, success=True)
                print(f"   ✅ Processing time: {exec_time:.2f}ms")
            else:
                error = data.get("error", "Unknown error")
                metrics.add_measurement(0, resp_time, success=False, error=error)
                print(f"   ❌ Processing failed: {error}")
                # Stop if processing fails
                break
        
        performance_metrics_collector["data_processing_limits"] = metrics


class TestConcurrencyPerformance:
    """BDD tests for concurrency performance."""

    def test_given_concurrent_requests_when_executed_then_system_stable(
        self,
        server_ready,
        performance_manager: PerformanceTestManager,
        performance_metrics_collector,
    ):
        """
        Given: Multiple concurrent requests
        When: Executed simultaneously
        Then: System handles load gracefully
        """
        # Given: Simple concurrent operation
        code = (
            "result = sum(i*i for i in range(10)); print(f'Result: {result}'); result"
        )

        metrics = PerformanceMetrics("concurrent_test")

        # When: Execute concurrently
        def execute_single():
            return performance_manager.execute_code_with_metrics(code, timeout=10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_single) for _ in range(3)]

            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    success, data, exec_time, resp_time = future.result()
                    if success:
                        metrics.add_measurement(exec_time, resp_time, success=True)
                    else:
                        error = data.get("error", "Unknown error")
                        metrics.add_measurement(
                            0, resp_time, success=False, error=error
                        )
                except Exception as e:
                    metrics.add_measurement(0, 0, success=False, error=str(e))

        # Then: Good success rate
        success_rate = metrics.success_count / (
            metrics.success_count + metrics.failure_count
        )
        assert success_rate >= 0.6, f"Success rate {success_rate:.2%} should be >= 60%"
        performance_metrics_collector["concurrent_test"] = metrics


if __name__ == "__main__":
    """Run performance tests directly."""
    import sys

    config = PerformanceConfig()
    manager = PerformanceTestManager(config)

    if not manager.wait_for_server_ready():
        print("❌ Server not ready")
        sys.exit(1)

    print("✅ Server ready - running tests")
    pytest.main([__file__, "-v", "-s"])
