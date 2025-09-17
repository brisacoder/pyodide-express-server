# Performance Tracking System

## 🎯 Overview

This system provides **automatic performance tracking over time** for the Pyodide Express Server. Every time you run performance tests, the results are saved and analyzed for trends, regressions, and system health.

## 📊 Key Features

- ✅ **Automatic Data Persistence**: Every test run saves detailed metrics to JSON and CSV files
- 📈 **Trend Analysis**: Compare current performance to historical baselines  
- 🚨 **Regression Detection**: Automatic alerts when performance degrades
- 📄 **Comprehensive Reporting**: Detailed statistics with mean, median, max, min, and standard deviation
- 🕐 **Historical Tracking**: Full timeline of performance changes
- 🎯 **No Manual Intervention**: Everything runs automatically

## 🚀 Quick Start

### Run Performance Tests
```bash
# Run all performance tests (automatically saves results)
uv run pytest tests/test_performance.py -v -s

# Run specific test category
uv run pytest tests/test_performance.py::TestCPUIntensivePerformance -v -s
```

### View Performance Trends
```bash
# Full trend analysis
python performance_analysis.py

# Just latest results
python performance_analysis.py --latest
```

### Automated Monitoring
```bash
# Run tests and check for regressions (ideal for CI/CD)
python performance_monitor.py

# Just analyze existing data
python performance_monitor.py --no-tests

# Custom regression threshold
python performance_monitor.py --alert-threshold 10
```

## 📁 Data Storage

All performance data is stored in the `performance_tracking/` directory:

```
performance_tracking/
├── performance_summary.csv           # Summary of all test runs
├── {test_name}_history.json         # Detailed history per test
└── monitor_report_YYYYMMDD_HHMMSS.json  # Monitoring reports
```

### CSV Format
```csv
timestamp,test_name,success_count,failure_count,success_rate,execution_time_mean,execution_time_max,response_time_mean
2025-09-16T00:03:26.485558,prime_calculation_limits,3,0,1.000,34.33,44.00,46.38
```

### JSON History Example
```json
[
  {
    "test_name": "prime_calculation_limits",
    "timestamp": "2025-09-16T00:03:26.485558",
    "success_count": 3,
    "failure_count": 0,
    "success_rate": 1.0,
    "execution_time_mean": 34.33,
    "execution_time_median": 35.0,
    "execution_time_max": 44.0,
    "execution_time_min": 24.0,
    "execution_time_stdev": 10.02
  }
]
```

## 🧪 Available Performance Tests

### CPU-Intensive Tests
- **Prime Calculation**: Tests computational limits with prime number generation
- **Matrix Operations**: Tests matrix multiplication performance scaling  
- **Recursive Algorithms**: Tests recursion limits with fibonacci, factorial, hanoi

### Memory Tests
- **Large Data Structures**: Tests memory allocation limits with lists, dicts, arrays
- **String Processing**: Tests string concatenation, join operations, regex processing

### System Limits
- **Timeout Discovery**: Finds maximum execution time limits (discovered: ~30 seconds)
- **Concurrency Testing**: Tests parallel execution capabilities
- **Progressive Load**: Scales load until system limits are reached

## 📈 Trend Analysis Features

### Performance Trends
- **IMPROVEMENT**: Performance improved by >10%
- **STABLE**: Performance within ±10% 
- **DEGRADED**: Performance degraded by >10%
- **REGRESSION**: Performance degraded by >15% (triggers alert)

### Reliability Scoring
- **🟢 EXCELLENT**: >95% success rate
- **🟡 GOOD**: 80-95% success rate  
- **🔴 NEEDS ATTENTION**: <80% success rate

## 🚨 Automated Alerts

The system automatically detects:
- Performance regressions above threshold (default: 15%)
- Failing tests
- System health issues
- Reliability problems

Example alert output:
```
🚨 PERFORMANCE REGRESSION ALERT
═══════════════════════════════════════════════════════════════════════════════
Threshold: 15.0% performance degradation
Detected: 1 regression(s)

🔴 Prime Calculation Limits
   Performance degradation: +18.5%
   Baseline: 34.33ms (avg of 2 runs)
   Recent: 40.67ms (avg of 3 runs)
```

## 🔧 Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Run Performance Tests
  run: python performance_monitor.py --alert-threshold 15

- name: Upload Performance Data
  uses: actions/upload-artifact@v3
  with:
    name: performance-tracking
    path: performance_tracking/
```

## 💡 Benefits

✅ **No Manual Tracking**: Automatically captures every test run  
✅ **Historical Context**: See performance trends over weeks/months  
✅ **Early Warning**: Detect regressions before they become problems  
✅ **Data-Driven Decisions**: Make optimization decisions based on real data  
✅ **CI/CD Integration**: Fail builds when performance degrades  
✅ **Zero Configuration**: Works out of the box

## 🎯 System Limits Discovered

From automated testing, we've discovered these system limits:

- **Maximum Execution Time**: ~30 seconds (hard timeout)
- **Optimal Concurrency**: 2-5 concurrent operations  
- **Memory Handling**: Efficient up to 1M+ elements
- **CPU Performance**: Matrix operations scale O(n³) as expected
- **String Operations**: Join >> concatenation for large data

## 📋 Example Output

```
📊 AUTOMATED PERFORMANCE TREND ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
📈 Total test runs recorded: 5
🕐 Date range: 2025-09-15 to 2025-09-16

🧪 Test categories tracked: 3

────────────────────────────────────────────────────────────
🔍 Prime Calculation Limits  
────────────────────────────────────────────────────────────
  📊 Total Runs: 3
  📅 Latest Run: 2025-09-16 00:03:47
  📈 Performance Change: +3.9%
  ➡️ Trend: STABLE
  ✅ Latest Success Rate: 100.0%
  🎯 Reliability: 🟢 EXCELLENT
  ⏱️ Latest Execution Time: 35.67ms
```

This system gives you **complete visibility** into your performance characteristics over time, with **zero manual effort** required! 🚀