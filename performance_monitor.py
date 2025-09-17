#!/usr/bin/env python3
"""
Automated Performance Monitor

This script can be run automatically (e.g., in CI/CD) to:
1. Run performance tests
2. Save results to persistent storage
3. Detect performance regressions
4. Generate alerts if needed

Usage:
  python performance_monitor.py [--alert-threshold 15] [--tests "pattern"]
"""

import subprocess
import sys
import argparse
from pathlib import Path
import csv
import json
from datetime import datetime


def run_performance_tests(test_pattern="tests/test_performance.py"):
    """Run performance tests and capture results."""
    print(f"🚀 Running performance tests: {test_pattern}")
    
    try:
        # Run tests with pytest
        result = subprocess.run([
            "uv", "run", "pytest", test_pattern, "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=".")
        
        print(f"📊 Test execution completed with exit code: {result.returncode}")
        
        if result.returncode != 0:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
        else:
            print("✅ All performance tests passed")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def check_performance_regressions(threshold_percent=15):
    """Check for performance regressions against threshold."""
    performance_dir = Path("performance_tracking")
    csv_file = performance_dir / "performance_summary.csv"
    
    if not csv_file.exists():
        print("⚠️ No performance history found - this is the first run")
        return []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if len(data) < 2:
            print("⚠️ Insufficient data for regression analysis")
            return []
        
        # Group by test name and check trends
        test_groups = {}
        for row in data:
            test_name = row['test_name']
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(row)
        
        regressions = []
        
        for test_name, runs in test_groups.items():
            if len(runs) >= 2:
                # Compare baseline (average of first 3 runs) vs recent (last 3 runs)
                baseline_runs = runs[:3] if len(runs) >= 6 else runs[:len(runs)//2] if len(runs) >= 4 else [runs[0]]
                recent_runs = runs[-3:] if len(runs) >= 3 else [runs[-1]]
                
                baseline_times = [float(run.get('execution_time_mean', 0)) for run in baseline_runs if run.get('execution_time_mean')]
                recent_times = [float(run.get('execution_time_mean', 0)) for run in recent_runs if run.get('execution_time_mean')]
                
                if baseline_times and recent_times:
                    baseline_avg = sum(baseline_times) / len(baseline_times)
                    recent_avg = sum(recent_times) / len(recent_times)
                    
                    if baseline_avg > 0:
                        change_percent = ((recent_avg - baseline_avg) / baseline_avg) * 100
                        
                        if change_percent > threshold_percent:
                            regressions.append({
                                'test_name': test_name,
                                'change_percent': change_percent,
                                'baseline_avg': baseline_avg,
                                'recent_avg': recent_avg,
                                'baseline_runs': len(baseline_runs),
                                'recent_runs': len(recent_runs)
                            })
        
        return regressions
        
    except Exception as e:
        print(f"❌ Error checking regressions: {e}")
        return []


def generate_alert_report(regressions, threshold_percent):
    """Generate alert report for regressions."""
    if not regressions:
        print(f"✅ No performance regressions detected (threshold: {threshold_percent}%)")
        return
    
    print(f"\n{'='*80}")
    print("🚨 PERFORMANCE REGRESSION ALERT")
    print(f"{'='*80}")
    print(f"Threshold: {threshold_percent}% performance degradation")
    print(f"Detected: {len(regressions)} regression(s)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for regression in regressions:
        print(f"\n🔴 {regression['test_name'].replace('_', ' ').title()}")
        print(f"   Performance degradation: {regression['change_percent']:+.1f}%")
        print(f"   Baseline: {regression['baseline_avg']:.2f}ms (avg of {regression['baseline_runs']} runs)")
        print(f"   Recent: {regression['recent_avg']:.2f}ms (avg of {regression['recent_runs']} runs)")
    
    print(f"\n💡 Recommended actions:")
    print(f"   1. Review recent code changes")
    print(f"   2. Check system resource usage")
    print(f"   3. Run full performance test suite")
    print(f"   4. Consider performance optimization")
    print(f"\n📊 Full analysis: python performance_analysis.py")
    print(f"{'='*80}\n")


def save_monitoring_report():
    """Save a monitoring report with timestamp."""
    report_dir = Path("performance_tracking")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Get latest performance data
    csv_file = report_dir / "performance_summary.csv"
    if csv_file.exists():
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            # Get latest results
            latest_by_test = {}
            for row in data:
                test_name = row['test_name']
                if test_name not in latest_by_test or row['timestamp'] > latest_by_test[test_name]['timestamp']:
                    latest_by_test[test_name] = row
            
            report = {
                "monitor_timestamp": datetime.now().isoformat(),
                "total_tests": len(latest_by_test),
                "latest_results": latest_by_test,
                "system_health": "healthy" if all(float(r.get('success_rate', 0)) == 1.0 for r in latest_by_test.values()) else "issues_detected"
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Monitoring report saved: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Could not save monitoring report: {e}")


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Automated Performance Monitor")
    parser.add_argument("--alert-threshold", type=float, default=15.0,
                        help="Performance regression threshold percentage (default: 15)")
    parser.add_argument("--tests", type=str, default="tests/test_performance.py",
                        help="Test pattern to run (default: tests/test_performance.py)")
    parser.add_argument("--no-tests", action="store_true",
                        help="Skip running tests, only analyze existing data")
    
    args = parser.parse_args()
    
    print(f"🔍 Performance Monitor Started")
    print(f"⚙️ Regression threshold: {args.alert_threshold}%")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests unless skipped
    if not args.no_tests:
        success = run_performance_tests(args.tests)
        if not success:
            print("⚠️ Tests failed, but continuing with regression analysis...")
    
    # Check for regressions
    print(f"\n🔍 Checking for performance regressions...")
    regressions = check_performance_regressions(args.alert_threshold)
    
    # Generate alerts if needed
    generate_alert_report(regressions, args.alert_threshold)
    
    # Save monitoring report
    save_monitoring_report()
    
    # Exit with appropriate code
    if regressions:
        print(f"💥 Exiting with error code due to {len(regressions)} regression(s)")
        sys.exit(1)
    else:
        print(f"✅ Performance monitoring completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()