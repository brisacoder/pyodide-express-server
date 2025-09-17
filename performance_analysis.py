#!/usr/bin/env python3
"""
Performance Analysis Utility

Run this script anytime to see performance trends without running tests.
Usage: python performance_analysis.py
"""

import csv
import json
from pathlib import Path
from datetime import datetime


def analyze_performance_trends():
    """Analyze performance trends from historical data."""
    performance_dir = Path("performance_tracking")
    
    if not performance_dir.exists():
        print("❌ No performance tracking data found")
        print("💡 Run the performance tests first: uv run pytest tests/test_performance.py")
        return
    
    csv_file = performance_dir / "performance_summary.csv"
    if not csv_file.exists():
        print("❌ No performance summary found")
        return
    
    print(f"\n{'='*80}")
    print("📊 AUTOMATED PERFORMANCE TREND ANALYSIS")
    print(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("❌ No historical data available")
            return
        
        print(f"📈 Total test runs recorded: {len(data)}")
        print(f"🕐 Date range: {data[0]['timestamp'][:10]} to {data[-1]['timestamp'][:10]}")
        
        # Group by test name
        test_groups = {}
        for row in data:
            test_name = row['test_name']
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(row)
        
        print(f"\n🧪 Test categories tracked: {len(test_groups)}")
        
        for test_name, runs in test_groups.items():
            print(f"\n{'─'*60}")
            print(f"🔍 {test_name.replace('_', ' ').title()}")
            print(f"{'─'*60}")
            print(f"  📊 Total Runs: {len(runs)}")
            print(f"  📅 Latest Run: {runs[-1]['timestamp'][:19].replace('T', ' ')}")
            
            if len(runs) >= 2:
                # Performance trend analysis
                first_run = runs[0]
                last_run = runs[-1]
                
                first_time = float(first_run.get('execution_time_mean', 0))
                last_time = float(last_run.get('execution_time_mean', 0))
                
                if first_time > 0:
                    change = ((last_time - first_time) / first_time) * 100
                    print(f"  📈 Performance Change: {change:+.1f}%")
                    
                    if change < -10:
                        trend_emoji = "🚀"
                        trend_text = "SIGNIFICANT IMPROVEMENT"
                    elif change < -5:
                        trend_emoji = "📈"
                        trend_text = "IMPROVED"
                    elif change > 10:
                        trend_emoji = "🐌"
                        trend_text = "PERFORMANCE REGRESSION"
                    elif change > 5:
                        trend_emoji = "📉"
                        trend_text = "DEGRADED"
                    else:
                        trend_emoji = "➡️"
                        trend_text = "STABLE"
                    
                    print(f"  {trend_emoji} Trend: {trend_text}")
                
                # Reliability analysis
                recent_success_rate = float(last_run.get('success_rate', 0))
                print(f"  ✅ Latest Success Rate: {recent_success_rate:.1%}")
                
                if recent_success_rate >= 0.95:
                    reliability = "🟢 EXCELLENT"
                elif recent_success_rate >= 0.8:
                    reliability = "🟡 GOOD"
                else:
                    reliability = "🔴 NEEDS ATTENTION"
                
                print(f"  🎯 Reliability: {reliability}")
                print(f"  ⏱️ Latest Execution Time: {last_run.get('execution_time_mean', 'N/A')}ms")
                
                # Show execution time trend over last 5 runs
                recent_runs = runs[-5:] if len(runs) >= 5 else runs
                exec_times = [float(run.get('execution_time_mean', 0)) for run in recent_runs if run.get('execution_time_mean')]
                
                if len(exec_times) >= 3:
                    trend_direction = "📈" if exec_times[-1] > exec_times[0] else "📉" if exec_times[-1] < exec_times[0] else "➡️"
                    print(f"  📊 Recent Trend: {trend_direction} {exec_times[0]:.1f}ms → {exec_times[-1]:.1f}ms")
            
            else:
                print("  💡 Need more runs for trend analysis")
        
        # Overall system health
        print(f"\n{'='*60}")
        print("🏥 SYSTEM HEALTH SUMMARY")
        print(f"{'='*60}")
        
        latest_data = {}
        for test_name, runs in test_groups.items():
            latest_data[test_name] = runs[-1]
        
        # Check for any failing tests
        failing_tests = [name for name, data in latest_data.items() if float(data.get('success_rate', 1)) < 1.0]
        if failing_tests:
            print(f"⚠️  Tests with failures: {', '.join(failing_tests)}")
        else:
            print("✅ All tests passing in latest runs")
        
        # Performance regression check
        regression_tests = []
        for test_name, runs in test_groups.items():
            if len(runs) >= 2:
                first_time = float(runs[0].get('execution_time_mean', 0))
                last_time = float(runs[-1].get('execution_time_mean', 0))
                if first_time > 0 and ((last_time - first_time) / first_time) * 100 > 15:
                    regression_tests.append(test_name)
        
        if regression_tests:
            print(f"🐌 Performance regressions detected: {', '.join(regression_tests)}")
        else:
            print("🚀 No significant performance regressions")
        
        print(f"\n📁 Detailed data available in: {performance_dir}")
        print(f"📊 CSV Summary: {csv_file}")
        print(f"💡 Run tests again: uv run pytest tests/test_performance.py")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error reading performance data: {e}")


def show_latest_results():
    """Show just the latest test results."""
    performance_dir = Path("performance_tracking")
    csv_file = performance_dir / "performance_summary.csv"
    
    if not csv_file.exists():
        print("❌ No performance data found")
        return
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("❌ No data available")
            return
        
        # Get latest run for each test
        latest_by_test = {}
        for row in data:
            test_name = row['test_name']
            if test_name not in latest_by_test or row['timestamp'] > latest_by_test[test_name]['timestamp']:
                latest_by_test[test_name] = row
        
        print(f"\n📊 LATEST PERFORMANCE RESULTS")
        print(f"{'='*60}")
        
        for test_name, result in latest_by_test.items():
            print(f"\n🧪 {test_name.replace('_', ' ').title()}")
            print(f"  ⏱️ Execution Time: {result.get('execution_time_mean', 'N/A')}ms")
            print(f"  ✅ Success Rate: {float(result.get('success_rate', 0)):.1%}")
            print(f"  📅 Run Time: {result['timestamp'][:19].replace('T', ' ')}")
        
        print(f"\n{'='*60}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--latest":
        show_latest_results()
    else:
        analyze_performance_trends()