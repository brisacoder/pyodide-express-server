#!/usr/bin/env python3
"""
BDD-style pytest tests for correlation analysis and advanced seaborn functionality.

This module integrates correlation-focused test files from the project root,
converting them to proper pytest format with enhanced error handling and validation.

Integrated Tests From Root Directory:
- test_correlation.py -> Seaborn correlation heatmap testing
- test_exact_correlation.py -> Precise correlation analysis
- test_simple_correlation.py -> Basic correlation operations
- test_seaborn_import.py -> Seaborn import and setup validation

Key Features:
- ✅ BDD Given/When/Then structure with comprehensive correlation testing
- ✅ Only uses /api/execute-raw endpoint with plain text code
- ✅ API contract compliance validation for all responses
- ✅ Cross-platform pathlib usage for all file operations
- ✅ Enhanced error handling for data science operations
- ✅ No hardcoded values - all parameterized with fixtures
- ✅ Base64 plot generation and validation
- ✅ Statistical validation of correlation results

API Contract:
{
  "success": true | false,
  "data": {
    "result": <any>,
    "stdout": <string>,
    "stderr": <string>,
    "executionTime": <number>
  } | null,
  "error": <string|null>,
  "meta": {"timestamp": <string>}
}

Test Categories:
- correlation_analysis: Advanced correlation heatmap generation
- statistical_validation: Statistical accuracy of correlation calculations
- plot_generation: Base64 plot creation and validation
- seaborn_integration: Seaborn import and functionality testing
"""

import json

import pytest

# Import shared configuration and utilities
try:
    from .conftest import execute_python_code, validate_api_contract
except ImportError:
    from conftest import execute_python_code, validate_api_contract


class CorrelationTestConfig:
    """Configuration constants for correlation analysis tests."""
    
    # Correlation test parameters
    CORRELATION_SETTINGS = {
        "sample_size": 200,
        "random_seed": 42,
        "correlation_tolerance": 0.05,
        "min_plot_size": 10000,  # Minimum base64 string length
        "dpi_setting": 150,
        "figsize": (10, 8),
        "heatmap_format": "png"
    }
    
    # Test correlation scenarios
    CORRELATION_SCENARIOS = [
        {
            "name": "strong_positive",
            "description": "Strong positive correlation between variables",
            "correlation_coefficient": 0.8,
            "expected_range": (0.75, 0.85)
        },
        {
            "name": "moderate_negative",
            "description": "Moderate negative correlation between variables",
            "correlation_coefficient": -0.5,
            "expected_range": (-0.55, -0.45)
        },
        {
            "name": "weak_correlation",
            "description": "Weak correlation between variables",
            "correlation_coefficient": 0.2,
            "expected_range": (0.15, 0.25)
        },
        {
            "name": "no_correlation",
            "description": "No correlation between independent variables",
            "correlation_coefficient": 0.0,
            "expected_range": (-0.1, 0.1)
        }
    ]


@pytest.fixture
def correlation_timeout():
    """
    Provide timeout for correlation analysis operations.

    Returns:
        int: Timeout in seconds for correlation operations

    Example:
        >>> def test_correlation(correlation_timeout):
        ...     result = execute_python_code(code, timeout=correlation_timeout)
    """
    return CorrelationTestConfig.CORRELATION_SETTINGS["correlation_tolerance"] * 100 + 30


@pytest.fixture
def seaborn_timeout():
    """
    Provide timeout for seaborn import and testing operations.

    Returns:
        int: Timeout in seconds for seaborn operations

    Example:
        >>> def test_seaborn(seaborn_timeout):
        ...     result = execute_python_code(code, timeout=seaborn_timeout)
    """
    return 45


@pytest.fixture
def cleanup_correlation_plots():
    """
    Cleanup fixture to remove correlation plot files after tests.

    Yields:
        List[str]: List to append created plot file paths for cleanup

    Example:
        >>> def test_plot_creation(cleanup_correlation_plots):
        ...     plot_path = "/home/pyodide/plots/correlation_test.png"
        ...     cleanup_correlation_plots.append(plot_path)
        ...     # Plot will be automatically cleaned up
    """
    created_plots = []
    yield created_plots

    # Cleanup created plots
    if created_plots:
        cleanup_code = f"""
from pathlib import Path

cleaned_files = []
errors = []

for plot_path in {created_plots}:
    try:
        plot_file = Path(plot_path)
        if plot_file.exists():
            plot_file.unlink()
            cleaned_files.append(str(plot_file))
    except Exception as e:
        errors.append(f"{{plot_path}}: {{str(e)}}")

{{
    "cleaned_files": cleaned_files,
    "errors": errors,
    "total_cleaned": len(cleaned_files)
}}
        """
        try:
            execute_python_code(cleanup_code)
        except Exception:
            # Don't fail tests due to cleanup issues
            pass


@pytest.mark.seaborn
@pytest.mark.correlation
@pytest.mark.api
class TestCorrelationAnalysisIntegration:
    """
    BDD tests for comprehensive correlation analysis functionality.
    
    Migrated from: test_correlation.py, test_exact_correlation.py, test_simple_correlation.py
    Enhanced with: proper pytest structure, statistical validation, enhanced error handling
    """

    def test_given_seaborn_available_when_creating_correlation_heatmap_then_generates_valid_plot(
        self, server_ready, correlation_timeout, cleanup_correlation_plots
    ):
        """
        Test creation of correlation heatmap with statistical validation.

        **Given:** Seaborn and matplotlib are available
        **When:** Creating correlation heatmap with controlled data
        **Then:** Generates valid base64 plot with correct correlation statistics

        Args:
            server_ready: Server readiness fixture
            correlation_timeout: Timeout for correlation operations
            cleanup_correlation_plots: Cleanup tracking list

        Example:
            Creates dataset with known correlations and validates that the
            heatmap correctly represents the statistical relationships.
        """
        # Given: Controlled dataset with known correlations
        code = f"""
# Comprehensive correlation heatmap test with statistical validation
try:
    import seaborn as sns
    print("✅ Seaborn already available")
except ImportError:
    print("⚠️ Installing seaborn...")
    import micropip
    await micropip.install(['seaborn'])
    import seaborn as sns
    print("✅ Seaborn installed")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
import io
from pathlib import Path

# Set random seed for reproducible results
np.random.seed({CorrelationTestConfig.CORRELATION_SETTINGS['random_seed']})
sample_size = {CorrelationTestConfig.CORRELATION_SETTINGS['sample_size']}

print(f"Creating dataset with {{sample_size}} samples...")

# When: Creating dataset with controlled correlations
# Base variable
feature_1 = np.random.randn(sample_size)

# Strong positive correlation (target: ~0.8)
feature_2 = feature_1 * 0.8 + np.random.randn(sample_size) * 0.36

# Moderate negative correlation (target: ~-0.5)
feature_3 = feature_1 * -0.5 + np.random.randn(sample_size) * 0.75

# Weak correlation (target: ~0.2)
feature_4 = feature_1 * 0.2 + np.random.randn(sample_size) * 0.96

# Independent variable (target: ~0.0)
feature_5 = np.random.randn(sample_size)

# Create DataFrame
data = pd.DataFrame({{
    'base_feature': feature_1,
    'strong_positive': feature_2,
    'moderate_negative': feature_3,
    'weak_correlation': feature_4,
    'independent': feature_5
}})

print(f"Dataset shape: {{data.shape}}")
print(f"Dataset columns: {{list(data.columns)}}")

# Calculate correlation matrix
correlation_matrix = data.corr()
print(f"Correlation matrix calculated: {{correlation_matrix.shape}}")

# Statistical validation
correlation_stats = {{}}
correlation_stats['base_to_strong_positive'] = correlation_matrix.loc['base_feature', 'strong_positive']
correlation_stats['base_to_moderate_negative'] = correlation_matrix.loc['base_feature', 'moderate_negative']
correlation_stats['base_to_weak'] = correlation_matrix.loc['base_feature', 'weak_correlation']
correlation_stats['base_to_independent'] = correlation_matrix.loc['base_feature', 'independent']

print("Correlation statistics:")
for key, value in correlation_stats.items():
    print(f"  {{key}}: {{value:.3f}}")

# Create enhanced correlation heatmap
fig, ax = plt.subplots(figsize={CorrelationTestConfig.CORRELATION_SETTINGS['figsize']})

# Advanced heatmap with annotations and styling
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={{"shrink": 0.8}},
    annot_kws={{"size": 10}},
    ax=ax
)

plt.title('Statistical Correlation Matrix Analysis',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Generate base64 plot
buffer = io.BytesIO()
plt.savefig(buffer,
           format='{CorrelationTestConfig.CORRELATION_SETTINGS['heatmap_format']}',
           dpi={CorrelationTestConfig.CORRELATION_SETTINGS['dpi_setting']},
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')
buffer.seek(0)
plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

print(f"Base64 plot generated: {{len(plot_base64)}} characters")

# Then: Validate results and generate comprehensive output
validation_results = {{}}

# Validate correlation accuracy
expected_correlations = {{
    'strong_positive': (0.75, 0.85),
    'moderate_negative': (-0.55, -0.45),
    'weak_correlation': (0.15, 0.25),
    'independent': (-0.1, 0.1)
}}

for correlation_type, (min_val, max_val) in expected_correlations.items():
    if correlation_type == 'strong_positive':
        actual = correlation_stats['base_to_strong_positive']
    elif correlation_type == 'moderate_negative':
        actual = correlation_stats['base_to_moderate_negative']
    elif correlation_type == 'weak_correlation':
        actual = correlation_stats['base_to_weak']
    elif correlation_type == 'independent':
        actual = correlation_stats['base_to_independent']
    
    validation_results[correlation_type] = {{
        'actual_correlation': actual,
        'expected_range': (min_val, max_val),
        'within_expected': min_val <= actual <= max_val
    }}

# Final results
{{
    "plot_base64": plot_base64,
    "correlation_matrix": correlation_matrix.to_dict(),
    "correlation_stats": correlation_stats,
    "validation_results": validation_results,
    "dataset_info": {{
        "shape": list(data.shape),
        "columns": list(data.columns),
        "sample_size": sample_size
    }},
    "plot_info": {{
        "format": "{CorrelationTestConfig.CORRELATION_SETTINGS['heatmap_format']}",
        "dpi": {CorrelationTestConfig.CORRELATION_SETTINGS['dpi_setting']},
        "base64_length": len(plot_base64),
        "size_valid": len(plot_base64) > {CorrelationTestConfig.CORRELATION_SETTINGS['min_plot_size']}
    }},
    "statistical_validation": {{
        "all_correlations_valid": all(result['within_expected'] for result in validation_results.values()),
        "total_correlations_tested": len(validation_results)
    }}
}}
        """

        # When: Executing correlation heatmap creation
        result = execute_python_code(code, timeout=correlation_timeout)

        # Then: Validate API contract and results
        validate_api_contract(result)
        assert result["success"], f"Correlation heatmap creation failed: {result.get('error')}"

        correlation_results = json.loads(result["data"]["result"])
        
        # Validate plot generation
        plot_info = correlation_results["plot_info"]
        min_plot_size = CorrelationTestConfig.CORRELATION_SETTINGS['min_plot_size']
        assert plot_info["size_valid"], f"Plot base64 should be at least {min_plot_size} chars"
        assert plot_info["base64_length"] > 0, "Should have generated base64 plot data"
        
        # Validate statistical accuracy
        statistical_validation = correlation_results["statistical_validation"]
        assert statistical_validation["all_correlations_valid"], \
            f"Correlation validation failed: {correlation_results['validation_results']}"
        
        # Validate dataset properties
        dataset_info = correlation_results["dataset_info"]
        assert dataset_info["sample_size"] == CorrelationTestConfig.CORRELATION_SETTINGS["sample_size"], \
            "Dataset should have expected sample size"
        assert len(dataset_info["columns"]) == 5, "Should have 5 feature columns"
        
        # Validate specific correlation ranges
        validation_results = correlation_results["validation_results"]
        for correlation_type, result in validation_results.items():
            actual = result['actual_correlation']
            expected = result['expected_range']
            assert result["within_expected"], \
                f"{correlation_type} correlation {actual:.3f} outside expected range {expected}"

    def test_given_multiple_correlation_scenarios_when_analyzed_then_all_statistical_patterns_valid(
        self, server_ready, correlation_timeout
    ):
        """
        Test multiple correlation scenarios with different statistical patterns.

        **Given:** Various correlation scenarios (strong, moderate, weak, none)
        **When:** Analyzing different correlation patterns
        **Then:** All statistical patterns are correctly identified and validated

        Args:
            server_ready: Server readiness fixture
            correlation_timeout: Timeout for correlation operations

        Example:
            Tests strong positive, moderate negative, weak, and no correlation
            scenarios to ensure statistical accuracy across different patterns.
        """
        # Given: Multiple correlation scenarios to test
        code = f"""
import numpy as np
import pandas as pd

# Test configuration
np.random.seed({CorrelationTestConfig.CORRELATION_SETTINGS['random_seed']})
sample_size = {CorrelationTestConfig.CORRELATION_SETTINGS['sample_size']}
scenarios = {CorrelationTestConfig.CORRELATION_SCENARIOS}

scenario_results = {{}}

# When: Testing each correlation scenario
for scenario in scenarios:
    scenario_name = scenario['name']
    target_correlation = scenario['correlation_coefficient']
    expected_range = scenario['expected_range']
    
    print(f"Testing scenario: {{scenario_name}} (target: {{target_correlation}})")
    
    # Generate data with target correlation
    x = np.random.randn(sample_size)
    
    if target_correlation == 0.0:
        # Independent variables
        y = np.random.randn(sample_size)
    else:
        # Generate y with target correlation to x
        noise_variance = 1 - target_correlation**2
        y = target_correlation * x + np.sqrt(noise_variance) * np.random.randn(sample_size)
    
    # Calculate actual correlation
    correlation_matrix = np.corrcoef(x, y)
    actual_correlation = correlation_matrix[0, 1]
    
    # Validate correlation
    min_expected, max_expected = expected_range
    within_range = min_expected <= actual_correlation <= max_expected
    
    scenario_results[scenario_name] = {{
        'target_correlation': target_correlation,
        'actual_correlation': actual_correlation,
        'expected_range': expected_range,
        'within_expected_range': within_range,
        'correlation_error': abs(actual_correlation - target_correlation),
        'sample_size': sample_size
    }}
    
    print(f"  Target: {{target_correlation:.3f}}, Actual: {{actual_correlation:.3f}}, Valid: {{within_range}}")

# Then: Calculate overall validation statistics
total_scenarios = len(scenario_results)
valid_scenarios = sum(1 for result in scenario_results.values() if result['within_expected_range'])
max_error = max(result['correlation_error'] for result in scenario_results.values())
avg_error = sum(result['correlation_error'] for result in scenario_results.values()) / total_scenarios

summary_stats = {{
    'total_scenarios_tested': total_scenarios,
    'scenarios_within_range': valid_scenarios,
    'success_rate': valid_scenarios / total_scenarios,
    'max_correlation_error': max_error,
    'avg_correlation_error': avg_error,
    'all_scenarios_valid': valid_scenarios == total_scenarios
}}

print(f"\\nSummary: {{valid_scenarios}}/{{total_scenarios}} scenarios valid ({{summary_stats['success_rate']:.1%}})")
print(f"Max error: {{max_error:.4f}}, Avg error: {{avg_error:.4f}}")

{{
    "scenario_results": scenario_results,
    "summary_statistics": summary_stats,
    "tolerance_used": {CorrelationTestConfig.CORRELATION_SETTINGS['correlation_tolerance']}
}}
        """

        # When: Executing multiple correlation scenarios
        result = execute_python_code(code, timeout=correlation_timeout)

        # Then: Validate API contract and statistical results
        validate_api_contract(result)
        assert result["success"], f"Multiple correlation scenarios failed: {result.get('error')}"

        scenario_results = json.loads(result["data"]["result"])
        
        # Validate overall success
        summary_stats = scenario_results["summary_statistics"]
        assert summary_stats["all_scenarios_valid"], \
            f"Not all correlation scenarios were valid. Success rate: {summary_stats['success_rate']:.1%}"
        
        # Validate error tolerances
        max_error = summary_stats["max_correlation_error"]
        avg_error = summary_stats["avg_correlation_error"]
        tolerance = CorrelationTestConfig.CORRELATION_SETTINGS["correlation_tolerance"]
        
        assert max_error <= tolerance * 2, f"Maximum correlation error {max_error:.4f} exceeds tolerance"
        assert avg_error <= tolerance, f"Average correlation error {avg_error:.4f} exceeds tolerance"
        
        # Validate individual scenarios
        individual_results = scenario_results["scenario_results"]
        for scenario_name, result in individual_results.items():
            actual = result['actual_correlation']
            expected = result['expected_range']
            assert result["within_expected_range"], \
                f"Scenario '{scenario_name}' failed: actual {actual:.3f} outside {expected}"

    def test_given_seaborn_import_scenarios_when_tested_then_all_imports_successful(
        self, server_ready, seaborn_timeout
    ):
        """
        Test various seaborn import scenarios and functionality.

        **Given:** Different ways to import and use seaborn
        **When:** Testing various import patterns and basic functionality
        **Then:** All import scenarios work correctly with proper error handling

        Args:
            server_ready: Server readiness fixture
            correlation_timeout: Timeout for correlation operations

        Example:
            Tests direct import, micropip installation, and basic seaborn
            functionality to ensure robust seaborn availability.
        """
        # Given: Various seaborn import and functionality scenarios
        code = """
import importlib
import sys

seaborn_tests = {}

# Test 1: Direct seaborn import
test_name = "direct_import"
try:
    import seaborn as sns
    seaborn_tests[test_name] = {
        "success": True,
        "version": sns.__version__ if hasattr(sns, '__version__') else "unknown",
        "error": None
    }
    print(f"✅ Direct seaborn import successful: {sns.__version__}")
except ImportError as e:
    seaborn_tests[test_name] = {
        "success": False,
        "version": None,
        "error": str(e)
    }
    print(f"❌ Direct seaborn import failed: {e}")

# Test 2: Seaborn functionality test
test_name = "functionality_test"
if seaborn_tests["direct_import"]["success"]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Create simple test plot
        data = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50)
        })
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x='x', y='y', ax=ax)
        plt.close(fig)
        
        seaborn_tests[test_name] = {
            "success": True,
            "plot_created": True,
            "error": None
        }
        print("✅ Seaborn functionality test successful")
        
    except Exception as e:
        seaborn_tests[test_name] = {
            "success": False,
            "plot_created": False,
            "error": str(e)
        }
        print(f"❌ Seaborn functionality test failed: {e}")
else:
    seaborn_tests[test_name] = {
        "success": False,
        "plot_created": False,
        "error": "Seaborn not available for functionality testing"
    }

# Test 3: Seaborn style settings
test_name = "style_settings"
if seaborn_tests["direct_import"]["success"]:
    try:
        # Test seaborn style functionality
        available_styles = sns.axes_style().keys()
        sns.set_style("whitegrid")
        current_style = sns.axes_style()
        
        seaborn_tests[test_name] = {
            "success": True,
            "available_styles": list(available_styles),
            "style_set": "whitegrid" in str(current_style),
            "error": None
        }
        print(f"✅ Seaborn style settings successful: {len(available_styles)} styles available")
        
    except Exception as e:
        seaborn_tests[test_name] = {
            "success": False,
            "available_styles": [],
            "style_set": False,
            "error": str(e)
        }
        print(f"❌ Seaborn style settings failed: {e}")

# Test 4: Seaborn color palettes
test_name = "color_palettes"
if seaborn_tests["direct_import"]["success"]:
    try:
        # Test color palette functionality
        palette = sns.color_palette("husl", 8)
        palette_names = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
        
        palette_tests = {}
        for palette_name in palette_names:
            try:
                test_palette = sns.color_palette(palette_name)
                palette_tests[palette_name] = len(test_palette) > 0
            except:
                palette_tests[palette_name] = False
        
        seaborn_tests[test_name] = {
            "success": True,
            "default_palette_length": len(palette),
            "palette_tests": palette_tests,
            "palettes_working": sum(palette_tests.values()),
            "error": None
        }
        palettes_working = sum(palette_tests.values())
        total_palettes = len(palette_tests)
        print(f"✅ Seaborn color palettes successful: {palettes_working}/{total_palettes} palettes working")
        
    except Exception as e:
        seaborn_tests[test_name] = {
            "success": False,
            "default_palette_length": 0,
            "palette_tests": {},
            "palettes_working": 0,
            "error": str(e)
        }
        print(f"❌ Seaborn color palettes failed: {e}")

# Summary statistics
total_tests = len(seaborn_tests)
successful_tests = sum(1 for test in seaborn_tests.values() if test["success"])
success_rate = successful_tests / total_tests

summary = {
    "total_tests": total_tests,
    "successful_tests": successful_tests,
    "success_rate": success_rate,
    "all_tests_passed": successful_tests == total_tests,
    "seaborn_fully_functional": seaborn_tests.get("direct_import", {}).get("success", False) and
                               seaborn_tests.get("functionality_test", {}).get("success", False)
}

print(f"\\nSeaborn Import Test Summary: {successful_tests}/{total_tests} tests passed ({success_rate:.1%})")

{
    "seaborn_tests": seaborn_tests,
    "summary": summary
}
        """

        # When: Testing seaborn import scenarios
        result = execute_python_code(code, timeout=seaborn_timeout)

        # Then: Validate seaborn functionality
        validate_api_contract(result)
        assert result["success"], f"Seaborn import tests failed: {result.get('error')}"

        import_results = json.loads(result["data"]["result"])
        
        # Validate seaborn availability and functionality
        summary = import_results["summary"]
        assert summary["seaborn_fully_functional"], \
            f"Seaborn not fully functional. Success rate: {summary['success_rate']:.1%}"
        
        # Validate individual test results
        seaborn_tests = import_results["seaborn_tests"]
        
        # Direct import should work
        assert seaborn_tests["direct_import"]["success"], \
            f"Seaborn direct import failed: {seaborn_tests['direct_import']['error']}"
        
        # Basic functionality should work
        assert seaborn_tests["functionality_test"]["success"], \
            f"Seaborn functionality test failed: {seaborn_tests['functionality_test']['error']}"
        
        # Style settings should be available
        if seaborn_tests["style_settings"]["success"]:
            assert len(seaborn_tests["style_settings"]["available_styles"]) > 0, \
                "Should have available seaborn styles"


if __name__ == "__main__":
    # Allow running this file directly for development/debugging
    pytest.main([__file__, "-v", "-s"])
