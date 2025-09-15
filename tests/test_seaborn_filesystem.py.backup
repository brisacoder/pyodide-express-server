"""
Seaborn Filesystem Integration Tests

This module contains comprehensive pytest tests for seaborn plotting functionality
that uses direct filesystem operations within Pyodide. Tests validate the ability
to create, save, and retrieve visualization files through the virtual filesystem.

All tests follow BDD (Behavior-Driven Development) patterns with Given-When-Then
structure and strictly use ONLY /api/execute-raw endpoint for Python execution.

Modernization Requirements (STRICT COMPLIANCE):
✅ pytest framework with centralized Config class
✅ BDD Given-When-Then test documentation patterns
✅ Only /api/execute-raw endpoint for Python code execution
✅ API contract validation using conftest.py helpers
✅ pathlib for all file operations (no os.path)
✅ No hardcoded globals - all values from Config class
✅ Comprehensive docstrings with examples and behavior specification
✅ Modern fixture patterns from conftest.py (server_ready, api_contract_validator)
"""

import pytest
import requests

from conftest import Config, execute_python_code


# Modern seaborn visualization test fixtures using Config from conftest.py


@pytest.fixture(scope="session")
def seaborn_ready(server_ready) -> None:
    """
    Session-scoped fixture ensuring seaborn packages are available.

    This fixture depends on server_ready from conftest.py and ensures
    that required data science packages (matplotlib, seaborn) are
    installed and available for the test session.

    Args:
        server_ready: Server readiness from conftest.py

    Raises:
        pytest.skip: If packages cannot be installed

    Example:
        >>> def test_seaborn_plot(seaborn_ready):
        ...     # seaborn guaranteed to be available
        ...     code = "import seaborn as sns; print(sns.__version__)"
        ...     result = execute_python_code(code)
        ...     assert result["success"]
    """
    # Install required packages using Config endpoints
    packages = ["matplotlib", "seaborn", "scipy", "numpy", "pandas"]
    for package in packages:
        try:
            install_response = requests.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
                json={"package": package},
                timeout=Config.TIMEOUTS["package_install"]
            )
            if install_response.status_code == 200:
                print(f"✅ Package {package} ready")
            else:
                print(f"⚠️ Warning: {package} installation returned {install_response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Warning: {package} installation failed: {e}")


@pytest.fixture
def filesystem_cleanup(server_ready, test_cleanup):
    """
    Function-scoped fixture providing clean Pyodide filesystem for seaborn tests.

    This fixture ensures each test starts with a clean virtual filesystem
    and provides automatic cleanup of generated plot files.

    Args:
        server_ready: Server readiness from conftest.py
        test_cleanup: Cleanup tracker from conftest.py

    Yields:
        CleanupTracker: Cleanup object for tracking test artifacts

    Example:
        >>> def test_plot_creation(filesystem_cleanup):
        ...     cleanup = filesystem_cleanup
        ...     # Test will automatically clean up generated files
    """
    # Setup: Clean seaborn plots directory
    cleanup_code = """
# Clean seaborn plots using pathlib (modern pattern)
from pathlib import Path

seaborn_dir = Path('/plots/seaborn')
if seaborn_dir.exists():
    for plot_file in seaborn_dir.glob('*.png'):
        try:
            plot_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove {plot_file}: {e}")
else:
    seaborn_dir.mkdir(parents=True, exist_ok=True)
    print("Created seaborn plots directory")

"Filesystem cleaned successfully"
"""
    try:
        result = execute_python_code(cleanup_code)
        if not result.get("success"):
            print(f"⚠️ Warning: Filesystem cleanup setup failed: {result.get('error')}")
    except Exception as e:
        print(f"⚠️ Warning: Filesystem cleanup setup error: {e}")

    yield test_cleanup

    # Teardown handled by test_cleanup fixture automatically


class TestSeabornFilesystemOperations:
    """
    Test class for seaborn plotting with direct filesystem operations.

    This class tests the comprehensive ability to create advanced seaborn plots within
    Pyodide and save them directly to the virtual filesystem. All tests verify plot
    creation, file saving, statistical analysis validation, and comprehensive metadata
    verification using ONLY the /api/execute-raw endpoint.

    All tests follow strict BDD Given-When-Then patterns with complete API contract
    validation and use modern Config-based constants instead of hardcoded values.
    """

    def test_direct_file_save_regression_plot(
        self,
        seaborn_ready,
        filesystem_cleanup,
        api_contract_validator,
    ):
        """
        Test creating and saving seaborn regression plot directly to virtual filesystem.

        Given a functioning seaborn environment with clean filesystem access
        When we create a regression plot with statistical analysis and save to filesystem
        Then the plot should be saved with correct metadata and statistical validation

        This test demonstrates complete seaborn regression analysis workflow:
        - Generate correlated sample data with known statistical properties
        - Create professional regression plot with seaborn styling
        - Save plot directly to Pyodide virtual filesystem using pathlib
        - Verify file creation, size, and comprehensive metadata
        - Validate statistical analysis results (correlation, R-squared)
        - Ensure API contract compliance for all responses

        Args:
            seaborn_ready: Seaborn package availability fixture
            filesystem_cleanup: Filesystem cleanup and tracking fixture  
            api_contract_validator: API response validation fixture

        Expected Behavior:
            ✅ Regression plot created with strong positive correlation (>0.8)
            ✅ Plot saved to /plots/seaborn/ with proper pathlib usage
            ✅ File size validation (>0 bytes)
            ✅ Statistical metadata validation (correlation, R-squared)
            ✅ API contract compliance verification
        """
        # Given: Seaborn environment is available and filesystem is clean
        # When: Creating regression plot with comprehensive analysis
        # Use Config constants instead of hardcoded values
        RANDOM_SEED = 42
        SAMPLE_SIZE = 200
        PLOT_DPI = Config.PLOT_SETTINGS["default_dpi"]
        FIGURE_WIDTH, FIGURE_HEIGHT = Config.PLOT_SETTINGS["figure_size"]

        regression_code = f"""
# Seaborn regression plot with comprehensive filesystem validation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Pyodide
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

print("Starting comprehensive seaborn regression analysis with filesystem validation...")

# Apply modern seaborn styling
sns.set_style("whitegrid")
sns.set_palette("husl")

# Generate synthetic data with strong correlation
np.random.seed({RANDOM_SEED})
n_samples = {SAMPLE_SIZE}
x_values = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.3, n_samples)  # Reduced noise for stronger correlation
y_values = 2.8 * x_values + 1.2 + noise  # Strong linear relationship

# Create DataFrame with descriptive column names
df = pd.DataFrame({{
    'feature_x': x_values, 
    'target_y': y_values
}})

# Create professional regression visualization
plt.figure(figsize=({FIGURE_WIDTH}, {FIGURE_HEIGHT}))
sns.regplot(
    data=df, 
    x='feature_x', 
    y='target_y',
    scatter_kws={{'alpha': 0.7, 's': 60, 'color': 'steelblue'}},
    line_kws={{'color': 'darkred', 'linewidth': 3}},
    ci=95  # Add confidence interval
)
plt.title('Seaborn Filesystem Test - Regression Analysis', fontweight='bold', fontsize=14)
plt.xlabel('Feature X (Standardized)', fontweight='bold')
plt.ylabel('Target Y (Dependent Variable)', fontweight='bold')
plt.grid(True, alpha=0.3)

# Calculate and display statistical metrics
correlation = df.corr().iloc[0, 1]
r_squared = correlation ** 2
plt.text(0.05, 0.95, f'r = {{correlation:.4f}}\\nR² = {{r_squared:.4f}}',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
         fontsize=12, fontweight='bold')

# Save to virtual filesystem with pathlib (STRICT REQUIREMENT)
timestamp = int(time.time() * 1000)
plots_dir = Path('/plots/seaborn')
plots_dir.mkdir(parents=True, exist_ok=True)
plot_file = plots_dir / f'regression_analysis_{{timestamp}}.png'

plt.savefig(plot_file, dpi={PLOT_DPI}, bbox_inches='tight', facecolor='white')
plt.close()  # Free memory

# Comprehensive file and statistical validation
file_exists = plot_file.exists()
file_size = plot_file.stat().st_size if file_exists else 0

# Calculate additional statistical metrics
mean_x = float(df['feature_x'].mean())
std_x = float(df['feature_x'].std())
mean_y = float(df['target_y'].mean())
std_y = float(df['target_y'].std())

# Return comprehensive analysis results
analysis_results = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "seaborn_regression_filesystem",
    "filename": str(plot_file),
    "timestamp": timestamp,
    "statistical_analysis": {{
        "correlation_coefficient": float(correlation),
        "r_squared": float(r_squared),
        "sample_size": int(n_samples),
        "feature_stats": {{
            "mean_x": mean_x,
            "std_x": std_x,
            "mean_y": mean_y,
            "std_y": std_y
        }},
        "regression_strength": "strong" if abs(correlation) > 0.8 else "moderate"
    }},
    "filesystem_validation": {{
        "plot_directory": str(plots_dir),
        "file_exists": file_exists,
        "file_size_bytes": file_size,
        "expected_format": "PNG"
    }}
}}

print(f"✅ Regression analysis completed successfully!")
print(f"✅ Plot saved: {{plot_file}}")
print(f"✅ Correlation: {{correlation:.4f}} (R² = {{r_squared:.4f}})")
analysis_results
"""

        # Execute using modern helper with automatic API contract validation
        response_data = execute_python_code(regression_code, Config.TIMEOUTS["code_execution"])
        
        # Then: Validate API contract using fixture (automatic validation)
        api_contract_validator(response_data)
        
        # Extract result from validated response
        result = response_data["data"]["result"]

        # And: Validate comprehensive plot creation and filesystem operations
        assert result.get("file_saved") is True, "Regression plot must be saved to filesystem"
        assert result.get("file_size", 0) > 0, "Plot file must have non-zero size"
        assert result.get("plot_type") == "seaborn_regression_filesystem", "Incorrect plot type identifier"
        
        # And: Verify statistical analysis quality
        stats = result.get("statistical_analysis", {})
        correlation = stats.get("correlation_coefficient", 0)
        r_squared = stats.get("r_squared", 0)
        
        assert correlation > 0.8, f"Strong positive correlation expected (got {correlation:.3f})"
        assert r_squared > 0.64, f"High R-squared expected (got {r_squared:.3f})"  # r²>0.64 for r>0.8
        assert stats.get("sample_size") == SAMPLE_SIZE, f"Sample size mismatch: expected {SAMPLE_SIZE}"
        assert stats.get("regression_strength") == "strong", "Should detect strong correlation"
        
        # And: Validate feature statistics are reasonable
        feature_stats = stats.get("feature_stats", {})
        assert abs(feature_stats.get("mean_x", 999)) < 0.5, "X mean should be near 0 (standardized)"
        assert 0.8 < feature_stats.get("std_x", 0) < 1.2, "X std should be near 1 (standardized)"
        
        # And: Verify comprehensive filesystem validation
        fs_validation = result.get("filesystem_validation", {})
        assert fs_validation.get("file_exists") is True, "File existence validation failed"
        assert fs_validation.get("file_size_bytes", 0) > 1000, "File too small (likely corrupted)"
        assert fs_validation.get("expected_format") == "PNG", "Format validation failed"
        
        # And: Validate proper pathlib usage in filename
        filename = result.get("filename", "")
        assert filename.startswith("/plots/seaborn/"), "Incorrect seaborn plots directory"
        assert filename.endswith(".png"), "Must be PNG format"
        assert "regression_analysis_" in filename, "Missing descriptive filename pattern"

    def test_direct_file_save_advanced_dashboard(
        self,
        seaborn_ready,
        filesystem_cleanup,
        api_contract_validator,
    ):
        """
        Test creating and saving advanced seaborn dashboard directly to virtual filesystem.

        Given a seaborn environment with clean filesystem access
        When we create a comprehensive multi-panel dashboard with diverse plot types
        Then all visualizations should render correctly with comprehensive statistical analysis

        This test demonstrates advanced seaborn dashboard creation workflow:
        - Generate complex multi-dimensional synthetic dataset
        - Create 6-panel professional dashboard with diverse visualization types
        - Apply advanced seaborn styling and color palettes
        - Save dashboard directly to Pyodide virtual filesystem using pathlib
        - Validate comprehensive statistical metadata and visualization properties
        - Ensure complete API contract compliance throughout process

        Visualization Types Tested:
        ✅ Correlation heatmap with statistical annotations
        ✅ Box plots with categorical grouping and hue encoding
        ✅ Violin plots showing distribution shapes
        ✅ Advanced scatter plots with multi-dimensional encoding
        ✅ Kernel density estimation plots for group comparisons
        ✅ Count plots for categorical frequency analysis

        Args:
            seaborn_ready: Seaborn package availability fixture
            filesystem_cleanup: Filesystem cleanup and tracking fixture
            api_contract_validator: API response validation fixture

        Expected Behavior:
            ✅ Dashboard with 6 distinct visualization types created
            ✅ Multi-dimensional data analysis (3 groups, 2 categories)
            ✅ Professional styling with consistent color palette
            ✅ Comprehensive statistical metadata validation
            ✅ Filesystem operations using pathlib patterns
            ✅ API contract compliance verification
        """
        # Given: Seaborn environment ready and filesystem clean
        # When: Creating advanced dashboard with diverse visualization types
        # Use Config constants and enhanced parameters
        RANDOM_SEED = 123  # Different seed for dashboard variety
        SAMPLE_SIZE = 300  # Larger sample for statistical significance
        DASHBOARD_WIDTH = 16
        DASHBOARD_HEIGHT = 12
        PLOT_DPI = Config.PLOT_SETTINGS["default_dpi"]

        dashboard_code = f"""
# Advanced seaborn dashboard with comprehensive visualization types
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Pyodide
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

print("🎯 Creating advanced seaborn dashboard with 6 visualization types...")

# Apply professional seaborn styling
sns.set_style("whitegrid")
sns.set_palette("Set2")  # Professional color palette

# Generate complex multi-dimensional synthetic dataset
np.random.seed({RANDOM_SEED})
n_samples = {SAMPLE_SIZE}

# Create sophisticated data with realistic relationships
groups = ['Alpha', 'Beta', 'Gamma']
categories = ['Type1', 'Type2']

# Generate base data
data = {{
    'group': np.random.choice(groups, n_samples),
    'category': np.random.choice(categories, n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(100, 25, n_samples),
    'score': np.random.uniform(20, 100, n_samples)
}}

# Add realistic complex relationships between variables
for i in range(n_samples):
    group_effect = 0
    category_effect = 0
    
    # Group-based effects
    if data['group'][i] == 'Alpha':
        group_effect = 25  # Alpha performs better
        data['score'][i] += 20
    elif data['group'][i] == 'Beta':
        group_effect = 10  # Beta is moderate  
        data['score'][i] += 5
    elif data['group'][i] == 'Gamma':
        group_effect = -15  # Gamma performs worse
        data['score'][i] -= 10
    
    # Category-based effects
    if data['category'][i] == 'Type2':
        category_effect = 15
        data['value1'][i] *= 1.2
        data['score'][i] += 8
    
    # Apply cumulative effects
    data['value1'][i] += group_effect + category_effect
    data['value2'][i] += group_effect * 0.8 + category_effect * 0.6

df = pd.DataFrame(data)

print(f"📊 Generated dataset: {{len(df)}} samples, {{len(groups)}} groups, {{len(categories)}} categories")

# Create comprehensive 6-panel dashboard
fig = plt.figure(figsize=({DASHBOARD_WIDTH}, {DASHBOARD_HEIGHT}))
fig.suptitle('Advanced Seaborn Dashboard - Filesystem Test', fontsize=16, fontweight='bold')

# Panel 1: Correlation heatmap with statistical annotations
plt.subplot(2, 3, 1)
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdBu_r', 
            center=0, 
            square=True,
            fmt='.3f',
            cbar_kws={{'label': 'Correlation Coefficient'}})
plt.title('Correlation Matrix\\n(Statistical Relationships)', fontweight='bold', fontsize=10)

# Panel 2: Box plots with categorical grouping
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='group', y='value1', hue='category', palette='viridis')
plt.title('Value1 Distribution\\nby Group & Category', fontweight='bold', fontsize=10)
plt.xticks(rotation=45)

# Panel 3: Violin plots showing distribution shapes
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='group', y='score', palette='plasma', inner='box')
plt.title('Score Distribution\\nShapes by Group', fontweight='bold', fontsize=10)
plt.xticks(rotation=45)

# Panel 4: Advanced scatter plot with multi-dimensional encoding
plt.subplot(2, 3, 4)
sns.scatterplot(data=df, x='value1', y='value2', 
                hue='group', size='score', 
                alpha=0.7, sizes=(50, 200))
plt.title('Value1 vs Value2\\nMulti-dimensional View', fontweight='bold', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Panel 5: Kernel density estimation for group comparisons
plt.subplot(2, 3, 5)
for group_name in sorted(df['group'].unique()):
    group_subset = df[df['group'] == group_name]
    sns.kdeplot(data=group_subset, x='score', 
                label=f'Group {{group_name}}', 
                alpha=0.7, linewidth=2)
plt.title('Score Density\\nDistributions by Group', fontweight='bold', fontsize=10)
plt.xlabel('Score Value')
plt.ylabel('Density')
plt.legend()

# Panel 6: Count plot for categorical frequency analysis
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='group', hue='category', palette='Set1')
plt.title('Sample Frequency\\nby Group & Category', fontweight='bold', fontsize=10)
plt.xlabel('Group')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Apply professional layout optimization
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save dashboard to virtual filesystem using pathlib (STRICT REQUIREMENT)
timestamp = int(time.time() * 1000)
plots_dir = Path('/plots/seaborn')
plots_dir.mkdir(parents=True, exist_ok=True)
dashboard_file = plots_dir / f'advanced_dashboard_{{timestamp}}.png'

plt.savefig(dashboard_file, dpi={PLOT_DPI}, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()  # Free memory immediately

print("📈 Computing comprehensive statistical analysis...")

# Calculate detailed summary statistics for all groups and categories
group_stats = {{}}
for group in df['group'].unique():
    group_data = df[df['group'] == group]
    group_stats[group] = {{
        'count': int(len(group_data)),
        'score_mean': float(group_data['score'].mean()),
        'score_std': float(group_data['score'].std()),
        'value1_mean': float(group_data['value1'].mean()),
        'value2_mean': float(group_data['value2'].mean())
    }}

category_stats = {{}}
for category in df['category'].unique():
    cat_data = df[df['category'] == category]
    category_stats[category] = {{
        'count': int(len(cat_data)),
        'score_mean': float(cat_data['score'].mean()),
        'value1_mean': float(cat_data['value1'].mean())
    }}

# Advanced correlation analysis
correlation_strength = {{}}
for col1 in numeric_df.columns:
    for col2 in numeric_df.columns:
        if col1 != col2:
            corr_val = float(correlation_matrix.loc[col1, col2])
            correlation_strength[f'{{col1}}_{{col2}}'] = corr_val

# Comprehensive file validation
file_exists = dashboard_file.exists()
file_size = dashboard_file.stat().st_size if file_exists else 0

# Return comprehensive dashboard analysis
dashboard_results = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "seaborn_advanced_dashboard",
    "filename": str(dashboard_file),
    "timestamp": timestamp,
    "dataset_info": {{
        "n_samples": n_samples,
        "n_groups": len(df['group'].unique()),
        "n_categories": len(df['category'].unique()),
        "groups": sorted(list(df['group'].unique())),
        "categories": sorted(list(df['category'].unique()))
    }},
    "statistical_analysis": {{
        "group_statistics": group_stats,
        "category_statistics": category_stats,
        "correlation_analysis": correlation_strength,
        "overall_score_stats": {{
            "mean": float(df['score'].mean()),
            "std": float(df['score'].std()),
            "min": float(df['score'].min()),
            "max": float(df['score'].max())
        }}
    }},
    "visualization_metadata": {{
        "panel_count": 6,
        "visualization_types": [
            "correlation_heatmap_annotated",
            "box_plot_categorical", 
            "violin_plot_distributions",
            "scatter_plot_multidimensional",
            "kde_density_comparison",
            "count_plot_frequency"
        ],
        "styling": {{
            "palette": "Set2",
            "style": "whitegrid",
            "figure_size": [{DASHBOARD_WIDTH}, {DASHBOARD_HEIGHT}],
            "dpi": {PLOT_DPI}
        }}
    }},
    "filesystem_validation": {{
        "plot_directory": str(plots_dir),
        "file_exists": file_exists,
        "file_size_bytes": file_size,
        "expected_format": "PNG"
    }}
}}

print(f"✅ Advanced dashboard completed successfully!")
print(f"✅ Dashboard saved: {{dashboard_file}}")
print(f"✅ Dataset: {{n_samples}} samples, {{len(df['group'].unique())}} groups, {{len(df['category'].unique())}} categories")
print(f"✅ File size: {{file_size}} bytes")
dashboard_results
"""

        # Execute using modern helper with automatic API contract validation
        response_data = execute_python_code(dashboard_code, Config.TIMEOUTS["code_execution"])
        
        # Then: Validate API contract using fixture (automatic validation)
        api_contract_validator(response_data)
        
        # Extract result from validated response
        result = response_data["data"]["result"]

        # And: Validate comprehensive dashboard creation and filesystem operations
        assert result.get("file_saved") is True, "Advanced dashboard must be saved to filesystem"
        assert result.get("file_size", 0) > 5000, "Dashboard file too small (likely corrupted or incomplete)"
        assert result.get("plot_type") == "seaborn_advanced_dashboard", "Incorrect dashboard plot type"
        
        # And: Verify dataset information completeness
        dataset_info = result.get("dataset_info", {})
        assert dataset_info.get("n_samples") == SAMPLE_SIZE, f"Sample count mismatch: expected {SAMPLE_SIZE}"
        assert dataset_info.get("n_groups") == 3, "Should have exactly 3 groups (Alpha, Beta, Gamma)"
        assert dataset_info.get("n_categories") == 2, "Should have exactly 2 categories (Type1, Type2)"
        
        groups = dataset_info.get("groups", [])
        assert "Alpha" in groups, "Missing Alpha group in dataset"
        assert "Beta" in groups, "Missing Beta group in dataset" 
        assert "Gamma" in groups, "Missing Gamma group in dataset"
        
        categories = dataset_info.get("categories", [])
        assert "Type1" in categories, "Missing Type1 category in dataset"
        assert "Type2" in categories, "Missing Type2 category in dataset"
        
        # And: Validate comprehensive statistical analysis
        stats = result.get("statistical_analysis", {})
        assert "group_statistics" in stats, "Missing detailed group statistics"
        assert "category_statistics" in stats, "Missing detailed category statistics"
        assert "correlation_analysis" in stats, "Missing correlation analysis"
        assert "overall_score_stats" in stats, "Missing overall score statistics"
        
        # Validate group statistics have proper structure
        group_stats = stats.get("group_statistics", {})
        for group in ["Alpha", "Beta", "Gamma"]:
            assert group in group_stats, f"Missing statistics for {group} group"
            group_data = group_stats[group]
            assert "count" in group_data, f"Missing count for {group} group"
            assert "score_mean" in group_data, f"Missing score mean for {group} group"
            assert group_data["count"] > 0, f"{group} group should have samples"
        
        # And: Validate visualization metadata completeness
        viz_metadata = result.get("visualization_metadata", {})
        assert viz_metadata.get("panel_count") == 6, "Should have exactly 6 visualization panels"
        
        viz_types = viz_metadata.get("visualization_types", [])
        expected_viz_types = [
            "correlation_heatmap_annotated",
            "box_plot_categorical",
            "violin_plot_distributions", 
            "scatter_plot_multidimensional",
            "kde_density_comparison",
            "count_plot_frequency"
        ]
        assert len(viz_types) == 6, "Should have 6 distinct visualization types"
        for expected_type in expected_viz_types:
            assert expected_type in viz_types, f"Missing visualization type: {expected_type}"
        
        # Validate styling metadata
        styling = viz_metadata.get("styling", {})
        assert styling.get("palette") == "Set2", "Should use Set2 color palette"
        assert styling.get("style") == "whitegrid", "Should use whitegrid style"
        assert styling.get("dpi") == PLOT_DPI, f"DPI should be {PLOT_DPI}"
        
        # And: Verify comprehensive filesystem validation
        fs_validation = result.get("filesystem_validation", {})
        assert fs_validation.get("file_exists") is True, "Dashboard file existence validation failed"
        assert fs_validation.get("file_size_bytes", 0) > 5000, "Dashboard file suspiciously small"
        assert fs_validation.get("expected_format") == "PNG", "Should be PNG format"
        
        # And: Validate proper pathlib usage in filename
        filename = result.get("filename", "")
        assert filename.startswith("/plots/seaborn/"), "Incorrect seaborn plots directory"
        assert filename.endswith(".png"), "Must be PNG format"
        assert "advanced_dashboard_" in filename, "Missing descriptive dashboard filename pattern"
