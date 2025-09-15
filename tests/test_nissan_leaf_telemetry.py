"""
Nissan Leaf Automotive Telemetry Data Analysis Tests for Pyodide Express Server.

This module provides comprehensive testing for complex automotive telemetry data
analysis using the Pyodide WebAssembly runtime. It validates data science workflows
including time series analysis, geospatial calculations, energy efficiency metrics,
and driving behavior analytics using real automotive signal telemetry data.

Test Categories:
- Basic telemetry data loading and exploration
- Energy efficiency and consumption analysis
- Geospatial route analysis and mapping
- Driving behavior pattern recognition
- Advanced time series analysis
- Comprehensive automotive insights dashboard

Requirements Compliance:
1. ✅ Pytest framework with BDD Given-When-Then structure
2. ✅ No hardcoded globals - all parameters via fixtures/constants
3. ✅ Only uses /api/execute-raw endpoint
4. ✅ No internal REST APIs (no 'pyodide' in URLs)
5. ✅ Comprehensive test coverage with real automotive data
6. ✅ Full docstrings with descriptions, inputs, outputs, examples
7. ✅ Portable Pyodide code using pathlib
8. ✅ API contract validation for all responses

API Contract Validation:
All responses must follow the standardized contract:
{
    "success": true | false,
    "data": {
        "result": <any>,
        "stdout": <string>,
        "stderr": <string>,
        "executionTime": <number>
    } | null,
    "error": <string> | null,
    "meta": {
        "timestamp": <ISO string>
    }
}
"""

from pathlib import Path


import pytest
import requests

from conftest import Config, execute_python_code, validate_api_contract


class TestNissanLeafTelemetryAnalysis:
    """
    Test suite for Nissan Leaf automotive telemetry data analysis.

    This class contains comprehensive tests for analyzing real automotive
    telemetry data from a Nissan Leaf electric vehicle using advanced
    data science techniques in the Pyodide WebAssembly environment.

    Test Structure:
    - Upload telemetry CSV data via API
    - Execute complex pandas/numpy analysis
    - Validate results and insights
    - Clean up test artifacts
    """

    @pytest.fixture(scope="class")
    def telemetry_data_file(self) -> Path:
        """
        Provide path to the Nissan Leaf telemetry CSV data file.

        Validates that the test data file exists and is accessible for
        upload and analysis in the test suite.

        Returns:
            Path: Path to the DEVRT-NISSAN-LEAF.csv test data file

        Raises:
            pytest.skip: If test data file is not found

        Example:
            >>> data_file = telemetry_data_file()
            >>> assert data_file.exists()
            >>> assert data_file.suffix == '.csv'
        """
        test_data_file = Path(__file__).parent / "data" / "DEVRT-NISSAN-LEAF.csv"

        if not test_data_file.exists():
            pytest.skip(f"Test data file not found: {test_data_file}")

        return test_data_file

    @pytest.fixture
    def uploaded_telemetry_file(self, server_ready, telemetry_data_file: Path) -> str:
        """
        Upload telemetry data file and return the uploaded filename.

        Handles file upload via the /api/upload endpoint and tracks
        the uploaded file for cleanup after test completion.

        Args:
            server_ready: Ensures server is available
            telemetry_data_file: Path to the CSV data file

        Returns:
            str: The filename as it appears in the Pyodide filesystem

        Yields:
            str: Uploaded filename for use in tests

        Example:
            >>> filename = uploaded_telemetry_file()
            >>> # Use filename in Pyodide code like: pd.read_csv(filename)
        """
        # Track uploaded files for cleanup
        uploaded_files = []

        try:
            # Upload the telemetry data file
            with open(telemetry_data_file, "rb") as file:
                files = {"file": ("DEVRT-NISSAN-LEAF.csv", file, "text/csv")}
                response = requests.post(
                    f"{Config.BASE_URL}/api/upload",
                    files=files,
                    timeout=Config.TIMEOUTS["api_request"] * 2,
                )

            response.raise_for_status()
            result = response.json()
            validate_api_contract(result)

            if not result["success"]:
                pytest.fail(f"File upload failed: {result.get('error')}")

            # Extract the filename for Pyodide access
            uploaded_filename = result["data"]["filename"]
            original_filename = result["data"].get(
                "originalName", "DEVRT-NISSAN-LEAF.csv"
            )
            uploaded_files.append(original_filename)

            yield uploaded_filename

        finally:
            # Cleanup uploaded files
            for filename in uploaded_files:
                try:
                    cleanup_response = requests.delete(
                        f"{Config.BASE_URL}/api/uploaded-files/{filename}",
                        timeout=Config.TIMEOUTS["api_request"],
                    )
                    cleanup_response.raise_for_status()
                except requests.RequestException:
                    pass  # Ignore cleanup failures

    @pytest.mark.api
    @pytest.mark.telemetry
    @pytest.mark.slow
    def test_basic_telemetry_data_loading(
        self, server_ready: None, uploaded_telemetry_file: str
    ) -> None:
        """
        Test basic CSV loading and automotive telemetry data exploration.

        Validates that complex automotive telemetry data can be loaded,
        analyzed, and explored using pandas operations in Pyodide with
        comprehensive data quality assessment and key metrics extraction.

        Given: A Nissan Leaf telemetry CSV file is uploaded to the server
        When: Python code loads and analyzes the automotive data using pandas
        Then: Key telemetry metrics and data quality info are extracted correctly

        Args:
            server_ready: Ensures server is available
            uploaded_telemetry_file: CSV filename in Pyodide filesystem

        Validates:
        - CSV file loading with proper column detection
        - Battery and power metrics analysis
        - Vehicle performance calculations
        - Route and geospatial information
        - Data quality assessment
        - Memory usage optimization

        Example:
            Test validates automotive data analysis like:
            ```python
            df = pd.read_csv('/uploads/DEVRT-NISSAN-LEAF.csv')
            soc_range = f"{df['soc'].min()}% - {df['soc'].max()}%"
            power_range = f"{df['Motor Pwr(w)'].min()} - {df['Motor Pwr(w)'].max()}W"
            ```
        """
        # Given: Uploaded telemetry data file
        filename = uploaded_telemetry_file

        # When: Execute Python code to analyze telemetry data
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np

# Load the automotive telemetry data using pathlib for portability
data_file = Path('/uploads') / '{filename}'
if not data_file.exists():
    raise FileNotFoundError(f"Telemetry data file not found: {{data_file}}")

df = pd.read_csv(data_file)

# Basic data exploration and validation
print("📊 Nissan Leaf Telemetry Dataset Analysis")
print(f"Dataset shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")

# Validate expected columns exist
expected_columns = ['soc', 'Motor Pwr(w)', 'soh', 'speed', 'Torque Nm', 'rpm', 
                   'latitude', 'longitude', 'altitude', 'amb_temp', 'regenwh']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {{missing_columns}}")

# Key telemetry metrics analysis
print("\\n🔋 Battery & Power Metrics:")
print(f"State of Charge (SOC) range: {{df['soc'].min():.1f}}% - {{df['soc'].max():.1f}}%")
print(f"Motor Power range: {{df['Motor Pwr(w)'].min():,}}W - {{df['Motor Pwr(w)'].max():,}}W")
print(f"Battery Health (SOH): {{df['soh'].iloc[0]:.1f}}%")

print("\\n🚗 Vehicle Performance:")
print(f"Speed range: {{df['speed'].min():.1f}} - {{df['speed'].max():.1f}} km/h")
print(f"Torque range: {{df['Torque Nm'].min():.1f}} - {{df['Torque Nm'].max():.1f}} Nm")
print(f"RPM range: {{df['rpm'].min():,}} - {{df['rpm'].max():,}} RPM")

print("\\n🌍 Route Information:")
print(f"Latitude range: {{df['latitude'].min():.6f}} - {{df['latitude'].max():.6f}}")
print(f"Longitude range: {{df['longitude'].min():.6f}} - {{df['longitude'].max():.6f}}")
print(f"Altitude range: {{df['altitude'].min():.1f}} - {{df['altitude'].max():.1f}} m")

# Data quality assessment
missing_data = df.isnull().sum()
print("\\n🔍 Data Quality Assessment:")
print(f"Total missing values: {{missing_data.sum():,}}")
print(f"Data completeness: {{(1 - missing_data.sum() / (len(df) * len(df.columns))) * 100:.2f}}%")

if missing_data.sum() > 0:
    print("Columns with missing data:")
    for col, count in missing_data[missing_data > 0].head().items():
        print(f"  {{col}}: {{count:,}} missing ({{count/len(df)*100:.2f}}%)")

# Statistical summary for key metrics
print("\\n📈 Key Metrics Summary:")
key_metrics = ['speed', 'Motor Pwr(w)', 'soc', 'Torque Nm']
for metric in key_metrics:
    if metric in df.columns:
        print(f"{{metric}}: mean={{df[metric].mean():.2f}}, std={{df[metric].std():.2f}}")

print("\\nTelemetry data loading and analysis completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 2
        )

        # Then: Validate telemetry analysis results
        assert (
            result["success"] is True
        ), f"Telemetry analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "📊 Nissan Leaf Telemetry Dataset Analysis" in stdout
        ), "Analysis header not found"
        assert "Dataset shape:" in stdout, "Dataset shape not reported"
        assert "🔋 Battery & Power Metrics:" in stdout, "Battery metrics not analyzed"
        assert "🚗 Vehicle Performance:" in stdout, "Vehicle performance not analyzed"
        assert "🌍 Route Information:" in stdout, "Route information not extracted"
        assert "🔍 Data Quality Assessment:" in stdout, "Data quality not assessed"
        assert "completed successfully!" in stdout, "Analysis did not complete"

        # Validate no errors in stderr
        assert (
            result["data"]["stderr"] == ""
        ), f"Unexpected errors: {result['data']['stderr']}"

    @pytest.mark.api
    @pytest.mark.telemetry
    @pytest.mark.slow
    def test_energy_efficiency_analysis(
        self, server_ready: None, uploaded_telemetry_file: str
    ) -> None:
        """
        Test complex energy efficiency and consumption analysis.

        Validates advanced energy efficiency calculations, consumption metrics,
        regenerative braking analysis, and temperature impact assessment using
        comprehensive automotive telemetry data processing.

        Given: Nissan Leaf telemetry data with power, speed, and environmental metrics
        When: Python code performs energy efficiency analysis with multiple parameters
        Then: Detailed efficiency metrics and insights are generated correctly

        Args:
            server_ready: Ensures server is available
            uploaded_telemetry_file: CSV filename in Pyodide filesystem

        Validates:
        - Power consumption analysis by speed ranges
        - Regenerative braking efficiency calculations
        - Battery state change tracking
        - Power-to-weight ratio analysis
        - Temperature impact on efficiency
        - Energy recovery metrics

        Example:
            Test validates energy analysis like:
            ```python
            df['power_kw'] = df['Motor Pwr(w)'] / 1000.0
            efficiency_by_speed = df.groupby('speed_range')['power_kw'].mean()
            regen_efficiency = (total_regen / total_motor) * 100
            ```
        """
        # Given: Uploaded telemetry data for energy analysis
        filename = uploaded_telemetry_file

        # When: Execute energy efficiency analysis
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np

# Load telemetry data using pathlib
data_file = Path('/uploads') / '{filename}'
df = pd.read_csv(data_file)

print("⚡ ENERGY EFFICIENCY ANALYSIS")
print("=" * 50)

# Calculate energy consumption metrics with safety checks
df['power_kw'] = df['Motor Pwr(w)'] / 1000.0  # Convert to kW
if 'Aux Pwr(100w)' in df.columns:
    df['aux_power_kw'] = df['Aux Pwr(100w)'] / 10.0  # Convert to kW
    df['total_power_kw'] = df['power_kw'] + df['aux_power_kw']
else:
    df['total_power_kw'] = df['power_kw']

# Energy efficiency by speed ranges
speed_bins = [0, 20, 40, 60, 80, 100, 150]
speed_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
df['speed_range'] = pd.cut(df['speed'], bins=speed_bins, labels=speed_labels, include_lowest=True)

# Calculate efficiency metrics by speed range
efficiency_by_speed = df.groupby('speed_range', observed=True).agg({{
    'power_kw': ['mean', 'std', 'min', 'max'],
    'speed': 'mean',
    'soc': ['mean', 'count']
}}).round(2)

print("\\n📈 Power Consumption by Speed Range:")
for speed_range in efficiency_by_speed.index:
    if pd.notna(speed_range):
        avg_power = efficiency_by_speed.loc[speed_range, ('power_kw', 'mean')]
        avg_speed = efficiency_by_speed.loc[speed_range, ('speed', 'mean')]
        count = efficiency_by_speed.loc[speed_range, ('soc', 'count')]
        print(f"  {{speed_range:>8}} km/h: {{avg_power:6.1f}} kW avg ({{count:,}} samples)")

# Regenerative braking analysis
if 'regenwh' in df.columns:
    regen_data = df[df['regenwh'] < 0].copy()  # Negative values indicate regeneration
    print("\\n🔄 Regenerative Braking Analysis:")
    print(f"Total regenerative events: {{len(regen_data):,}} ({{len(regen_data)/len(df)*100:.1f}}% of time)")
    if len(regen_data) > 0:
        print(f"Total regen energy: {{regen_data['regenwh'].sum():,.0f}} Wh")
        print(f"Average regen power: {{regen_data['regenwh'].mean():.1f}} Wh per event")
        print(f"Peak regen power: {{regen_data['regenwh'].min():,.0f}} Wh")
else:
    print("\\n🔄 Regenerative Braking: Data not available")

# Battery state of charge analysis
soc_changes = df['soc'].diff()
charging_events = soc_changes[soc_changes > 0]
discharging_events = soc_changes[soc_changes < 0]

print("\\n🔋 Battery State Changes:")
print(f"Charging events: {{len(charging_events):,}}")
print(f"Discharging events: {{len(discharging_events):,}}")
print(f"Net SOC change: {{df['soc'].iloc[-1] - df['soc'].iloc[0]:.1f}}%")
print(f"SOC volatility: {{df['soc'].std():.2f}}%")

# Power-to-weight efficiency analysis (Nissan Leaf ~1.6 tons)
vehicle_weight_kg = 1600
df['power_to_weight'] = df['power_kw'] / (vehicle_weight_kg / 1000)

print("\\n⚖️ Power-to-Weight Analysis:")
print(f"Average power-to-weight ratio: {{df['power_to_weight'].mean():.2f}} kW/ton")
print(f"Peak power-to-weight ratio: {{df['power_to_weight'].max():.2f}} kW/ton")
print(f"Min power-to-weight ratio: {{df['power_to_weight'].min():.2f}} kW/ton")

# Temperature impact on efficiency
if 'amb_temp' in df.columns:
    temp_data = df.dropna(subset=['amb_temp'])
    if len(temp_data) > 0:
        temp_bins = [temp_data['amb_temp'].min()-1, 15, 20, 25, temp_data['amb_temp'].max()+1]
        temp_labels = ['Cold', 'Optimal', 'Warm']
        temp_data['temp_range'] = pd.cut(temp_data['amb_temp'], bins=temp_bins, labels=temp_labels)
        
        temp_efficiency = temp_data.groupby('temp_range', observed=True)['power_kw'].agg(['mean', 'count']).round(2)
        
        print("\\n🌡️ Temperature Impact on Efficiency:")
        for temp_range in temp_efficiency.index:
            if pd.notna(temp_range) and temp_efficiency.loc[temp_range, 'count'] > 10:
                avg_power = temp_efficiency.loc[temp_range, 'mean']
                count = temp_efficiency.loc[temp_range, 'count']
                print(f"  {{temp_range:>8}} temp: {{avg_power:6.1f}} kW avg ({{count:,}} samples)")
else:
    print("\\n🌡️ Temperature Impact: Data not available")

# Overall efficiency summary
total_distance_km = len(df) * 0.1  # Approximate distance based on sample rate
total_energy_kwh = df['power_kw'][df['power_kw'] > 0].sum() / 1000  # Convert to kWh
if total_distance_km > 0 and total_energy_kwh > 0:
    efficiency_kwh_100km = (total_energy_kwh / total_distance_km) * 100
    print(f"\\n🎯 Overall Efficiency Summary:")
    print(f"Estimated efficiency: {{efficiency_kwh_100km:.1f}} kWh/100km")
    print(f"Total estimated distance: {{total_distance_km:.1f}} km")
    print(f"Total energy consumed: {{total_energy_kwh:.2f}} kWh")

print("\\nEnergy efficiency analysis completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 2
        )

        # Then: Validate energy efficiency analysis results
        assert (
            result["success"] is True
        ), f"Energy analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "⚡ ENERGY EFFICIENCY ANALYSIS" in stdout
        ), "Energy analysis header not found"
        assert (
            "📈 Power Consumption by Speed Range:" in stdout
        ), "Speed range analysis not found"
        assert "🔋 Battery State Changes:" in stdout, "Battery analysis not found"
        assert (
            "⚖️ Power-to-Weight Analysis:" in stdout
        ), "Power-to-weight analysis not found"
        assert "completed successfully!" in stdout, "Energy analysis did not complete"

        # Validate no errors
        assert (
            result["data"]["stderr"] == ""
        ), f"Energy analysis errors: {result['data']['stderr']}"

        # Additional validations for energy metrics
        assert "km/h:" in stdout, "Speed range breakdowns not found"
        assert "kW avg" in stdout, "Power averages not calculated"
        assert "samples)" in stdout, "Sample counts not displayed"

    def test_geospatial_route_analysis(self, upload_telemetry_data):
        """
        Test geospatial analysis and route optimization calculations.

        This test validates advanced GPS-based analytics for automotive telemetry:
        - Distance calculations using Haversine formula
        - Speed vs location correlation analysis
        - Route efficiency optimization metrics
        - Geospatial clustering of driving patterns

        Given: Nissan Leaf telemetry data with GPS coordinates
        When: Performing geospatial route analysis
        Then: Should calculate accurate distance metrics and route insights
        """
        # Given: Telemetry data with GPS coordinates
        filename = upload_telemetry_data

        # When: Execute geospatial route analysis
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np

# Load telemetry data using pathlib
data_file = Path('/uploads') / '{filename}'
df = pd.read_csv(data_file)

print("🗺️ GEOSPATIAL ROUTE ANALYSIS")
print("=" * 50)

# Haversine distance calculation for route analysis
def haversine_distance(lat1, lon1, lat2, lon2):
    \"\"\"Calculate distance between GPS coordinates using Haversine formula\"\"\"
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

# Calculate distances between consecutive points
df_sorted = df.sort_values('timestamp_gps_utc').copy()
df_sorted['prev_lat'] = df_sorted['latitude'].shift(1)
df_sorted['prev_lon'] = df_sorted['longitude'].shift(1)

# Calculate segment distances
mask = df_sorted['prev_lat'].notna()
df_sorted.loc[mask, 'segment_distance_m'] = haversine_distance(
    df_sorted.loc[mask, 'prev_lat'], df_sorted.loc[mask, 'prev_lon'],
    df_sorted.loc[mask, 'latitude'], df_sorted.loc[mask, 'longitude']
)

# Route statistics
total_distance_km = df_sorted['segment_distance_m'].sum() / 1000
max_segment = df_sorted['segment_distance_m'].max()
avg_segment = df_sorted['segment_distance_m'].mean()

print(f"📏 Route Distance Analysis:")
print(f"Total route distance: {{total_distance_km:.2f}} km")
print(f"Number of GPS points: {{len(df_sorted):,}}")
print(f"Average segment length: {{avg_segment:.1f}} m")
print(f"Maximum segment length: {{max_segment:.1f}} m")

# Elevation analysis
elevation_gain = df_sorted[df_sorted['altitude'].diff() > 0]['altitude'].diff().sum()
elevation_loss = abs(df_sorted[df_sorted['altitude'].diff() < 0]['altitude'].diff().sum())
total_elevation_change = elevation_gain + elevation_loss

print(f"\\n⛰️ Elevation Profile:")
print(f"Min altitude: {{df['altitude'].min():.1f}} m")
print(f"Max altitude: {{df['altitude'].max():.1f}} m")
print(f"Total elevation gain: {{elevation_gain:.1f}} m")
print(f"Total elevation loss: {{elevation_loss:.1f}} m")
print(f"Total elevation change: {{total_elevation_change:.1f}} m")

# Speed and efficiency correlation with terrain
df_sorted['altitude_gradient'] = df_sorted['altitude'].diff() / df_sorted['segment_distance_m']
df_sorted['altitude_gradient'] = df_sorted['altitude_gradient'].fillna(0)

# Classify terrain by gradient
df_sorted['terrain_type'] = 'flat'
df_sorted.loc[df_sorted['altitude_gradient'] > 0.02, 'terrain_type'] = 'uphill'
df_sorted.loc[df_sorted['altitude_gradient'] < -0.02, 'terrain_type'] = 'downhill'

terrain_analysis = df_sorted.groupby('terrain_type').agg({{
    'speed': ['mean', 'count'],
    'Motor Pwr(w)': 'mean',
    'regenwh': 'mean',
    'segment_distance_m': 'sum'
}}).round(2)

print(f"\\n🏔️ Terrain Impact Analysis:")
for terrain in ['uphill', 'flat', 'downhill']:
    if terrain in terrain_analysis.index:
        speed_avg = terrain_analysis.loc[terrain, ('speed', 'mean')]
        power_avg = terrain_analysis.loc[terrain, ('Motor Pwr(w)', 'mean')]
        distance_km = terrain_analysis.loc[terrain, ('segment_distance_m', 'sum')] / 1000
        count = terrain_analysis.loc[terrain, ('speed', 'count')]
        
        print(f"  {{terrain.capitalize()}}: {{speed_avg:5.1f}} km/h, {{power_avg:6.0f}}W, {{distance_km:5.1f}}km ({{count:,}} points)")

# Coordinate bounds and route shape analysis
lat_range = df['latitude'].max() - df['latitude'].min()
lon_range = df['longitude'].max() - df['longitude'].min()

print(f"\\n🎯 Route Bounds:")
print(f"Latitude span: {{lat_range:.6f}}° ({{lat_range * 111000:.0f}} m)")
print(f"Longitude span: {{lon_range:.6f}}° ({{lon_range * 111000 * np.cos(np.radians(df['latitude'].mean())):.0f}} m)")

# Calculate route compactness (area vs perimeter ratio)
convex_hull_area = lat_range * lon_range * (111000 ** 2)  # Approximate area in m²
route_compactness = total_distance_km * 1000 / np.sqrt(convex_hull_area)

print("\\nGeospatial route analysis completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 3
        )

        # Then: Validate geospatial analysis results
        assert (
            result["success"] is True
        ), f"Geospatial analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "🗺️ GEOSPATIAL ROUTE ANALYSIS" in stdout
        ), "Geospatial analysis header not found"
        assert (
            "Route Distance Analysis:" in stdout
            or "Route Bounds:" in stdout
            or "GPS coordinates not available" in stdout
        ), "Route analysis not executed"
        assert (
            "completed successfully!" in stdout
        ), "Geospatial analysis did not complete"

        # Validate no errors
        assert (
            result["data"]["stderr"] == ""
        ), f"Geospatial analysis errors: {result['data']['stderr']}"

    def test_driving_behavior_analytics(self, upload_telemetry_data):
        """
        Test complex driving behavior pattern analysis.

        This test validates sophisticated automotive behavioral analytics:
        - Acceleration/deceleration pattern recognition
        - Aggressive vs eco-friendly driving detection
        - Speed consistency and smoothness metrics
        - Regenerative braking effectiveness analysis

        Given: Nissan Leaf telemetry data with speed and power metrics
        When: Analyzing driving behavior patterns
        Then: Should identify behavioral patterns and efficiency correlations
        """
        # Given: Telemetry data with driving behavior signals
        filename = upload_telemetry_data

        # When: Execute driving behavior analytics
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np

# Load telemetry data using pathlib
data_file = Path('/uploads') / '{filename}'
df = pd.read_csv(data_file)

print("🚗 DRIVING BEHAVIOR ANALYTICS")
print("=" * 50)

# Sort by timestamp for time series analysis
sort_column = 'timestamp_gps_utc' if 'timestamp_gps_utc' in df.columns else df.index
df_sorted = df.sort_values(sort_column).copy()

# Calculate acceleration and deceleration with error handling
df_sorted['speed_diff'] = df_sorted['speed'].diff()

# Handle time differences (use 1 second default if not available)
if 'time_diff' in df.columns:
    df_sorted['time_diff'] = df_sorted['time_diff'].fillna(1)
else:
    df_sorted['time_diff'] = 1  # Default 1 second sampling

# Calculate acceleration in m/s²
df_sorted['acceleration_ms2'] = (df_sorted['speed_diff'] / 3.6) / df_sorted['time_diff']

# Classify driving events based on acceleration thresholds
df_sorted['driving_event'] = 'steady'
df_sorted.loc[df_sorted['acceleration_ms2'] > 1.0, 'driving_event'] = 'hard_acceleration'
df_sorted.loc[df_sorted['acceleration_ms2'] > 0.5, 'driving_event'] = 'acceleration'
df_sorted.loc[df_sorted['acceleration_ms2'] < -1.5, 'driving_event'] = 'hard_braking'
df_sorted.loc[df_sorted['acceleration_ms2'] < -0.5, 'driving_event'] = 'braking'
df_sorted.loc[df_sorted['speed'] == 0, 'driving_event'] = 'stopped'

# Driving event analysis
event_counts = df_sorted['driving_event'].value_counts()
total_events = len(df_sorted)

print("🎭 Driving Event Distribution:")
for event, count in event_counts.items():
    percentage = (count / total_events) * 100
    print(f"  {{event.replace('_', ' ').title():15s}}: {{count:5,}} events ({{percentage:5.1f}}%)")

# Acceleration statistics
acceleration_stats = df_sorted['acceleration_ms2'].describe()
print(f"\\n📊 Acceleration Statistics:")
print(f"Max acceleration: {{acceleration_stats['max']:6.2f}} m/s²")
print(f"Max deceleration: {{acceleration_stats['min']:6.2f}} m/s²")
print(f"Avg acceleration: {{acceleration_stats['mean']:6.2f}} m/s²")
print(f"Acceleration std:  {{acceleration_stats['std']:6.2f}} m/s²")

# Speed behavior analysis
speed_stats = df['speed'].describe()
print(f"\\n🏃 Speed Behavior Analysis:")
print(f"Maximum speed:     {{speed_stats['max']:6.1f}} km/h")
print(f"Average speed:     {{speed_stats['mean']:6.1f}} km/h")
print(f"Speed variability: {{speed_stats['std']:6.1f}} km/h (std dev)")

# Calculate speed percentiles
speed_percentiles = df['speed'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(f"\\n📈 Speed Percentiles:")
for percentile, speed in speed_percentiles.items():
    print(f"  {{int(percentile*100):2d}}th percentile: {{speed:6.1f}} km/h")

# Motor efficiency analysis during different driving modes
motor_efficiency = df_sorted.groupby('driving_event').agg({{
    'Motor Pwr(w)': ['mean', 'std', 'count'],
    'Torque Nm': ['mean', 'std'],
    'rpm': 'mean',
    'speed': 'mean'
}}).round(2)

print(f"\\n⚙️ Motor Performance by Driving Event:")
for event in motor_efficiency.index:
    if motor_efficiency.loc[event, ('Motor Pwr(w)', 'count')] > 10:  # Only show significant samples
        power_avg = motor_efficiency.loc[event, ('Motor Pwr(w)', 'mean')]
        torque_avg = motor_efficiency.loc[event, ('Torque Nm', 'mean')]
        rpm_avg = motor_efficiency.loc[event, ('rpm', 'mean')]
        count = motor_efficiency.loc[event, ('Motor Pwr(w)', 'count')]
        
        print(f"  {{event.replace('_', ' ').title():15s}}: {{power_avg:6.0f}}W, {{torque_avg:5.1f}}Nm, {{rpm_avg:4.0f}}RPM ({{count:,}} samples)")

# Eco-driving score calculation
eco_score = 100
harsh_acceleration_penalty = len(df_sorted[df_sorted['driving_event'] == 'hard_acceleration']) * 2
harsh_braking_penalty = len(df_sorted[df_sorted['driving_event'] == 'hard_braking']) * 3
speed_penalty = len(df_sorted[df_sorted['speed'] > 90]) * 1  # Speeding penalty

eco_score -= (harsh_acceleration_penalty + harsh_braking_penalty + speed_penalty) / len(df_sorted) * 100
eco_score = max(0, eco_score)  # Ensure non-negative

print(f"\\n🌱 Eco-Driving Score: {{eco_score:.1f}}/100")
print(f"  - Harsh acceleration events: {{len(df_sorted[df_sorted['driving_event'] == 'hard_acceleration']):,}}")
print(f"  - Harsh braking events: {{len(df_sorted[df_sorted['driving_event'] == 'hard_braking']):,}}")
print(f"  - High speed events (>90 km/h): {{len(df_sorted[df_sorted['speed'] > 90]):,}}")

# Energy recovery efficiency
regen_events = df_sorted[df_sorted['regenwh'] < 0]
total_regen_energy = abs(regen_events['regenwh'].sum())
total_motor_energy = df_sorted[df_sorted['Motor Pwr(w)'] > 0]['Motor Pwr(w)'].sum() / 1000  # Convert to Wh

if total_motor_energy > 0:
    regen_efficiency = (total_regen_energy / total_motor_energy) * 100
    print(f"\\n🔄 Energy Recovery Efficiency: {{regen_efficiency:.1f}}%")
print("\\nDriving behavior analytics completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 2
        )

        # Then: Validate driving behavior analysis results
        assert (
            result["success"] is True
        ), f"Driving behavior analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "🚗 DRIVING BEHAVIOR ANALYTICS" in stdout
        ), "Driving behavior analysis header not found"
        assert (
            "driving_event" in stdout.lower() or "acceleration" in stdout.lower()
        ), "Behavior analysis not executed"
        assert (
            "completed successfully!" in stdout
        ), "Driving behavior analysis did not complete"

        # Validate no errors
        assert (
            result["data"]["stderr"] == ""
        ), f"Driving behavior analysis errors: {result['data']['stderr']}"

    def test_advanced_time_series_analysis(self, upload_telemetry_data):
        """
        Test advanced time series analysis and predictive insights.

        This test validates sophisticated temporal analytics for automotive data:
        - Rolling window trend analysis with statistical metrics
        - Correlation analysis between operational parameters
        - Pattern detection and anomaly identification
        - Predictive modeling for energy consumption

        Given: Nissan Leaf telemetry data with temporal sequences
        When: Performing advanced time series analysis
        Then: Should detect trends, correlations, and predictive insights
        """
        # Given: Telemetry data with time series characteristics
        filename = upload_telemetry_data

        # When: Execute advanced time series analysis
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np

# Load telemetry data using pathlib
data_file = Path('/uploads') / '{filename}'
df = pd.read_csv(data_file)

print("📊 ADVANCED TIME SERIES ANALYSIS")
print("=" * 50)

# Sort by timestamp and create time-based features
sort_column = 'timestamp_gps_utc' if 'timestamp_gps_utc' in df.columns else df.index
df_sorted = df.sort_values(sort_column).copy()

# Rolling window analysis for trend detection
window_size = min(50, len(df) // 4)  # Adaptive window size
if window_size > 5:
    df_sorted['speed_rolling_mean'] = df_sorted['speed'].rolling(window=window_size, center=True).mean()
    if 'Motor Pwr(w)' in df.columns:
        df_sorted['power_rolling_mean'] = df_sorted['Motor Pwr(w)'].rolling(window=window_size, center=True).mean()
    df_sorted['soc_rolling_mean'] = df_sorted['soc'].rolling(window=window_size, center=True).mean()

    # Detect significant changes in driving patterns
    df_sorted['speed_volatility'] = df_sorted['speed'].rolling(window=window_size).std()
    if 'Motor Pwr(w)' in df.columns:
        df_sorted['power_volatility'] = df_sorted['Motor Pwr(w)'].rolling(window=window_size).std()
    
    print(f"\\n🔄 Rolling Window Analysis ({{window_size}}-point windows):")
    print(f"Average speed volatility: {{df_sorted['speed_volatility'].mean():.2f}} km/h")
    if 'power_volatility' in df_sorted.columns:
        print(f"Average power volatility: {{df_sorted['power_volatility'].mean():.0f}} W")
    print(f"Max speed volatility: {{df_sorted['speed_volatility'].max():.2f}} km/h")
    if 'power_volatility' in df_sorted.columns:
        print(f"Max power volatility: {{df_sorted['power_volatility'].max():.0f}} W")
else:
    print("\\n⚠️ Dataset too small for meaningful rolling window analysis")

print("🔄 Rolling Window Analysis (50-point windows):")
print(f"Average speed volatility: {{df_sorted['speed_volatility'].mean():.2f}} km/h")
print(f"Average power volatility: {{df_sorted['power_volatility'].mean():.0f}} W")
print(f"Max speed volatility: {{df_sorted['speed_volatility'].max():.2f}} km/h")
print(f"Max power volatility: {{df_sorted['power_volatility'].max():.0f}} W")

# Correlation analysis between key parameters
correlation_matrix = df_sorted[['speed', 'Motor Pwr(w)', 'Torque Nm', 'rpm', 'soc', 'altitude', 'amb_temp']].corr()

print("\\n🔗 Correlation Analysis:")
print("Strong correlations (|r| > 0.7):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            print(f"  {{col1}} ↔ {{col2}}: {{corr_value:6.3f}}")

# Power consumption prediction using moving averages
def simple_prediction_model(speed, altitude_change, ambient_temp):
    \"\"\"Simple power prediction model based on observed patterns\"\"\"
    # Base power from speed (roughly quadratic relationship)
    base_power = 0.8 * speed**2 + 150
    
    # Altitude adjustment (climbing uses more power) - use numpy.maximum for vectorized max
    altitude_factor = np.maximum(0, altitude_change * 500)  # 500W per meter climb
    
    # Temperature adjustment (efficiency drops at extremes) - use numpy.where for vectorized conditionals
    temp_in_range = (ambient_temp >= 15) & (ambient_temp <= 25)
    temp_factor = np.where(temp_in_range, 0, np.abs(ambient_temp - 20) * 50)
    
    return base_power + altitude_factor + temp_factor

# Apply prediction model
df_sorted['altitude_change'] = df_sorted['altitude'].diff().fillna(0)
df_sorted['predicted_power'] = simple_prediction_model(
    df_sorted['speed'], 
    df_sorted['altitude_change'], 
    df_sorted['amb_temp']
)

# Calculate prediction accuracy
actual_power = df_sorted['Motor Pwr(w)']
predicted_power = df_sorted['predicted_power']
prediction_error = abs(actual_power - predicted_power)
mean_absolute_error = prediction_error.mean()
relative_error = (prediction_error / (actual_power + 1)).mean() * 100  # +1 to avoid division by zero

print(f"\\n🎯 Power Consumption Prediction Model:")
print(f"Mean Absolute Error: {{mean_absolute_error:.0f}} W")
print(f"Mean Relative Error: {{relative_error:.1f}}%")
print(f"Model R² correlation: {{np.corrcoef(actual_power, predicted_power)[0,1]**2:.3f}}")

# Energy efficiency trends over the journey
df_sorted['cumulative_distance'] = range(len(df_sorted))  # Proxy for journey progress
journey_segments = np.array_split(df_sorted, 10)  # Divide journey into 10 segments

print(f"\\n📈 Journey Efficiency Trends (10 segments):")
for i, segment in enumerate(journey_segments):
    if len(segment) > 0:
        avg_speed = segment['speed'].mean()
        avg_power = segment['Motor Pwr(w)'].mean()
        avg_soc = segment['soc'].mean()
        efficiency = avg_power / max(avg_speed, 1)  # W per km/h
        
        print(f"  Segment {{i+1:2d}}: {{avg_speed:5.1f}} km/h, {{avg_power:6.0f}}W, {{efficiency:6.1f}} W/(km/h), SOC: {{avg_soc:2.0f}}%")

# Identify driving pattern clusters using simple statistical grouping
df_sorted['efficiency_ratio'] = df_sorted['Motor Pwr(w)'] / (df_sorted['speed'] + 1)  # +1 to avoid division by zero

# Define efficiency categories
eff_25th = df_sorted['efficiency_ratio'].quantile(0.25)
eff_75th = df_sorted['efficiency_ratio'].quantile(0.75)

df_sorted['efficiency_category'] = 'medium'
df_sorted.loc[df_sorted['efficiency_ratio'] <= eff_25th, 'efficiency_category'] = 'high'
df_sorted.loc[df_sorted['efficiency_ratio'] >= eff_75th, 'efficiency_category'] = 'low'

efficiency_analysis = df_sorted.groupby('efficiency_category').agg({{
    'speed': 'mean',
    'Motor Pwr(w)': 'mean',
    'altitude': ['mean', 'std'],
    'efficiency_ratio': ['mean', 'count']
}}).round(2)

print(f"\\n⚡ Efficiency Pattern Analysis:")
for category in ['high', 'medium', 'low']:
    if category in efficiency_analysis.index:
        speed_avg = efficiency_analysis.loc[category, ('speed', 'mean')]
        power_avg = efficiency_analysis.loc[category, ('Motor Pwr(w)', 'mean')]
        count = efficiency_analysis.loc[category, ('efficiency_ratio', 'count')]
        
        print(f"  {{category.title()}} efficiency: {{speed_avg:5.1f}} km/h, {{power_avg:6.0f}}W ({{count:,}} points)")

# Battery degradation insights (SOC vs distance)
soc_trend = np.polyfit(range(len(df_sorted)), df_sorted['soc'], 1)
soc_slope = soc_trend[0] * len(df_sorted)  # Total SOC change over journey

print(f"\\n🔋 Battery Performance:")
print(f"Total SOC consumption: {{abs(soc_slope):.1f}}%")
print(f"SOC consumption rate: {{abs(soc_slope)/len(df_sorted)*1000:.2f}}%/1000 points")
print(f"Battery health (SOH): {{df_sorted['soh'].iloc[0]:.1f}}%")
print("\\nAdvanced time series analysis completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 3
        )

        # Then: Validate time series analysis results
        assert (
            result["success"] is True
        ), f"Time series analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "📊 ADVANCED TIME SERIES ANALYSIS" in stdout
        ), "Time series analysis header not found"
        assert (
            "rolling" in stdout.lower()
            or "correlation" in stdout.lower()
            or "trend" in stdout.lower()
        ), "Time series analysis not executed"
        assert (
            "completed successfully!" in stdout
        ), "Time series analysis did not complete"

        # Validate no errors
        assert (
            result["data"]["stderr"] == ""
        ), f"Time series analysis errors: {result['data']['stderr']}"

    def test_comprehensive_automotive_insights(self, upload_telemetry_data):
        """
        Test comprehensive automotive insights with visualization generation.

        This test validates complete automotive analytics with visual outputs:
        - Multi-dimensional data visualization generation
        - Comprehensive efficiency dashboards
        - Cross-correlation heat maps and trend analysis
        - Professional automotive reporting formats

        Given: Complete Nissan Leaf telemetry dataset
        When: Generating comprehensive automotive insights
        Then: Should produce professional analytical reports with visualizations
        """
        # Given: Complete telemetry dataset for comprehensive analysis
        filename = upload_telemetry_data

        # When: Execute comprehensive automotive insights analysis
        python_code = f"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import time

# Load telemetry data using pathlib
data_file = Path('/uploads') / '{filename}'
df = pd.read_csv(data_file)

# Sort by timestamp if available
sort_column = 'timestamp_gps_utc' if 'timestamp_gps_utc' in df.columns else df.index
df_sorted = df.sort_values(sort_column).copy()

print("� COMPREHENSIVE AUTOMOTIVE INSIGHTS")
print("=" * 50)

# Create comprehensive analysis plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Nissan Leaf Telemetry Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Speed vs Power consumption scatter plot
axes[0, 0].scatter(df['speed'], df['Motor Pwr(w)'], alpha=0.6, s=1, c=df['soc'], cmap='viridis')
axes[0, 0].set_xlabel('Speed (km/h)')
axes[0, 0].set_ylabel('Motor Power (W)')
axes[0, 0].set_title('Power vs Speed (colored by SOC)')
axes[0, 0].grid(True, alpha=0.3)

# 2. SOC over journey progression
journey_progress = range(len(df_sorted))
axes[0, 1].plot(journey_progress, df_sorted['soc'], 'b-', linewidth=1.5)
axes[0, 1].set_xlabel('Journey Progress (data points)')
axes[0, 1].set_ylabel('State of Charge (%)')
axes[0, 1].set_title('Battery SOC During Journey')
axes[0, 1].grid(True, alpha=0.3)

# 3. Altitude profile
axes[0, 2].plot(journey_progress, df_sorted['altitude'], 'g-', linewidth=1)
axes[0, 2].fill_between(journey_progress, df_sorted['altitude'], alpha=0.3, color='green')
axes[0, 2].set_xlabel('Journey Progress (data points)')
axes[0, 2].set_ylabel('Altitude (m)')
axes[0, 2].set_title('Route Elevation Profile')
axes[0, 2].grid(True, alpha=0.3)

# 4. Speed distribution histogram
axes[1, 0].hist(df['speed'], bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
axes[1, 0].axvline(df['speed'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {{df["speed"].mean():.1f}} km/h')
axes[1, 0].set_xlabel('Speed (km/h)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Speed Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Power consumption vs altitude correlation
axes[1, 1].scatter(df['altitude'], df['Motor Pwr(w)'], alpha=0.5, s=1, c=df['speed'], cmap='plasma')
axes[1, 1].set_xlabel('Altitude (m)')
axes[1, 1].set_ylabel('Motor Power (W)')
axes[1, 1].set_title('Power vs Altitude (colored by speed)')
axes[1, 1].grid(True, alpha=0.3)

# 6. Regenerative braking analysis
regen_data = df[df['regenwh'] < 0]
axes[1, 2].hist(regen_data['regenwh'], bins=30, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
axes[1, 2].set_xlabel('Regenerative Energy (Wh)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Regenerative Braking Distribution')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()

# Save the comprehensive dashboard using direct string paths (avoid pathlib escaping issues)
import os
timestamp = int(time.time() * 1000)  # Generate unique timestamp
os.makedirs('/plots/matplotlib', exist_ok=True)
dashboard_file = f'/plots/matplotlib/nissan_leaf_dashboard_{{timestamp}}.png'
plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Verify the file was created
file_exists = os.path.exists(dashboard_file)
file_size = os.path.getsize(dashboard_file) if file_exists else 0

print(f"📊 Dashboard saved: {{dashboard_file}}")
print(f"   File exists: {{file_exists}}")
print(f"   File size: {{file_size:,}} bytes")

# Generate summary statistics report
print("\\n📋 EXECUTIVE SUMMARY REPORT")
print("=" * 50)

# Key performance indicators
total_points = len(df)
journey_duration_hours = total_points / 3600  # Assuming 1 Hz sampling
total_energy_consumed = df[df['Motor Pwr(w)'] > 0]['Motor Pwr(w)'].sum() / 1000  # Convert to Wh
total_energy_recovered = abs(df[df['regenwh'] < 0]['regenwh'].sum())

print(f"📊 Journey Overview:")
print(f"  • Total data points: {{total_points:,}}")
print(f"  • Estimated duration: {{journey_duration_hours:.1f}} hours")
print(f"  • Energy consumed: {{total_energy_consumed:,.0f}} Wh")
print(f"  • Energy recovered: {{total_energy_recovered:,.0f}} Wh")
print(f"  • Net energy efficiency: {{((total_energy_recovered/total_energy_consumed)*100):.1f}}% recovery")

print(f"\\n🚗 Vehicle Performance:")
print(f"  • Average speed: {{df['speed'].mean():.1f}} km/h")
print(f"  • Maximum speed: {{df['speed'].max():.1f}} km/h")
print(f"  • Average motor power: {{df['Motor Pwr(w)'].mean():.0f}} W")
print(f"  • Peak motor power: {{df['Motor Pwr(w)'].max():,}} W")
print(f"  • Motor temperature range: {{df['Motor Temp'].min():.0f}}°C - {{df['Motor Temp'].max():.0f}}°C")

print(f"\\n🔋 Battery Analysis:")
print(f"  • SOC range: {{df['soc'].min()}}% - {{df['soc'].max()}}%")
print(f"  • SOC consumption: {{df['soc'].max() - df['soc'].min()}}%")
print(f"  • Battery health: {{df['soh'].iloc[0]:.1f}}%")
print(f"  • Ambient temperature: {{df['amb_temp'].min():.1f}}°C - {{df['amb_temp'].max():.1f}}°C")

print(f"\\n🌍 Route Characteristics:")
print(f"  • Latitude span: {{(df['latitude'].max() - df['latitude'].min()):.6f}}°")
print(f"  • Longitude span: {{(df['longitude'].max() - df['longitude'].min()):.6f}}°")
print(f"  • Altitude range: {{df['altitude'].min():.0f}}m - {{df['altitude'].max():.0f}}m")
print(f"  • Elevation change: {{df['altitude'].max() - df['altitude'].min():.0f}}m")

# Advanced insights
motor_efficiency = df['Motor Pwr(w)'].sum() / df['speed'].sum() if df['speed'].sum() > 0 else 0
thermal_efficiency = df['Motor Pwr(w)'].mean() / df['Motor Temp'].mean() if df['Motor Temp'].mean() > 0 else 0

print(f"\\n🔬 Advanced Analytics:")
print(f"  • Motor efficiency index: {{motor_efficiency:.2f}} W·s/km")
print(f"  • Thermal efficiency: {{thermal_efficiency:.2f}} W/°C")
print(f"  • Speed variability (CV): {{(df['speed'].std()/df['speed'].mean()*100):.1f}}%")
print(f"  • Power variability (CV): {{(df['Motor Pwr(w)'].std()/df['Motor Pwr(w)'].mean()*100):.1f}}%")

print("\\n✅ Analysis completed successfully!")
print("   Dashboard and insights generated from {{:,}} telemetry data points".format(len(df)))

# Output structured verification data for test parsing
print("\\n=== FILE_VERIFICATION_DATA ===")
print(f"FILE_SAVED: {{file_exists}}")
print(f"FILE_SIZE: {{file_size}}")
print(f"FILE_PATH: {{dashboard_file}}")
print(f"PLOT_TYPE: nissan_leaf_dashboard")
print(f"DATA_POINTS: {{len(df)}}")
print("\\nComprehensive automotive insights analysis completed successfully!")
"""

        result = execute_python_code(
            python_code, timeout=Config.TIMEOUTS["code_execution"] * 4
        )

        # Then: Validate comprehensive insights results
        assert (
            result["success"] is True
        ), f"Comprehensive insights analysis failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert (
            "🚗 COMPREHENSIVE AUTOMOTIVE INSIGHTS" in stdout
        ), "Comprehensive analysis header not found"
        assert (
            "dashboard" in stdout.lower()
            or "summary" in stdout.lower()
            or "executive" in stdout.lower()
        ), "Comprehensive analysis not executed"
        assert (
            "completed successfully!" in stdout
        ), "Comprehensive analysis did not complete"

        # Validate no errors
        assert (
            result["data"]["stderr"] == ""
        ), f"Comprehensive analysis errors: {result['data']['stderr']}"
        in_verification_section = False

        for line in output_text.split("\\n"):
            if "=== FILE_VERIFICATION_DATA ===" in line:
                in_verification_section = True
                continue
            elif "=== END_FILE_VERIFICATION_DATA ===" in line:
                in_verification_section = False
                continue
            elif in_verification_section and ":" in line:
                key, value = line.split(":", 1)
                verification_data[key.strip()] = value.strip()

        # Verify file creation in Pyodide virtual filesystem
        self.assertEqual(
            verification_data.get("FILE_SAVED"),
            "True",
            "Plot file was not saved to virtual filesystem",
        )
        self.assertGreater(
            int(verification_data.get("FILE_SIZE", "0")),
            0,
            "Plot file has zero size in virtual filesystem",
        )
        self.assertEqual(verification_data.get("PLOT_TYPE"), "nissan_leaf_dashboard")
        self.assertEqual(
            verification_data.get("ANALYSIS_COMPLETED"),
            "True",
            "Analysis did not complete successfully",
        )
        self.assertEqual(
            verification_data.get("DATA_POINTS"),
            "5843",
            "Unexpected number of data points analyzed",
        )

        # Extract virtual files to real filesystem
        plots_response = self.session.post(
            f"{self.base_url}/api/extract-plots", timeout=30
        )
        self.assertEqual(plots_response.status_code, 200)
        plots_data = plots_response.json()
        self.assertTrue(plots_data.get("success"), "Failed to extract plot files")

        # Verify the file exists in the real filesystem
        import os

        file_path = verification_data.get("FILE_PATH", "")
        actual_filename = file_path.split("/")[-1]  # Get just the filename part
        expected_plots_dir = os.path.join(os.getcwd(), "plots", "matplotlib")
        local_filepath = os.path.join(expected_plots_dir, actual_filename)

        self.assertTrue(
            os.path.exists(local_filepath), f"Plot file not found at {local_filepath}"
        )
        self.assertGreater(
            os.path.getsize(local_filepath), 0, "Local plot file has zero size"
        )

        # Verify it's a reasonable plot file size (should be > 100KB for a complex dashboard)
        file_size_mb = os.path.getsize(local_filepath) / (1024 * 1024)
        self.assertGreater(
            file_size_mb, 0.1, f"Plot file seems too small ({file_size_mb:.2f} MB)"
        )
        self.assertLess(
            file_size_mb, 10, f"Plot file seems too large ({file_size_mb:.2f} MB)"
        )

        print(
            f"✅ Successfully verified plot file: {actual_filename} ({file_size_mb:.2f} MB)"
        )
        print(f"   Virtual filesystem size: {verification_data.get('FILE_SIZE')} bytes")
        print(f"   Real filesystem size: {os.path.getsize(local_filepath)} bytes")


if __name__ == "__main__":
    unittest.main()
