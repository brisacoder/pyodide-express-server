"""
Comprehensive test for Nissan Leaf automotive telemetry data analysis using Pyodide.
This test uploads real automotive signal telemetry data and performs complex
pandas and numpy operations including time series analysis, geospatial calculations,
energy efficiency metrics, and driving behavior analytics.
"""
import json
import time
import unittest
from pathlib import Path

import requests


class NissanLeafTelemetryTestCase(unittest.TestCase):
    """Test complex automotive telemetry data analysis with Pyodide"""
    
    @classmethod
    def setUpClass(cls):
        """One-time setup for telemetry analysis tests"""
        cls.base_url = "http://localhost:3000"
        cls.session = requests.Session()
        cls.test_csv_file = Path("tests/data/DEVRT-NISSAN-LEAF.csv")
        
        # Verify test data exists
        if not cls.test_csv_file.exists():
            raise FileNotFoundError(f"Test data file not found: {cls.test_csv_file}")
            
    def setUp(self):
        """Per-test setup with tracking"""
        self.uploaded_files = []
        self.temp_files = []
        self.start_time = time.time()
        
    def tearDown(self):
        """Comprehensive cleanup after each test"""
        # Clean uploaded files via API
        for filename in self.uploaded_files:
            try:
                self.session.delete(f"{self.base_url}/api/uploaded-files/{filename}", timeout=10)
            except requests.RequestException:
                pass  # File might already be deleted
                
        # Log test duration for performance monitoring
        duration = time.time() - self.start_time
        if duration > 30:  # Log slow tests
            print(f"SLOW TEST: {self._testMethodName} took {duration:.2f}s")
            
    def upload_telemetry_data(self):
        """Upload the Nissan Leaf telemetry CSV file"""
        with open(self.test_csv_file, 'rb') as file:
            files = {'file': ('DEVRT-NISSAN-LEAF.csv', file, 'text/csv')}
            response = self.session.post(f"{self.base_url}/api/upload", files=files, timeout=30)
            
        if response.status_code != 200:
            print(f"Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])
        
        # Extract the Pyodide filename (this is what we use to access the file in Python)
        pyodide_filename = result["file"]["pyodideFilename"]
        # Store the original filename for deletion (actual uploaded file name)
        original_filename = result["file"]["originalName"]
        self.uploaded_files.append(original_filename)
        return pyodide_filename
        
    def test_basic_telemetry_data_loading(self):
        """Test basic CSV loading and data exploration"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the automotive telemetry data
df = pd.read_csv("{filename}")

# Basic data exploration
print(f"📊 Nissan Leaf Telemetry Dataset Analysis")
print(f"Dataset shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")

# Key telemetry metrics
print("\\n🔋 Battery & Power Metrics:")
print(f"State of Charge (SOC) range: {{df['soc'].min()}}% - {{df['soc'].max()}}%")
print(f"Motor Power range: {{df['Motor Pwr(w)'].min():,}}W - {{df['Motor Pwr(w)'].max():,}}W")
print(f"Battery Health (SOH): {{df['soh'].iloc[0]}}%")

print("\\n🚗 Vehicle Performance:")
print(f"Speed range: {{df['speed'].min()}} - {{df['speed'].max()}} km/h")
print(f"Torque range: {{df['Torque Nm'].min()}} - {{df['Torque Nm'].max()}} Nm")
print(f"RPM range: {{df['rpm'].min():,}} - {{df['rpm'].max():,}} RPM")

print("\\n🌍 Route Information:")
print(f"Latitude range: {{df['latitude'].min():.6f}} - {{df['latitude'].max():.6f}}")
print(f"Longitude range: {{df['longitude'].min():.6f}} - {{df['longitude'].max():.6f}}")
print(f"Altitude range: {{df['altitude'].min():.1f}} - {{df['altitude'].max():.1f}} m")

# Data quality check
missing_data = df.isnull().sum()
print("\\n🔍 Data Quality:")
print(f"Total missing values: {{missing_data.sum():,}}")
if missing_data.sum() > 0:
    print("Columns with missing data:")
    print(missing_data[missing_data > 0].head())
        """
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=60)
        
        if response.status_code != 200:
            print(f"Execute failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
        self.assertEqual(response.status_code, 200)
        
        result = response.text
        self.assertIn("Nissan Leaf Telemetry Dataset Analysis", result)
        self.assertIn("Dataset shape:", result)
        self.assertIn("Battery & Power Metrics:", result)
        self.assertIn("Vehicle Performance:", result)
        self.assertIn("Route Information:", result)
        
    def test_energy_efficiency_analysis(self):
        """Test complex energy efficiency and consumption analysis"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load telemetry data
df = pd.read_csv("{filename}")

print("⚡ ENERGY EFFICIENCY ANALYSIS")
print("=" * 50)

# Calculate energy consumption metrics
df['power_kw'] = df['Motor Pwr(w)'] / 1000.0  # Convert to kW
df['aux_power_kw'] = df['Aux Pwr(100w)'] / 10.0  # Convert to kW
df['total_power_kw'] = df['power_kw'] + df['aux_power_kw']

# Energy efficiency by speed ranges
speed_bins = [0, 20, 40, 60, 80, 100, 150]
speed_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
df['speed_range'] = pd.cut(df['speed'], bins=speed_bins, labels=speed_labels, include_lowest=True)

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
        print(f"  {{speed_range}} km/h: {{avg_power:6.1f}} kW avg ({{count:,}} samples)")

# Regenerative braking analysis
regen_data = df[df['regenwh'] < 0].copy()  # Negative values indicate regeneration
print(f"\\n🔄 Regenerative Braking Analysis:")
print(f"Total regenerative events: {{len(regen_data):,}} ({{len(regen_data)/len(df)*100:.1f}}% of time)")
print(f"Total regen energy: {{regen_data['regenwh'].sum():,}} Wh")
print(f"Average regen power: {{regen_data['regenwh'].mean():.1f}} Wh")
print(f"Peak regen power: {{regen_data['regenwh'].min():,}} Wh")

# Battery state analysis
soc_changes = df['soc'].diff()
charging_events = soc_changes[soc_changes > 0]
discharging_events = soc_changes[soc_changes < 0]

print(f"\\n🔋 Battery State Changes:")
print(f"Charging events: {{len(charging_events):,}}")
print(f"Discharging events: {{len(discharging_events):,}}")
print(f"Net SOC change: {{df['soc'].iloc[-1] - df['soc'].iloc[0]}}%")

# Power-to-weight efficiency (assume 1.6 tons for Nissan Leaf)
vehicle_weight_kg = 1600
df['power_to_weight'] = df['power_kw'] / (vehicle_weight_kg / 1000)

print(f"\\n⚖️ Power-to-Weight Analysis:")
print(f"Average power-to-weight ratio: {{df['power_to_weight'].mean():.2f}} kW/ton")
print(f"Peak power-to-weight ratio: {{df['power_to_weight'].max():.2f}} kW/ton")

# Temperature impact on efficiency
temp_bins = [10, 15, 20, 25, 30]
temp_labels = ['10-15°C', '15-20°C', '20-25°C', '25-30°C']
df['temp_range'] = pd.cut(df['amb_temp'], bins=temp_bins, labels=temp_labels, include_lowest=True)

temp_efficiency = df.groupby('temp_range', observed=True)['power_kw'].agg(['mean', 'count']).round(2)
print(f"\\n🌡️ Temperature Impact on Power Consumption:")
for temp_range in temp_efficiency.index:
    if pd.notna(temp_range) and temp_efficiency.loc[temp_range, 'count'] > 10:
        avg_power = temp_efficiency.loc[temp_range, 'mean']
        count = temp_efficiency.loc[temp_range, 'count']
        print(f"  {{temp_range}}: {{avg_power:5.1f}} kW avg ({{count:,}} samples)")
        """
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=60)
        self.assertEqual(response.status_code, 200)
        
        result = response.text
        self.assertIn("ENERGY EFFICIENCY ANALYSIS", result)
        self.assertIn("Power Consumption by Speed Range:", result)
        self.assertIn("Regenerative Braking Analysis:", result)
        self.assertIn("Battery State Changes:", result)
        self.assertIn("Power-to-Weight Analysis:", result)
        
    def test_geospatial_route_analysis(self):
        """Test geospatial analysis and route optimization calculations"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load telemetry data
df = pd.read_csv("{filename}")

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

print(f"Route compactness index: {{route_compactness:.3f}} (lower = more direct)")
        """
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=60)
        self.assertEqual(response.status_code, 200)
        
        result = response.text
        self.assertIn("GEOSPATIAL ROUTE ANALYSIS", result)
        self.assertIn("Route Distance Analysis:", result)
        self.assertIn("Elevation Profile:", result)
        self.assertIn("Terrain Impact Analysis:", result)
        self.assertIn("Route Bounds:", result)
        
    def test_driving_behavior_analytics(self):
        """Test complex driving behavior pattern analysis"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load telemetry data
df = pd.read_csv("{filename}")

print("🚗 DRIVING BEHAVIOR ANALYTICS")
print("=" * 50)

# Sort by timestamp for time series analysis
df_sorted = df.sort_values('timestamp_gps_utc').copy()

# Calculate acceleration and deceleration
df_sorted['speed_diff'] = df_sorted['speed'].diff()
df_sorted['time_diff'] = df_sorted['time_diff'].fillna(1)  # Default 1 second if missing
df_sorted['acceleration_ms2'] = (df_sorted['speed_diff'] / 3.6) / df_sorted['time_diff']  # Convert km/h to m/s²

# Classify driving events
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
    print(f"  Total energy consumed: {{total_motor_energy:,.0f}} Wh")
    print(f"  Total energy recovered: {{total_regen_energy:,.0f}} Wh")
        """
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=60)
        self.assertEqual(response.status_code, 200)
        
        result = response.text
        self.assertIn("DRIVING BEHAVIOR ANALYTICS", result)
        self.assertIn("Driving Event Distribution:", result)
        self.assertIn("Acceleration Statistics:", result)
        self.assertIn("Speed Behavior Analysis:", result)
        self.assertIn("Motor Performance by Driving Event:", result)
        self.assertIn("Eco-Driving Score:", result)
        
    def test_advanced_time_series_analysis(self):
        """Test advanced time series analysis and predictive insights"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load telemetry data
df = pd.read_csv("{filename}")

print("📊 ADVANCED TIME SERIES ANALYSIS")
print("=" * 50)

# Sort by timestamp and create time-based features
df_sorted = df.sort_values('timestamp_gps_utc').copy()

# Rolling window analysis for trend detection
window_size = 50  # 50-point rolling window
df_sorted['speed_rolling_mean'] = df_sorted['speed'].rolling(window=window_size, center=True).mean()
df_sorted['power_rolling_mean'] = df_sorted['Motor Pwr(w)'].rolling(window=window_size, center=True).mean()
df_sorted['soc_rolling_mean'] = df_sorted['soc'].rolling(window=window_size, center=True).mean()

# Detect significant changes in driving patterns
df_sorted['speed_volatility'] = df_sorted['speed'].rolling(window=window_size).std()
df_sorted['power_volatility'] = df_sorted['Motor Pwr(w)'].rolling(window=window_size).std()

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
        """
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=60)
        self.assertEqual(response.status_code, 200)
        
        result = response.text
        self.assertIn("ADVANCED TIME SERIES ANALYSIS", result)
        self.assertIn("Rolling Window Analysis", result)
        self.assertIn("Correlation Analysis:", result)
        self.assertIn("Power Consumption Prediction Model:", result)
        self.assertIn("Journey Efficiency Trends", result)
        self.assertIn("Efficiency Pattern Analysis:", result)
        
    def test_comprehensive_automotive_insights(self):
        """Test comprehensive automotive insights with visualization generation"""
        filename = self.upload_telemetry_data()
        
        python_code = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Load telemetry data
df = pd.read_csv("{filename}")
df_sorted = df.sort_values('timestamp_gps_utc').copy()

print("🚀 COMPREHENSIVE AUTOMOTIVE INSIGHTS")
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
print(f"ANALYSIS_COMPLETED: True")
print("=== END_FILE_VERIFICATION_DATA ===")
        """
        
        
        response = self.session.post(f"{self.base_url}/api/execute-raw",
                                   data=python_code,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=120)
        self.assertEqual(response.status_code, 200)
        
        output_text = response.text
        
        # Verify the analysis sections are present in output
        self.assertIn("COMPREHENSIVE AUTOMOTIVE INSIGHTS", output_text)
        self.assertIn("Dashboard saved:", output_text)
        self.assertIn("EXECUTIVE SUMMARY REPORT", output_text)
        self.assertIn("Journey Overview:", output_text)
        self.assertIn("Vehicle Performance:", output_text)
        self.assertIn("Battery Analysis:", output_text)
        self.assertIn("Route Characteristics:", output_text)
        self.assertIn("Advanced Analytics:", output_text)
        self.assertIn("Analysis completed successfully!", output_text)
        
        # Parse structured verification data from output
        verification_data = {}
        in_verification_section = False
        
        for line in output_text.split('\\n'):
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
        self.assertEqual(verification_data.get("FILE_SAVED"), "True", "Plot file was not saved to virtual filesystem")
        self.assertGreater(int(verification_data.get("FILE_SIZE", "0")), 0, "Plot file has zero size in virtual filesystem")
        self.assertEqual(verification_data.get("PLOT_TYPE"), "nissan_leaf_dashboard")
        self.assertEqual(verification_data.get("ANALYSIS_COMPLETED"), "True", "Analysis did not complete successfully")
        self.assertEqual(verification_data.get("DATA_POINTS"), "5843", "Unexpected number of data points analyzed")
        
        # Extract virtual files to real filesystem
        plots_response = self.session.post(f"{self.base_url}/api/extract-plots", timeout=30)
        self.assertEqual(plots_response.status_code, 200)
        plots_data = plots_response.json()
        self.assertTrue(plots_data.get("success"), "Failed to extract plot files")
        
        # Verify the file exists in the real filesystem
        import os
        file_path = verification_data.get("FILE_PATH", "")
        actual_filename = file_path.split("/")[-1]  # Get just the filename part
        expected_plots_dir = os.path.join(os.getcwd(), "plots", "matplotlib")
        local_filepath = os.path.join(expected_plots_dir, actual_filename)
        
        self.assertTrue(os.path.exists(local_filepath), f"Plot file not found at {local_filepath}")
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local plot file has zero size")
        
        # Verify it's a reasonable plot file size (should be > 100KB for a complex dashboard)
        file_size_mb = os.path.getsize(local_filepath) / (1024 * 1024)
        self.assertGreater(file_size_mb, 0.1, f"Plot file seems too small ({file_size_mb:.2f} MB)")
        self.assertLess(file_size_mb, 10, f"Plot file seems too large ({file_size_mb:.2f} MB)")
        
        print(f"✅ Successfully verified plot file: {actual_filename} ({file_size_mb:.2f} MB)")
        print(f"   Virtual filesystem size: {verification_data.get('FILE_SIZE')} bytes")
        print(f"   Real filesystem size: {os.path.getsize(local_filepath)} bytes")


if __name__ == '__main__':
    unittest.main()
