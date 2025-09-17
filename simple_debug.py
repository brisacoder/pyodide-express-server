#!/usr/bin/env python3
"""
Simple debug for one failing test
"""
import requests
import json

def test_simple_seaborn():
    """Test what one of the failing tests is actually doing"""
    
    code = '''
# Install seaborn if not available
try:
    import seaborn as sns
    print("Seaborn already available")
except ImportError:
    import micropip
    print("Installing seaborn...")
    await micropip.install(['seaborn', 'matplotlib'])
    import seaborn as sns
    print("Seaborn installed and imported")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
import io
from pathlib import Path

# Create test data for correlation heatmap
np.random.seed(42)
data = pd.DataFrame({
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
    'feature_3': np.random.randn(100),
    'target': np.random.randn(100)
})

# Add some correlation
data['feature_2'] = data['feature_1'] * 0.5 + np.random.randn(100) * 0.5
data['target'] = data['feature_1'] * 0.3 + data['feature_3'] * 0.4 + np.random.randn(100) * 0.6

print(f"Dataset shape: {data.shape}")
print(f"Dataset columns: {list(data.columns)}")

# Calculate correlation matrix
correlation_matrix = data.corr()
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax
)

plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

print(f"Base64 string length: {len(plot_base64)}")
print(f"Base64 preview: {plot_base64[:100]}...")

result = {
    "plot_base64": plot_base64,
    "correlation_stats": {
        "shape": list(correlation_matrix.shape),
        "max_correlation": float(correlation_matrix.abs().max().max()),
        "min_correlation": float(correlation_matrix.abs().min().min())
    }
}

print("Test completed successfully")
'''

    try:
        response = requests.post(
            "http://localhost:3000/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success field: {data.get('success')}")
            print(f"Error field: {data.get('error')}")
            
            if data.get('error'):
                print(f"Error details: {data['error']}")
            
            if 'data' in data:
                result_data = data['data']
                if 'stdout' in result_data:
                    print(f"Stdout: {result_data['stdout']}")
                if 'stderr' in result_data:
                    print(f"Stderr: {result_data['stderr']}")
        else:
            print(f"HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_simple_seaborn()