#!/usr/bin/env python3
import requests

def test_correlation_heatmap():
    """Test the exact code from the failing test"""
    code = '''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

# Create sample dataset with controlled correlations
np.random.seed(42)
n_samples = 200

# Generate base features
feature_1 = np.random.randn(n_samples)
feature_2 = feature_1 * 0.7 + np.random.randn(n_samples) * 0.3  # Strong positive correlation
feature_3 = feature_1 * -0.5 + np.random.randn(n_samples) * 0.5  # Moderate negative correlation
feature_4 = np.random.randn(n_samples)  # Independent feature

data = {
    'feature_1': feature_1,
    'feature_2': feature_2,
    'feature_3': feature_3,
    'feature_4': feature_4
}

df = pd.DataFrame(data)

# Create correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.3f')
plt.title('Feature Correlation Heatmap')

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer,
           format='png',
           dpi=150,
           bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Prepare result with validation data
result = {
    "plot_base64": plot_b64,
    "plot_type": "correlation_heatmap",
    "correlations": {
        "feature_1_feature_2": float(correlation_matrix.loc['feature_1', 'feature_2']),
        "feature_1_feature_3": float(correlation_matrix.loc['feature_1', 'feature_3'])
    },
    "dataset_info": {
        "n_samples": len(df),
        "n_features": len(df.columns)
    }
}

print(f"Generated correlation heatmap with {len(df)} samples")
json.dumps(result)
'''

    try:
        response = requests.post(
            "http://localhost:3000/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Error: {data.get('error')}")
            
            if 'data' in data and data['data']:
                result_data = data['data']
                if 'stdout' in result_data:
                    print(f"Stdout: {result_data['stdout'][:500]}...")
                if 'stderr' in result_data:
                    print(f"Stderr: {result_data['stderr']}")
                if 'result' in result_data:
                    print(f"Result type: {type(result_data['result'])}")
                    if isinstance(result_data['result'], str):
                        print(f"Result length: {len(result_data['result'])}")
                        print(f"Result preview: {result_data['result'][:200]}...")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_correlation_heatmap()