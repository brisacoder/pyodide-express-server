#!/usr/bin/env python3
import requests

def test_simple_correlation():
    """Test just the correlation part without plotting"""
    code = '''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

print("Starting correlation test...")

# Create sample dataset with controlled correlations
np.random.seed(42)
n_samples = 50  # Smaller dataset

print("Creating dataset...")

# Generate base features
feature_1 = np.random.randn(n_samples)
feature_2 = feature_1 * 0.7 + np.random.randn(n_samples) * 0.3
feature_3 = feature_1 * -0.5 + np.random.randn(n_samples) * 0.5
feature_4 = np.random.randn(n_samples)

data = {
    'feature_1': feature_1,
    'feature_2': feature_2,
    'feature_3': feature_3,
    'feature_4': feature_4
}

df = pd.DataFrame(data)
print(f"Dataset created with shape: {df.shape}")

# Calculate correlation matrix
correlation_matrix = df.corr()
print(f"Correlation matrix calculated")

# Prepare result without plotting
result = {
    "plot_type": "correlation_test",
    "correlations": {
        "feature_1_feature_2": float(correlation_matrix.loc['feature_1', 'feature_2']),
        "feature_1_feature_3": float(correlation_matrix.loc['feature_1', 'feature_3'])
    },
    "dataset_info": {
        "n_samples": len(df),
        "n_features": len(df.columns)
    }
}

print(f"Generated correlation data with {len(df)} samples")
print(f"Result: {result}")
json.dumps(result)
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
            print(f"Success: {data.get('success')}")
            print(f"Error: {data.get('error')}")
            
            if 'data' in data and data['data']:
                result_data = data['data']
                if 'stdout' in result_data:
                    print(f"Stdout: {result_data['stdout']}")
                if 'stderr' in result_data:
                    print(f"Stderr: {result_data['stderr']}")
                if 'result' in result_data:
                    print(f"Result: {result_data['result']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_simple_correlation()