#!/usr/bin/env python3
import requests

def test_matplotlib_heatmap():
    """Test the matplotlib heatmap approach"""
    code = '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

print("Starting heatmap test...")

# Create simple test data
np.random.seed(42)
data = {
    'a': np.random.randn(50),
    'b': np.random.randn(50)
}
# Make b correlated with a
data['b'] = data['a'] * 0.5 + np.random.randn(50) * 0.5

df = pd.DataFrame(data)
print(f"Data created with shape: {df.shape}")

# Calculate correlation
correlation_matrix = df.corr()
print(f"Correlation calculated: {correlation_matrix}")

# Create simple heatmap
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title('Test Correlation Heatmap')

print("Plot created successfully")

# Save to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

print(f"Base64 plot created, length: {len(plot_b64)}")

result = {
    "plot_base64": plot_b64[:100] + "...",  # Truncate for testing
    "status": "success"
}

print("Test completed successfully!")
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
                    print(f"Result length: {len(str(result_data['result']))}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_matplotlib_heatmap()