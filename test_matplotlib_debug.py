import requests

# Get the matplotlib code from test_api.py
matplotlib_code = """
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate test data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Use pathlib for plot file handling
plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

# Create unique filename with timestamp
timestamp = int(time.time())
plot_file = plots_dir / f'pathlib_test_{timestamp}.png'

# Create and save plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Pathlib Matplotlib Test')
plt.xlabel('X values')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig(plot_file, dpi=100, bbox_inches='tight')
plt.close()

print(f'Plot saved successfully: {plot_file}')
print(f'File exists: {plot_file.exists()}')
""".strip()

# Execute the code
response = requests.post(
    "http://localhost:3000/api/execute-raw",
    headers={"Content-Type": "text/plain"},
    data=matplotlib_code
)

# Print the full response
import json
print(json.dumps(response.json(), indent=2))