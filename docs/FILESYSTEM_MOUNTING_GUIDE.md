# Pyodide Express Server - Direct Filesystem Mounting Guide

## Overview

The Pyodide Express Server implements **true filesystem mounting** as described in the [official Pyodide documentation](https://pyodide.org/en/stable/usage/accessing-files.html). This means that when Python code running in Pyodide creates files in mounted directories, those files appear **directly** in your local filesystem automatically - no API calls required!

## 🎯 Quick Start

When you write Python code in the Pyodide Express Server, you can create files that appear directly on your local filesystem:

```python
# This Python code runs in Pyodide but creates files on your local machine
import matplotlib.pyplot as plt
import os

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('My Plot')

# Save directly to local filesystem - no API calls needed!
plt.savefig('/plots/matplotlib/my_chart.png')
plt.close()

# The file immediately appears at:
# <project-root>/plots/matplotlib/my_chart.png
```

## 📁 Directory Structure & Mounting

### Pre-Created Directories

The following directories are **automatically created** and **mounted** when the server starts:

```
<project-root>/
├── plots/                    # Main mounted directory (/plots in Pyodide)
│   ├── base64/              # For base64-encoded plots (API-based)
│   │   ├── matplotlib/
│   │   └── seaborn/
│   ├── matplotlib/          # ✅ MOUNTED - Direct filesystem access
│   ├── seaborn/            # ✅ MOUNTED - Direct filesystem access  
│   ├── vfs/                # Virtual filesystem (API-based)
│   └── README.md
```

### Mounted Paths

| Pyodide Path | Local Filesystem Path | Purpose |
|--------------|----------------------|---------|
| `/plots` | `<project-root>/plots` | Root mounted directory |
| `/plots/matplotlib` | `<project-root>/plots/matplotlib` | Matplotlib plots |
| `/plots/seaborn` | `<project-root>/plots/seaborn` | Seaborn plots |
| `/plots/` (any subdirectory) | `<project-root>/plots/` (same subdirectory) | Custom directories |

## 🎯 How to Write Python Code for Direct File Creation

### 1. Text Files

```python
import os

# Ensure directory exists (good practice)
os.makedirs("/plots/matplotlib", exist_ok=True)

# Create a text file - appears immediately in local filesystem
with open("/plots/matplotlib/my_data.txt", "w") as f:
    f.write("Hello from Pyodide!\nThis file appears directly on your local machine.")

# File location: <project-root>/plots/matplotlib/my_data.txt
```

### 2. Matplotlib Plots

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Create your plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Sine Wave - Direct File Save')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)

# Ensure directory exists
os.makedirs("/plots/matplotlib", exist_ok=True)

# Save directly to local filesystem
plt.savefig("/plots/matplotlib/sine_wave.png", dpi=150, bbox_inches='tight')
plt.close()

# File appears at: <project-root>/plots/matplotlib/sine_wave.png
```

### 3. Seaborn Plots

```python
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Create seaborn plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='x', y='y', hue='category', style='category', s=100)
plt.title('Seaborn Scatter Plot - Direct File Save')

# Ensure directory exists
os.makedirs("/plots/seaborn", exist_ok=True)

# Save directly to local filesystem
plt.savefig("/plots/seaborn/scatter_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# File appears at: <project-root>/plots/seaborn/scatter_plot.png
```

### 4. Multiple Files

```python
import os
import json
import pandas as pd

# Create directory for your files
os.makedirs("/plots/data_output", exist_ok=True)

# Create multiple files
files_created = []

# Text file
with open("/plots/data_output/summary.txt", "w") as f:
    f.write("Data Analysis Summary\n")
    f.write("====================\n")
    f.write("Files created: 3\n")
files_created.append("summary.txt")

# JSON file
data = {"results": [1, 2, 3, 4, 5], "mean": 3.0, "std": 1.58}
with open("/plots/data_output/results.json", "w") as f:
    json.dump(data, f, indent=2)
files_created.append("results.json")

# CSV file
df = pd.DataFrame(data["results"], columns=["values"])
df.to_csv("/plots/data_output/data.csv", index=False)
files_created.append("data.csv")

print(f"Created {len(files_created)} files in /plots/data_output/")
# All files appear at: <project-root>/plots/data_output/
```

## 🛠️ Best Practices

### 1. Always Use `os.makedirs()`

```python
import os

# Always ensure the directory exists before writing files
os.makedirs("/plots/matplotlib", exist_ok=True)

# Then create your file
with open("/plots/matplotlib/myfile.txt", "w") as f:
    f.write("Content")
```

### 2. Use Absolute Paths from `/plots`

```python
# ✅ GOOD - Use absolute paths starting with /plots
plt.savefig("/plots/matplotlib/chart.png")

# ❌ AVOID - Relative paths may not work as expected
plt.savefig("chart.png")
```

### 3. Choose Appropriate Subdirectories

```python
# ✅ GOOD - Organize by library/purpose
"/plots/matplotlib/line_chart.png"      # For matplotlib plots
"/plots/seaborn/heatmap.png"           # For seaborn plots
"/plots/data_export/results.csv"       # For data files
"/plots/reports/summary.txt"           # For text reports

# ✅ ALSO GOOD - Organize by project/analysis
"/plots/project_alpha/visualization.png"
"/plots/experiment_1/data.json"
```

### 4. Handle File Extensions Properly

```python
# Different file types
plt.savefig("/plots/matplotlib/chart.png")     # PNG images
plt.savefig("/plots/matplotlib/chart.pdf")     # PDF documents
plt.savefig("/plots/matplotlib/chart.svg")     # SVG vector graphics

# Data files
df.to_csv("/plots/data/output.csv")            # CSV data
df.to_json("/plots/data/output.json")          # JSON data
df.to_excel("/plots/data/output.xlsx")         # Excel files (if openpyxl available)
```

## 🔍 Verification & Debugging

### Check if Files Were Created

```python
import os

# Check if file exists in Pyodide's view
file_path = "/plots/matplotlib/my_chart.png"
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f"✅ File created successfully: {file_path} ({file_size} bytes)")
else:
    print(f"❌ File not found: {file_path}")

# List directory contents
print("Files in /plots/matplotlib:")
for file in os.listdir("/plots/matplotlib"):
    print(f"  - {file}")
```

### Verify on Local Filesystem

After running your Python code, check your local filesystem:

```bash
# Windows
dir "C:\path\to\your\project\plots\matplotlib"

# Linux/Mac
ls -la /path/to/your/project/plots/matplotlib
```

## ⚠️ Important Notes

### Directory Creation
- The **main directories** (`/plots`, `/plots/matplotlib`, `/plots/seaborn`) are **automatically created** by the server
- **Subdirectories** you create (like `/plots/my_analysis`) will be **automatically created** in the local filesystem when you write files
- You **don't need to manually create directories** on the local filesystem

### File Permissions
- Files are created with standard user permissions
- The server process must have write access to the `plots` directory

### Path Separators
- Always use **forward slashes** (`/`) in Python code, even on Windows
- Pyodide handles the path conversion automatically

### Performance
- File creation is **immediate** - no delays or buffering
- Large files (like high-resolution plots) appear as soon as Python finishes writing them

## 🧪 Testing Your Setup

You can test if filesystem mounting is working with this simple script:

```python
import os
import time

# Test file creation
test_file = f"/plots/test_mount_{int(time.time())}.txt"

# Create directory (if needed)
os.makedirs("/plots", exist_ok=True)

# Write test file
with open(test_file, "w") as f:
    f.write(f"Mount test successful at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Verify in Pyodide
if os.path.exists(test_file):
    size = os.path.getsize(test_file)
    print(f"✅ Test successful! File created: {test_file} ({size} bytes)")
    print("Check your local plots/ directory - the file should be there!")
else:
    print("❌ Test failed - file not created")
```

## 🏆 Examples Repository

For complete working examples, see the test files:
- `tests/test_direct_filesystem_mount.py` - Comprehensive test examples
- `tests/test_matplotlib_filesystem.py` - Matplotlib-specific examples  
- `tests/test_seaborn_filesystem.py` - Seaborn-specific examples

## 🔧 Troubleshooting

### Files Not Appearing?

1. **Check the path**: Ensure you're using `/plots/` prefix
2. **Verify mounting**: Look for "✅ Mount successful" in server logs
3. **Check permissions**: Ensure server has write access to plots directory
4. **Server restart**: If mounting fails, restart the server

### Server Logs

Watch server startup logs for mounting confirmation:
```
{"message":"Setting up filesystem mounting for plots..."}
{"message":"✅ Mount successful"}
{"message":"✅ Filesystem mounting setup successfully"}
```

If you see these messages, filesystem mounting is working correctly!

---

## Summary

With Pyodide Express Server's filesystem mounting:

- ✅ **Write normal Python file operations** using `/plots/` paths
- ✅ **Files appear immediately** in your local `plots/` directory
- ✅ **No API calls needed** - direct filesystem access
- ✅ **Works with all file types** - images, data, text, etc.
- ✅ **Automatic directory creation** for subdirectories
- ✅ **True Node.js mounting** as per Pyodide documentation

**Just write Python code as you normally would, and your files will appear on your local machine automatically!** 🚀
