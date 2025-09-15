# Quick Reference: Direct Filesystem Mounting

## 🚀 Quick Start

```python
# Files created with these paths appear directly in your local filesystem:

# Text files
with open("/home/pyodide/plots/my_data.txt", "w") as f:
    f.write("Hello World!")
# → Appears at: <project>/pyodide_data/plots/my_data.txt

# Matplotlib plots  
import matplotlib.pyplot as plt
plt.plot([1,2,3], [1,4,2])
plt.savefig("/home/pyodide/plots/matplotlib/chart.png")
# → Appears at: <project>/pyodide_data/plots/matplotlib/chart.png

# Seaborn plots
import seaborn as sns
sns.scatterplot(x=[1,2,3], y=[1,4,2])
plt.savefig("/home/pyodide/plots/seaborn/scatter.png")
# → Appears at: <project>/pyodide_data/plots/seaborn/scatter.png

# Custom directories (auto-created)
with open("/home/pyodide/plots/my_analysis/results.csv", "w") as f:
    f.write("value,score\n1,100\n")
# → Appears at: <project>/pyodide_data/plots/my_analysis/results.csv
```

## 📁 Mounted Directories

| Pyodide Path | Local Path | Auto-Created |
|--------------|------------|--------------|
| `/home/pyodide/uploads` | `<project>/pyodide_data/uploads` | ✅ Yes |
| `/home/pyodide/plots` | `<project>/pyodide_data/plots` | ✅ Yes |
| `/home/pyodide/plots/matplotlib` | `<project>/pyodide_data/plots/matplotlib` | ✅ Yes |
| `/home/pyodide/plots/seaborn` | `<project>/pyodide_data/plots/seaborn` | ✅ Yes |
| `/home/pyodide/plots/your_folder` | `<project>/pyodide_data/plots/your_folder` | ✅ Yes (when you write files) |

## ✅ Best Practices

1. **Always use `/plots/` prefix**: `/plots/matplotlib/my_chart.png`
2. **Use `os.makedirs()`**: `os.makedirs("/plots/data", exist_ok=True)`  
3. **Forward slashes only**: `/plots/file.txt` (not `\plots\file.txt`)
4. **Check file creation**: `os.path.exists("/plots/file.txt")`

## 🧪 Test Script

```python
import os, time
test_file = f"/plots/test_{int(time.time())}.txt"
with open(test_file, "w") as f:
    f.write("Mount test successful!")
print(f"✅ Check your local plots/ directory for: {test_file}")
```

## 📖 Full Guide

See `docs/FILESYSTEM_MOUNTING_GUIDE.md` for comprehensive documentation.
