import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Generate sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# Create seaborn plot
sns.scatterplot(data=data, x='x', y='y')
plt.title('Seaborn Scatter Plot Test')

# Save to filesystem
plots_dir = Path('/plots/seaborn')
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / 'test_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print('Seaborn plot saved successfully!')
print(f'Plot saved to: {plots_dir / "test_scatter.png"}')