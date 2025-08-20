/**
 * Data Science Example Client
 * 
 * Demonstrates how to use the Pyodide Express Server for data science workflows
 * including matplotlib plotting, pandas data analysis, and package installation.
 *
 * Run the server in another terminal with `npm start` and then execute:
 *   node examples/data-science-client.js
 */

async function main() {
  const baseUrl = 'http://localhost:3000';

  try {
    console.log('🚀 Starting Data Science Example...\n');

    // 1. Check server health
    console.log('1. Checking server health...');
    const healthResponse = await fetch(`${baseUrl}/health`);
    const health = await healthResponse.json();
    console.log('Health status:', health.status);

    // 2. Install additional packages if needed
    console.log('\n2. Installing seaborn package...');
    const installResponse = await fetch(`${baseUrl}/api/install-package`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ package: 'seaborn' }),
    });
    const installResult = await installResponse.json();
    console.log('Package installation:', installResult.message);

    // 3. Create and analyze sample data
    console.log('\n3. Creating sample dataset and performing analysis...');
    const analysisCode = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data
np.random.seed(42)
data = {
    'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 20,
    'value': np.random.normal(50, 15, 120),
    'score': np.random.uniform(0, 100, 120)
}
df = pd.DataFrame(data)

# Basic statistics
print("Dataset shape:", df.shape)
print("\\nBasic statistics:")
print(df.describe())

# Correlation analysis
correlation = df[['value', 'score']].corr()
print("\\nCorrelation matrix:")
print(correlation)

# Create a visualization and save to filesystem
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='value')
plt.title('Value Distribution by Category')
plt.savefig('/plots/seaborn/category_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Plot saved to /plots/seaborn/category_analysis.png")
print("📊 Analysis complete!")`;

    const analysisResponse = await fetch(`${baseUrl}/api/execute-raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: analysisCode,
    });

    const analysisResult = await analysisResponse.json();
    
    if (analysisResult.success) {
      console.log('Analysis Output:');
      console.log(analysisResult.stdout);
    } else {
      console.error('Analysis Error:', analysisResult.error);
    }

    // 4. List available files
    console.log('\n4. Checking available files...');
    const filesResponse = await fetch(`${baseUrl}/api/pyodide-files`);
    const filesResult = await filesResponse.json();
    console.log('Files in Pyodide filesystem:', filesResult.files?.length || 0);

    console.log('\n🎉 Data Science Example completed successfully!');
    console.log('Check the plots/ directory for generated visualizations.');

  } catch (error) {
    console.error('❌ Example failed:', error.message);
  }
}

main();
