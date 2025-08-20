/**
 * File Upload Example Client
 * 
 * Demonstrates how to upload CSV files and process them with Python/pandas.
 *
 * Run the server in another terminal with `npm start` and then execute:
 *   node examples/file-upload-client.js
 */

const fs = require('fs');
const path = require('path');

async function main() {
  const baseUrl = 'http://localhost:3000';

  try {
    console.log('📁 Starting File Upload Example...\n');

    // 1. Create a sample CSV file
    console.log('1. Creating sample CSV file...');
    const csvData = `name,age,city,salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,80000
Bob Johnson,35,Chicago,65000
Alice Brown,28,Seattle,70000
Charlie Wilson,32,Boston,72000`;

    const csvPath = path.join(__dirname, 'sample_data.csv');
    fs.writeFileSync(csvPath, csvData);
    console.log('Sample CSV created:', csvPath);

    // 2. Upload the CSV file
    console.log('\n2. Uploading CSV file...');
    const formData = new FormData();
    const fileBlob = new Blob([csvData], { type: 'text/csv' });
    formData.append('csvFile', fileBlob, 'sample_data.csv');

    const uploadResponse = await fetch(`${baseUrl}/api/upload-csv`, {
      method: 'POST',
      body: formData,
    });

    const uploadResult = await uploadResponse.json();
    console.log('Upload result:', uploadResult.message);
    console.log('Uploaded filename:', uploadResult.filename);

    // 3. List uploaded files
    console.log('\n3. Listing uploaded files...');
    const listResponse = await fetch(`${baseUrl}/api/uploaded-files`);
    const listResult = await listResponse.json();
    console.log('Uploaded files:', listResult.files.map(f => f.name));

    // 4. Process the uploaded file with pandas
    console.log('\n4. Processing uploaded file with pandas...');
    const processingCode = `import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
df = pd.read_csv('/uploads/sample_data.csv')

print("📊 Dataset Analysis:")
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nFirst 3 rows:")
print(df.head(3))

print(f"\\n📈 Statistics:")
print(f"Average age: {df['age'].mean():.1f}")
print("Average salary: $" + f"{df['salary'].mean():,.0f}")
print(f"Cities: {df['city'].unique().tolist()}")

# Create a visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=5, alpha=0.7, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.scatter(df['age'], df['salary'], alpha=0.7, color='lightcoral')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('/plots/matplotlib/data_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Analysis complete! Plot saved to /plots/matplotlib/data_analysis.png")`;

    const processResponse = await fetch(`${baseUrl}/api/execute-raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: processingCode,
    });

    const processResult = await processResponse.json();
    
    if (processResult.success) {
      console.log('Processing Output:');
      console.log(processResult.stdout);
    } else {
      console.error('Processing Error:', processResult.error);
    }

    // 5. Get file info
    console.log('\n5. Getting file information...');
    const infoResponse = await fetch(`${baseUrl}/api/file-info/sample_data.csv`);
    const infoResult = await infoResponse.json();
    console.log('File info:', {
      name: infoResult.name,
      size: infoResult.size + ' bytes',
      uploadedAt: infoResult.uploadedAt
    });

    // 6. Cleanup - delete the uploaded file
    console.log('\n6. Cleaning up uploaded file...');
    const deleteResponse = await fetch(`${baseUrl}/api/uploaded-files/sample_data.csv`, {
      method: 'DELETE'
    });
    const deleteResult = await deleteResponse.json();
    console.log('Delete result:', deleteResult.message);

    // Clean up local file
    fs.unlinkSync(csvPath);
    console.log('Local CSV file deleted');

    console.log('\n🎉 File Upload Example completed successfully!');

  } catch (error) {
    console.error('❌ Example failed:', error.message);
    
    // Cleanup on error
    const csvPath = path.join(__dirname, 'sample_data.csv');
    if (fs.existsSync(csvPath)) {
      fs.unlinkSync(csvPath);
    }
  }
}

// Polyfill for FormData and Blob in Node.js
if (typeof FormData === 'undefined') {
  console.log('⚠️  This example requires Node.js 18+ with built-in fetch support.');
  console.log('For older Node.js versions, install node-fetch and form-data packages.');
  process.exit(1);
}

main();
