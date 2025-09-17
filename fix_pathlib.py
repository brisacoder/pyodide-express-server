#!/usr/bin/env python3
"""
Fix pathlib usage in matplotlib tests for Pyodide compatibility.
Replace pathlib with os.path operations.
"""

import re
import os

def fix_pathlib_in_file(file_path):
    """Fix pathlib usage in a single file"""
    print(f"Fixing pathlib usage in {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace patterns
    patterns = [
        # Import replacement
        (r'from pathlib import Path', 'import os'),
        
        # Basic path creation and mkdir
        (r'plots_dir = Path\(([^)]+)\)', r'plots_dir = \1'),
        (r'plots_dir\.mkdir\(parents=True, exist_ok=True\)', 
         r'os.makedirs(plots_dir, exist_ok=True)'),
        
        # Path joining
        (r'output_path = plots_dir / ([^;]+)', r'output_path = os.path.join(plots_dir, \1)'),
        
        # File existence and size checks
        (r'file_exists = output_path\.exists\(\)', 
         r'file_exists = os.path.exists(output_path)'),
        (r'file_size = output_path\.stat\(\)\.st_size if file_exists else 0',
         r'file_size = os.path.getsize(output_path) if file_exists else 0'),
        
        # savefig calls
        (r'plt\.savefig\(str\(output_path\)', r'plt.savefig(output_path'),
        
        # str() wrapper removal
        (r'"filename": str\(output_path\)', r'"filename": output_path'),
        
        # Parent directory creation
        (r'output_path\.parent\.mkdir\(parents=True, exist_ok=True\)',
         r'os.makedirs(os.path.dirname(output_path), exist_ok=True)'),
        
        # More complex path operations
        (r'Path\(([^)]+)\)\.mkdir\(parents=True, exist_ok=True\)',
         r'os.makedirs(\1, exist_ok=True)'),
        (r'Path\(([^)]+)\)\.exists\(\)',
         r'os.path.exists(\1)'),
    ]
    
    # Apply all patterns
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed pathlib usage in {file_path}")

def main():
    """Main function to fix pathlib usage"""
    file_path = "tests/test_matplotlib_filesystem.py"
    
    if os.path.exists(file_path):
        fix_pathlib_in_file(file_path)
    else:
        print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    main()