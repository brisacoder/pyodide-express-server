#!/usr/bin/env python3
"""
Script to fix API calls from JSON to plain text format in test file.
"""
import re

def fix_api_calls():
    """Fix all JSON API calls to use plain text format."""
    
    # Read the file
    try:
        with open('tests/test_dynamic_modules_and_execution_robustness_pytest.py', 'r', encoding='utf-8') as f:
            content = f.read()
        print('File read successfully, length:', len(content))
        
        # Replace all json API calls with plain text calls
        original_content = content
        
        # Pattern: json={"code": variable, "timeout": timeout}
        content = re.sub(
            r'json=\{"code":\s*([^,]+),\s*"timeout":\s*([^}]+)\}',
            r'data=\1, headers={"Content-Type": "text/plain"}',
            content
        )
        
        changes = content != original_content
        print('Changes made:', changes)
        
        if changes:
            # Count the changes
            original_matches = len(re.findall(r'json=\{"code":', original_content))
            new_matches = len(re.findall(r'json=\{"code":', content))
            print(f'Original json calls: {original_matches}, Remaining: {new_matches}')
            
            # Write the corrected file
            with open('tests/test_dynamic_modules_and_execution_robustness_pytest_fixed.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print('File fixed and saved as test_dynamic_modules_and_execution_robustness_pytest_fixed.py')
        else:
            print('No changes needed')
            
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    fix_api_calls()