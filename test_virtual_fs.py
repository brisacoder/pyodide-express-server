import requests
import time

# Simple test to see if we can save files to virtual filesystem
code = """
import os
print('Current working directory:', os.getcwd())
print('Contents of /:', os.listdir('/'))
print('Contents of /plots:', os.listdir('/plots'))
print('Contents of /plots/matplotlib:', os.listdir('/plots/matplotlib'))

# Try to create a simple text file
with open('/plots/matplotlib/test.txt', 'w') as f:
    f.write('Hello from virtual filesystem')

print('File created. Contents of /plots/matplotlib:', os.listdir('/plots/matplotlib'))

# Check if the file exists and read it back  
file_exists = os.path.exists('/plots/matplotlib/test.txt')
print('File exists:', file_exists)

if file_exists:
    with open('/plots/matplotlib/test.txt', 'r') as f:
        content = f.read()
    print('File content:', content)

# Now try to save a simple matplotlib plot
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 2, 3])

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title('Simple Test Plot')

# Try to save it
plot_path = '/plots/matplotlib/simple_test.png'
print('Attempting to save plot to:', plot_path)
plt.savefig(plot_path)
plt.close()

# Check if it was saved
plot_exists = os.path.exists(plot_path)
plot_size = os.path.getsize(plot_path) if plot_exists else 0
print('Plot file exists:', plot_exists)
print('Plot file size:', plot_size)

print('Final contents of /plots/matplotlib:', os.listdir('/plots/matplotlib'))

result = {"text_file": file_exists, "plot_file": plot_exists, "plot_size": plot_size}
result
"""

print('Testing virtual filesystem access...')
r = requests.post('http://localhost:3000/api/execute', json={'code': code}, timeout=60)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print(f'Success: {data.get("success")}')
    print(f'Result: {data.get("result")}')
    if data.get('stdout'):
        print('STDOUT:')
        print(data.get('stdout'))
    if data.get('stderr'):
        print('STDERR:')
        print(data.get('stderr'))
else:
    print(f'Error: {r.text}')
