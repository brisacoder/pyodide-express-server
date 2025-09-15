import sys
print(f'Python version: {sys.version}')

try:
    import seaborn as sns
    print(f'✅ seaborn imported successfully, version: {sns.__version__}')
except ImportError as e:
    print(f'❌ seaborn import failed: {e}')

try:
    import matplotlib.pyplot as plt
    print('✅ matplotlib imported successfully')
except ImportError as e:
    print(f'❌ matplotlib import failed: {e}')

try:
    import pandas as pd
    print(f'✅ pandas imported successfully, version: {pd.__version__}')
except ImportError as e:
    print(f'❌ pandas import failed: {e}')

try:
    import numpy as np
    print(f'✅ numpy imported successfully, version: {np.__version__}')
except ImportError as e:
    print(f'❌ numpy import failed: {e}')