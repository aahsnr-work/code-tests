import sys
import platform

import

print("=" * 60)
print("ENVIRONMENT INFORMATION")
print("=" * 60)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print("=" * 60)

"""Test that all required packages from flake.nix are available"""
required_packages = [
    'scipy', 'pandas', 'sklearn',
    'matplotlib', 'seaborn', 'ipywidgets',
    'debugpy', 'biopython', 'numpy'
]

print("=" * 60)
print("PACKAGE AVAILABILITY CHECK")
print("=" * 60)

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package:15s} - Available")
    except ImportError:
        print(f"✗ {package:15s} - MISSING")
        missing_packages.append(package)

print("=" * 60)
if missing_packages:
    print(f"⚠ Missing packages: {', '.join(missing_packages)}")
else:
    print("✓ All required packages are available!")
print("=" * 60)

import pandas as pd
import numpy as np

# Test LSP completion here:
# Type: pd.<TAB> to see completions
# Type: np.<TAB> to see completions
# Hover over 'DataFrame' to see documentation
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)

"""Test dataframe display and variable inspection"""
import pandas as pd
import numpy as np

# Create test data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Score': [92.5, 88.0, 95.5, 79.0, 91.5],
    'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
}

df = pd.DataFrame(data)
print(df)

"""Test matplotlib inline image display"""
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax1.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax1.set_title('Trigonometric Functions', fontsize=14, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot
np.random.seed(42)
x2 = np.random.randn(100)
y2 = 2 * x2 + np.random.randn(100) * 0.5
ax2.scatter(x2, y2, alpha=0.6, s=50)
ax2.set_title('Scatter Plot with Linear Trend', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save and display
output_file = 'test_matplotlib.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
output_file
