import sys
import platform

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
    'debugpy', 'numpy'
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

"""Test seaborn visualization"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arry1 = np.array([1,2,3])

# Set theme
sns.set_theme(style="whitegrid")

# Generate sample data
np.random.seed(42)
tips_data = {
    'total_bill': np.random.gamma(20, 2, 200),
    'tip': np.random.gamma(3, 0.5, 200),
    'size': np.random.choice([2, 3, 4, 5, 6], 200),
    'day': np.random.choice(['Thu', 'Fri', 'Sat', 'Sun'], 200),
    'time': np.random.choice(['Lunch', 'Dinner'], 200)
}
tips = pd.DataFrame(tips_data)

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution plot
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax1, color='skyblue')
ax1.set_title('Total Bill Distribution', fontsize=12, fontweight='bold')

# Plot 2: Box plot
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax2, palette='Set2')
ax2.set_title('Total Bill by Day', fontsize=12, fontweight='bold')

# Plot 3: Scatter with regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=ax3, scatter_kws={'alpha':0.5})
ax3.set_title('Tip vs Total Bill', fontsize=12, fontweight='bold')

# Plot 4: Violin plot
sns.violinplot(data=tips, x='time', y='total_bill', hue='day', ax=ax4, palette='muted')
ax4.set_title('Total Bill Distribution by Time and Day', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save and display
output_file = 'test_seaborn.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
output_file

"""Test scientific computing capabilities"""
import numpy as np
from scipy import stats, integrate, optimize

print("=" * 60)
print("NUMPY & SCIPY COMPUTATION TESTS")
print("=" * 60)

# NumPy operations
print("\n1. NumPy Array Operations:")
arr = np.random.randn(1000)
print(f"   Mean: {arr.mean():.4f}")
print(f"   Std Dev: {arr.std():.4f}")
print(f"   Min: {arr.min():.4f}")
print(f"   Max: {arr.max():.4f}")

# Statistical tests
print("\n2. Statistical Test (t-test):")
sample1 = np.random.normal(0, 1, 100)
sample2 = np.random.normal(0.5, 1, 100)
t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value:.4f}")

# Integration
print("\n3. Numerical Integration:")
result, error = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"   ∫exp(-x²)dx from -∞ to ∞ = {result:.6f}")
print(f"   Expected: {np.sqrt(np.pi):.6f}")

# Optimization
print("\n4. Function Optimization:")
result = optimize.minimize(lambda x: x**2 + 5*np.sin(x), x0=0)
print(f"   Minimum of x² + 5sin(x) at x = {result.x[0]:.4f}")
print(f"   Function value: {result.fun:.4f}")

print("=" * 60)

"""Test scikit-learn machine learning capabilities"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("=" * 60)
print("MACHINE LEARNING TEST (Random Forest)")
print("=" * 60)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("\nTraining Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTraining Set Size: {len(X_train)}")
print(f"Test Set Size: {len(X_test)}")
print(f"Number of Features: {X.shape[1]}")
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# Feature importance
print("\nTop 5 Most Important Features:")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(indices, 1):
    print(f"   {i}. Feature {idx}: {importances[idx]:.4f}")

print("=" * 60)

"""Test 3D plotting capabilities"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(12, 5))

# Subplot 1: 3D Surface
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Surface: sin(√(x²+y²))', fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Subplot 2: 3D Scatter
ax2 = fig.add_subplot(122, projection='3d')
np.random.seed(42)
n = 500
xs = np.random.randn(n)
ys = np.random.randn(n)
zs = xs**2 + ys**2 + np.random.randn(n) * 0.5
colors = zs

scatter = ax2.scatter(xs, ys, zs, c=colors, cmap='plasma', s=20, alpha=0.6)
ax2.set_title('3D Scatter: z = x² + y² + noise', fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig.colorbar(scatter, ax=ax2, shrink=0.5)

plt.tight_layout()

output_file = 'test_3d_plot.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
output_file

"""Test LSP diagnostics and error handling"""
import sys
from io import StringIO

print("=" * 60)
print("ERROR HANDLING AND DIAGNOSTICS TEST")
print("=" * 60)

# Test 1: Syntax error (commented out to prevent failure)
print("\n1. Testing syntax error detection:")
print("   (LSP should show red squiggles for syntax errors)")
# Uncomment the line below to test:
# this will cause syntax error

# Test 2: Type checking
print("\n2. Testing type hints and diagnostics:")
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 10)
print(f"   add_numbers(5, 10) = {result}")

# This should show a warning if type checking is enabled:
# result = add_numbers("5", "10")  # Type error

# Test 3: Undefined variable
print("\n3. Testing undefined variable detection:")
try:
    print(undefined_variable)
except NameError as e:
    print(f"   ✓ Caught NameError: {e}")

# Test 4: Import error
print("\n4. Testing import error handling:")
try:
    import nonexistent_module
except ImportError as e:
    print(f"   ✓ Caught ImportError: {e}")

# Test 5: Division by zero
print("\n5. Testing runtime errors:")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"   ✓ Caught ZeroDivisionError: {e}")

print("\n" + "=" * 60)
print("All error handling tests completed successfully!")
print("=" * 60)

"""Test asynchronous execution"""
import time
import numpy as np

print("=" * 60)
print("ASYNC EXECUTION TEST")
print("=" * 60)
print("\nThis block should execute asynchronously.")
print("You should be able to interact with Emacs while it runs.\n")

for i in range(5):
    print(f"Iteration {i+1}/5 - Computing...")
    # Simulate heavy computation
    _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
    time.sleep(1)
    print(f"  Completed iteration {i+1}")

print("\n" + "=" * 60)
print("Async execution completed!")
print("=" * 60)

"""Test ipywidgets integration"""
import ipywidgets as widgets
from IPython.display import display

print("=" * 60)
print("IPYWIDGETS TEST")
print("=" * 60)

# Simple widgets
print("\n1. Creating interactive slider widget:")
slider = widgets.IntSlider(
    value=50,
    min=0,
    max=100,
    step=1,
    description='Value:',
    continuous_update=False
)

print("\n2. Creating text input widget:")
text = widgets.Text(
    value='Hello Jupyter!',
    placeholder='Type something',
    description='Text:',
)

print("\n3. Creating button widget:")
button = widgets.Button(
    description='Click Me!',
    button_style='success',
    tooltip='Click to test',
)

def on_button_click(b):
    print(f"Button clicked! Slider value: {slider.value}, Text: {text.value}")

button.on_click(on_button_click)

print("\nWidgets created. Display them with:")
print("  display(slider)")
print("  display(text)")
print("  display(button)")

print("\n" + "=" * 60)

# Note: Widget display might not work in org-mode, but creation should succeed
display(slider)
display(text)
display(button)

"""Test memory usage and performance"""
import numpy as np
import time
import sys

print("=" * 60)
print("MEMORY AND PERFORMANCE TEST")
print("=" * 60)

# Memory test
print("\n1. Memory Allocation Test:")
sizes = [100, 1000, 5000]
for size in sizes:
    start_time = time.time()
    arr = np.random.randn(size, size)
    alloc_time = time.time() - start_time
    memory_mb = arr.nbytes / (1024 * 1024)
    print(f"   {size}x{size} array: {memory_mb:.2f} MB, allocated in {alloc_time:.4f}s")

# Performance test
print("\n2. Matrix Multiplication Performance:")
sizes = [100, 500, 1000]
for size in sizes:
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    start_time = time.time()
    C = A @ B
    mult_time = time.time() - start_time

    print(f"   {size}x{size} matrices: {mult_time:.4f}s")

# Cleanup
print("\n3. Memory cleanup:")
del arr, A, B, C
print("   ✓ Large arrays deleted")

print("\n" + "=" * 60)

"""
Test LSP features interactively:

1. HOVER: Position cursor over function/variable names and press 'K' or use eldoc
   - Try hovering over: pd.DataFrame, np.array, calculate_statistics

2. GOTO DEFINITION: Press 'gd' or 'C-c c d' on a function name
   - Try going to definition of: calculate_statistics

3. FIND REFERENCES: Press 'C-c c R' on a variable/function name
   - Try finding references to: test_data

4. CODE ACTIONS: Press 'C-c c a' to see available code actions
   - Try it on an import statement
"""

import pandas as pd
import numpy as np

def calculate_statistics(data: np.ndarray) -> dict:
    """Calculate basic statistics for array data.

    Args:
        data: Input array

    Returns:
        Dictionary with mean, std, min, max
    """
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

# Create test data
test_data = np.random.randn(1000)

# Use the function - hover over it to see docstring
stats = calculate_statistics(test_data)

# Create dataframe - hover over DataFrame to see documentation
df = pd.DataFrame({
    'values': test_data,
    'squared': test_data ** 2
})

"""Final summary of all tests"""
print("=" * 70)
print(" " * 15 + "JUPYTER INTEGRATION TEST SUMMARY")
print("=" * 70)

tests = [
    "✓ Test 1: Basic Execution and Environment",
    "✓ Test 2: Package Availability",
    "✓ Test 3: LSP Bridge Code Completion",
    "✓ Test 4: Variable Inspection",
    "✓ Test 5: Matplotlib Plotting",
    "✓ Test 6: Seaborn Visualization",
    "✓ Test 7: NumPy & SciPy Computations",
    "✓ Test 8: Machine Learning (scikit-learn)",
    "✓ Test 9: 3D Plotting",
    "✓ Test 10: Error Handling",
    "✓ Test 11: Async Execution",
    "✓ Test 12: IPyWidgets",
    "✓ Test 13: Memory & Performance",
    "✓ Test 14: LSP Features (Hover, Goto)"
]

print("\nTest Results:")
for test in tests:
    print(f"  {test}")

print("\n" + "=" * 70)
print("\nLSP Bridge Features to Verify Manually:")
print("  • Code completion (C-SPC or TAB)")
print("  • Hover documentation (K or eldoc)")
print("  • Go to definition (gd or C-c c d)")
print("  • Find references (C-c c R)")
print("  • Code actions (C-c c a)")
print("  • Diagnostics (should appear in-line for errors)")
print("  • Inlay hints (parameter names, type hints)")

print("\nOrg-babel Features to Verify:")
print("  • :session - shared session across blocks ✓")
print("  • :async - asynchronous execution ✓")
print("  • :results output/value/file - various result types ✓")
print("  • Inline image display ✓")
print("  • org-babel-execute-buffer (SPC l x) ✓")

print("\n" + "=" * 70)
print("If all tests above executed successfully, your Jupyter")
print("integration with LSP Bridge is working correctly!")
print("=" * 70)
