#!/usr/bin/env python3
"""
Comprehensive Visualization Test Suite

Tests inline image display in org-babel jupyter-python blocks
with matplotlib and seaborn. All functions return filenames
for :results file blocks.

Usage in org-mode:
    #+begin_src jupyter-python :results file
    from test_visualizations import create_basic_plot
    create_basic_plot()
    #+end_src
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# Basic Plots
# ==============================================================================

def create_basic_plot(filename: str = 'basic_plot.png') -> str:
    """Create a simple line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
    ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
    
    ax.set_title('Basic Line Plot', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_scatter_plot(filename: str = 'scatter_plot.png') -> str:
    """Create a scatter plot with colors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5
    colors = x + y
    sizes = np.abs(x * 100)
    
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Color Scale')
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax.set_title('Scatter Plot with Trend Line', fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_histogram(filename: str = 'histogram.png') -> str:
    """Create histogram with KDE overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(3, 1.5, 500)
    ])
    
    ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add KDE
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    ax.plot(xs, density(xs), 'r-', linewidth=2, label='KDE')
    
    ax.set_title('Histogram with Kernel Density Estimate', fontsize=16, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


# ==============================================================================
# Statistical Plots (Seaborn)
# ==============================================================================

def create_boxplot(filename: str = 'boxplot.png') -> str:
    """Create box plot comparing distributions."""
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'Group': np.repeat(['A', 'B', 'C', 'D'], 100),
        'Value': np.concatenate([
            np.random.normal(10, 2, 100),
            np.random.normal(12, 3, 100),
            np.random.normal(11, 2.5, 100),
            np.random.normal(13, 2, 100)
        ])
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='Group', y='Value', ax=ax, palette='Set2')
    sns.swarmplot(data=data, x='Group', y='Value', ax=ax, color='black', alpha=0.3, size=3)
    
    ax.set_title('Box Plot with Swarm Overlay', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Group', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_violinplot(filename: str = 'violinplot.png') -> str:
    """Create violin plot."""
    np.random.seed(42)
    data = pd.DataFrame({
        'Category': np.repeat(['Low', 'Medium', 'High'], 200),
        'Value': np.concatenate([
            np.random.gamma(2, 2, 200),
            np.random.gamma(5, 1.5, 200),
            np.random.gamma(8, 1, 200)
        ])
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=data, x='Category', y='Value', ax=ax, palette='muted', inner='box')
    
    ax.set_title('Violin Plot Showing Distribution Shape', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_heatmap(filename: str = 'heatmap.png') -> str:
    """Create correlation heatmap."""
    np.random.seed(42)
    # Generate correlated data
    n = 100
    data = pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n),
        'C': np.random.randn(n),
        'D': np.random.randn(n),
        'E': np.random.randn(n)
    })
    data['F'] = data['A'] * 0.8 + np.random.randn(n) * 0.2
    data['G'] = data['B'] * 0.6 + data['C'] * 0.4 + np.random.randn(n) * 0.2
    
    # Calculate correlation matrix
    corr = data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_pairplot(filename: str = 'pairplot.png') -> str:
    """Create pair plot for multivariate analysis."""
    np.random.seed(42)
    n = 150
    
    # Generate data with three classes
    data = []
    for i, (mean_x, mean_y) in enumerate([(0, 0), (3, 3), (0, 3)]):
        class_data = pd.DataFrame({
            'Feature1': np.random.normal(mean_x, 1, n),
            'Feature2': np.random.normal(mean_y, 1, n),
            'Feature3': np.random.normal(mean_x + mean_y, 1.5, n),
            'Class': f'Class_{i}'
        })
        data.append(class_data)
    
    data = pd.concat(data, ignore_index=True)
    
    # Create pair plot
    g = sns.pairplot(data, hue='Class', palette='Set1', diag_kind='kde', 
                     plot_kws={'alpha': 0.6, 's': 30}, height=2.5)
    g.fig.suptitle('Pair Plot - Multivariate Distribution', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


# ==============================================================================
# Advanced Plots
# ==============================================================================

def create_subplots_grid(filename: str = 'subplots_grid.png') -> str:
    """Create a grid of different plot types."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    np.random.seed(42)
    
    # Plot 1: Line plot
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    ax1.plot(x, np.cos(x), 'r-', label='cos(x)', linewidth=2)
    ax1.set_title('A) Trigonometric Functions', fontweight='bold', loc='left')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(20, 100, 5)
    bars = ax2.bar(categories, values, color=plt.cm.viridis(np.linspace(0, 1, 5)))
    ax2.set_title('B) Bar Chart', fontweight='bold', loc='left')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Scatter
    x = np.random.randn(100)
    y = np.random.randn(100)
    colors = np.random.rand(100)
    ax3.scatter(x, y, c=colors, s=50, alpha=0.6, cmap='plasma')
    ax3.set_title('C) Scatter Plot', fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram
    data = np.random.randn(1000)
    ax4.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data.mean():.2f}')
    ax4.set_title('D) Histogram', fontweight='bold', loc='left')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Multi-Panel Visualization', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_3d_surface(filename: str = 'surface_3d.png') -> str:
    """Create 3D surface plot."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # Create data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.sin(np.sqrt(X**2 + Y**2))
    Z2 = np.exp(-(X**2 + Y**2) / 10) * np.cos(X) * np.sin(Y)
    
    # Plot 1: Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
    ax1.set_title('3D Surface: sin(√(x²+y²))', fontweight='bold', pad=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Plot 2: Different surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
    ax2.set_title('3D Surface: Gaussian × Waves', fontweight='bold', pad=20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_contour_plot(filename: str = 'contour_plot.png') -> str:
    """Create contour plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2)) + 0.5 * np.exp(-((X-1)**2 + (Y-1)**2))
    
    # Filled contours
    contourf = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax1.set_title('Filled Contour Plot', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(contourf, ax=ax1)
    
    # Line contours
    contour = ax2.contour(X, Y, Z, levels=15, colors='black', linewidths=1, alpha=0.4)
    contourf2 = ax2.contourf(X, Y, Z, levels=15, cmap='RdYlBu_r', alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Contour Plot with Labels', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(contourf2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_time_series(filename: str = 'time_series.png') -> str:
    """Create time series plot with confidence bands."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Generate time series data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.randn(365) * 5
    values = trend + seasonal + noise
    
    # Calculate moving average and std
    window = 30
    rolling_mean = pd.Series(values).rolling(window=window).mean()
    rolling_std = pd.Series(values).rolling(window=window).std()
    
    # Plot
    ax.plot(dates, values, 'o-', alpha=0.3, markersize=2, label='Actual Values')
    ax.plot(dates, rolling_mean, 'r-', linewidth=2, label=f'{window}-day Moving Average')
    
    # Confidence bands
    ax.fill_between(dates, 
                    rolling_mean - 2*rolling_std,
                    rolling_mean + 2*rolling_std,
                    alpha=0.2, color='red', label='95% Confidence Band')
    
    ax.set_title('Time Series with Moving Average and Confidence Bands', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def create_polar_plot(filename: str = 'polar_plot.png') -> str:
    """Create polar plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                   subplot_kw=dict(projection='polar'))
    
    # Plot 1: Spiral
    theta = np.linspace(0, 8*np.pi, 1000)
    r = theta
    ax1.plot(theta, r, linewidth=2)
    ax1.set_title('Spiral Pattern', fontweight='bold', pad=20)
    ax1.grid(True)
    
    # Plot 2: Rose pattern
    theta = np.linspace(0, 2*np.pi, 1000)
    r = np.abs(np.sin(5*theta))
    ax2.plot(theta, r, linewidth=2, color='red')
    ax2.fill(theta, r, alpha=0.3, color='red')
    ax2.set_title('Rose Pattern (5 petals)', fontweight='bold', pad=20)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


# ==============================================================================
# Comprehensive Test Function
# ==============================================================================

def run_all_visualizations(prefix: str = 'test_') -> dict:
    """
    Run all visualization tests and return dictionary of filenames.
    
    Args:
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary mapping test names to filenames
    """
    results = {}
    
    print("=" * 70)
    print("RUNNING VISUALIZATION TESTS")
    print("=" * 70)
    
    tests = [
        ('Basic Line Plot', lambda: create_basic_plot(f'{prefix}basic_plot.png')),
        ('Scatter Plot', lambda: create_scatter_plot(f'{prefix}scatter_plot.png')),
        ('Histogram', lambda: create_histogram(f'{prefix}histogram.png')),
        ('Box Plot', lambda: create_boxplot(f'{prefix}boxplot.png')),
        ('Violin Plot', lambda: create_violinplot(f'{prefix}violinplot.png')),
        ('Heatmap', lambda: create_heatmap(f'{prefix}heatmap.png')),
        ('Pair Plot', lambda: create_pairplot(f'{prefix}pairplot.png')),
        ('Subplots Grid', lambda: create_subplots_grid(f'{prefix}subplots_grid.png')),
        ('3D Surface', lambda: create_3d_surface(f'{prefix}surface_3d.png')),
        ('Contour Plot', lambda: create_contour_plot(f'{prefix}contour_plot.png')),
        ('Time Series', lambda: create_time_series(f'{prefix}time_series.png')),
        ('Polar Plot', lambda: create_polar_plot(f'{prefix}polar_plot.png')),
    ]
    
    for name, test_func in tests:
        try:
            print(f"Creating {name}...", end=' ')
            filename = test_func()
            results[name] = filename
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"✗ Error: {e}")
            results[name] = None
    
    print("=" * 70)
    print(f"✓ Completed {len([r for r in results.values() if r])} of {len(tests)} tests")
    print("=" * 70)
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Visualization Test Suite")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - create_basic_plot()")
    print("  - create_scatter_plot()")
    print("  - create_histogram()")
    print("  - create_boxplot()")
    print("  - create_violinplot()")
    print("  - create_heatmap()")
    print("  - create_pairplot()")
    print("  - create_subplots_grid()")
    print("  - create_3d_surface()")
    print("  - create_contour_plot()")
    print("  - create_time_series()")
    print("  - create_polar_plot()")
    print("  - run_all_visualizations()")
    print("\nUsage in org-mode jupyter-python blocks:")
    print("  #+begin_src jupyter-python :results file")
    print("  from test_visualizations import create_basic_plot")
    print("  create_basic_plot()")
    print("  #+end_src")
    print("=" * 70)
