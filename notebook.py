# [[file:sample.org::*Block 1: Imports and Type Definitions][Block 1: Imports and Type Definitions:1]]
import numpy as np
arry1  = np
import pandas as pd
from typing import List, Dict, Optional, Tuple
# Block 1: Imports and Type Definitions:1 ends here

# [[file:sample.org::*Block 2: Function Definition][Block 2: Function Definition:1]]
def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a list of numbers.

    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary containing mean, median, std, min, max
    """
    arr = np.array(numbers)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

# This function is now in both Jupyter kernel AND notebook.py
# Block 2: Function Definition:1 ends here

# [[file:sample.org::*Block 3: Create Sample Data][Block 3: Create Sample Data:1]]
# Create sample dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'score': [85, 92, 78, 88, 95],
    'category': ['A', 'B', 'A', 'B', 'A']
}

df = pd.DataFrame(data)
print("Sample Data:")
print(df)
# Block 3: Create Sample Data:1 ends here

# [[file:sample.org::*Block 4: Using the Function - PRESS C-c ' HERE!][Block 4: Using the Function - PRESS C-c ' HERE!:1]]
# When editing this block with C-c ':
# 
# 1. Type: calculate_statistics(
#    → LSP shows the function signature from Block 2!
#    → You get parameter hints and return type
#
# 2. Type: stats['
#    → Jupyter shows actual dict keys from runtime!
#    → LSP provides type information
#
# 3. Try: go to definition (gd) on calculate_statistics
#    → It jumps to Block 2's definition!
#    → This works because LSP sees the full file

scores = df['score'].tolist()
stats = calculate_statistics(scores)

print("Statistics for scores:")
for key, value in stats.items():
    print(f"  {key}: {value:.2f}")
# Block 4: Using the Function - PRESS C-c ' HERE!:1 ends here

# [[file:sample.org::*Block 5: Another Function - Testing Cross-Block References][Block 5: Another Function - Testing Cross-Block References:1]]
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze a DataFrame and return comprehensive statistics.
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        Dictionary with various statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    result = {}
    for col in numeric_cols:
        # Notice: We're using calculate_statistics from Block 2!
        result[col] = calculate_statistics(df[col].tolist())
    
    return result

# Function now available to all blocks
# Block 5: Another Function - Testing Cross-Block References:1 ends here

# [[file:sample.org::*Block 6: Using Multiple Functions - PRESS C-c ' HERE TOO!][Block 6: Using Multiple Functions - PRESS C-c ' HERE TOO!:1]]
# When you press C-c ' on THIS block:
# 
# Try typing: analyze_dataframe(
# → LSP shows the signature (even though it's in Block 5)
#
# Try typing: calculate_statistics(
# → LSP shows the signature (even though it's in Block 2)
#
# This is TRUE cross-block LSP awareness!

full_analysis = analyze_dataframe(df)

print("Full DataFrame Analysis:")
for column, stats in full_analysis.items():
    print(f"\n{column}:")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value:.2f}")
# Block 6: Using Multiple Functions - PRESS C-c ' HERE TOO!:1 ends here

# [[file:sample.org::*Block 7: Type Checking Demo][Block 7: Type Checking Demo:1]]
# Uncomment these lines and press C-c ' to see LSP diagnostics:

# This will show a type error from basedpyright:
# wrong_type: int = "not an integer"

# This will show undefined variable:
# print(undefined_variable)

# This will show wrong argument type:
# result = calculate_statistics("not a list")

# LSP catches these BEFORE execution!
# In the org-src buffer, you'll see red underlines
# Block 7: Type Checking Demo:1 ends here

# [[file:sample.org::*Block 8: Demonstrating Jump to Definition][Block 8: Demonstrating Jump to Definition:1]]
# Press C-c ' to edit this block, then:
# 
# 1. Place cursor on 'calculate_statistics'
# 2. Press 'gd' (go to definition)
# 3. It jumps to Block 2!
#
# 4. Place cursor on 'analyze_dataframe'  
# 5. Press 'gd'
# 6. It jumps to Block 5!
#
# This is the power of cross-block LSP awareness

final_scores = calculate_statistics([78, 85, 88, 92, 95])
final_analysis = analyze_dataframe(df)

print("Final Results:")
print(f"Score stats: {final_scores}")
# Block 8: Demonstrating Jump to Definition:1 ends here
