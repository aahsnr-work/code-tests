# [[file:sample.org::+begin_src python :tangle ./analysis.py :comments link :session py][No heading:1]]
import pandas as pd
import numpy as np

def load_data(path):
    """Load CSV data with error handling."""
    return pd.read_csv(path)

def analyze(df):
    """Perform basic statistical analysis."""
    return df.describe()

# Test it
df = load_data('data.csv')
result = analyze(df)
print(result)
# No heading:1 ends here
