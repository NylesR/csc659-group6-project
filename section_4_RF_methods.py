"""
This file displays the versions of all libraries used in the Random Forest junyper notebook.
It also implements a comprehensive data provenance system that captures experimental settings
and results for reproducibility.
"""

# Core imports
import pandas as pd
import numpy as np
import matplotlib
import sklearn
import sys
import json
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

### 1. LIBRARY VERSION TRACKING
# Create a comprehensive table of all library versions used in the project
# This ensures reproducibility by documenting exact package versions

versions_data = {
    'Library': [
        'Python',
        'pandas',
        'numpy', 
        'matplotlib',
        'scikit-learn',
        'math',
        'collections'
    ],
    'Version': [
        sys.version.split()[0],
        pd.__version__,
        np.__version__,
        matplotlib.__version__,
        sklearn.__version__,
        'Python built-in',
        'Python built-in'
    ],
    'Purpose': [
        'Programming language',
        'Data manipulation and analysis',
        'Numerical computations',
        'Data visualization',
        'Machine learning algorithms',
        'Mathematical functions (sqrt)',
        'Container datatypes (Counter - imported but not used)'
    ]
}

# Create DataFrame for better display
versions_df = pd.DataFrame(versions_data)

# Display library versions in table format
print("=" * 60)
print("LIBRARY VERSIONS USED IN RANDOM FOREST IMPLEMENTATION")
print("=" * 60)
print(versions_df.to_string(index=False))
print("=" * 60)

### 2. MARKDOWN TABLE FORMAT FOR DOCUMENTATION
# Create markdown table format for easy inclusion in reports/documentation
print("\n" + "=" * 60)
print("MARKDOWN TABLE FORMAT:")
print("=" * 60)
print("| Library | Version | Purpose |")
print("|---------|---------|---------|")
for _, row in versions_df.iterrows():
    print(f"| {row['Library']} | {row['Version']} | {row['Purpose']} |")
print("=" * 60)

