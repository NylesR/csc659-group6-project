"""
Test configuration for Random Forest model tests.
This file provides test-specific settings that work from the Test folder.
"""

import sys
import os

# Add the src directory to path to import the model
sys.path.append('../src/models')

# Define test settings without importing the original file
# This avoids the data loading issue when importing from the original rf.py
test_settings = {
    # Data configuration - file path and target variable settings
    'data_path': "../data/studentdras.csv",
    'target_column': 'Target',
    'test_size': 0.3,  # 30% for testing to get more test data
    'shuffle': True,    # Randomly shuffle data before splitting
    'random_state': 42, # For reproducible results
    
    # Model hyperparameters for Random Forest tuning (reduced for faster testing)
    'n_trees_options': [50, 100],  # Reduced for faster testing
    'max_depth_multipliers': [0.5, 1.0, 2.0],  # Multipliers for sqrt(n_features) to set max_depth
    'oob_score': True,  # Calculate out-of-bag score for model validation
    
    # Cross-validation settings (reduced for faster testing)
    'cv_folds': 2,  # Reduced for faster testing
    
    # Evaluation metrics configuration
    'f1_average': 'macro',  # Use macro averaging for multi-class F1 score
    'confusion_matrix_labels': ["Dropout(0)", "Enrolled(1)", "Graduate(2)"],  # Class labels for visualization
    
    # Visualization settings for plots
    'confusion_matrix_figsize': (5, 5),
    'feature_importance_figsize': (12, 9),
    'confusion_matrix_cmap': 'Blues',  # Color map for confusion matrix
    'grid_visible': False,  # Whether to show grid in plots
    
    # Test-specific performance thresholds
    'min_accuracy': 0.3,  # Lower threshold for testing
    'min_f1_score': 0.2,  # Lower threshold for testing
    'min_oob_score': 0.2,  # Lower threshold for testing
}



# Function to get test settings
def get_test_settings():
    """Get test-specific settings."""
    return test_settings

# Function to check if data file exists
def check_data_file():
    """Check if the data file exists and is accessible."""
    data_path = test_settings['data_path']
    if os.path.exists(data_path):
        return True
    else:
        print(f"Warning: Data file not found at {data_path}")
        return False 