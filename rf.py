# Import required libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

# Import scikit-learn modules for Random Forest classification and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, classification_report,
                             roc_curve, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)

# Configuration object containing all settings for the Random Forest analysis
# This makes it easy to modify parameters without changing the main code
rf_settings = {
    # Data configuration - file path and target variable settings
    'data_path': "./data/studentdras.csv",
    'target_column': 'Target',
    'test_size': 0.2,  # 20% of data for testing
    'shuffle': True,    # Randomly shuffle data before splitting
    'random_state': 42, # For reproducible results
    
    # Model hyperparameters for Random Forest tuning
    'n_trees_options': [400, 600, 800],  # Number of trees to try
    'max_depth_multipliers': [0.5, 1.0, 2.0],  # Multipliers for sqrt(n_features) to set max_depth
    'oob_score': True,  # Calculate out-of-bag score for model validation
    
    # Cross-validation settings
    'cv_folds': 3,  # Number of folds for cross-validation
    
    # Evaluation metrics configuration
    'f1_average': 'macro',  # Use macro averaging for multi-class F1 score
    'confusion_matrix_labels': ["Dropout(0)", "Enrolled(1)", "Graduate(2)"],  # Class labels for visualization
    
    # Visualization settings for plots
    'confusion_matrix_figsize': (5, 5),
    'feature_importance_figsize': (12, 9),
    'confusion_matrix_cmap': 'Blues',  # Color map for confusion matrix
    'grid_visible': False  # Whether to show grid in plots
}

# Load the dataset using the configured path
students = rf_settings['data_path']
students_df = pd.read_csv(students)

# Display first few rows to inspect the data structure
students_df.head()

# Prepare features (X) by removing the target column
X_data = students_df.drop(rf_settings['target_column'], axis=1)

# Extract target variable (y) for classification
y_data = students_df[rf_settings['target_column']]

# Split data into training and testing sets
# This ensures we have separate data for training and evaluating the model
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=rf_settings['test_size'], shuffle=rf_settings['shuffle'])

# Print dataset shapes to verify the split
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Hyperparameter tuning: Test different combinations of number of trees and max depth
n_trees = rf_settings['n_trees_options']
sqrt_n_features = sqrt(X_train.shape[1])  # Calculate sqrt of number of features
max_depths = [(int)(multiplier*sqrt_n_features) for multiplier in rf_settings['max_depth_multipliers']]  # Generate max_depth values

# Initialize variables to track the best model
best_model = None
best_oob = 0

# Grid search through all combinations of hyperparameters
for trees in n_trees:
    print("============================================")
    for depth in max_depths:
        # Create and train Random Forest with current hyperparameters
        clf = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=rf_settings['random_state'], oob_score=rf_settings['oob_score'])
        clf.fit(X_train, y_train)
        
        # Get out-of-bag score (unbiased estimate of generalization error)
        oob_score = clf.oob_score_
        
        # Update best model if current model has better OOB score
        if best_oob < oob_score:
            best_model = clf
            best_oob = oob_score
        
        print(f"Trees: {trees}, Max Depth: {depth}, OOB Score: {oob_score:.3f}")

print("============================================")
print(f"Best OOB Score: {best_oob:.3f}")

# Perform cross-validation on the best model using the full dataset
# This gives us a more robust estimate of model performance
cv_scores = cross_val_score(best_model, X_data, y_data, cv=rf_settings['cv_folds'])

# Display cross-validation results
print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.3f}")

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate and display key performance metrics
print(f"F1 Score: {f1_score(y_test, y_pred, average=rf_settings['f1_average'])}")
print(f"Accuracy:{accuracy_score(y_test, y_pred):.3f}")

# Create and display confusion matrix to understand classification performance
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=rf_settings['confusion_matrix_labels']
)

# Plot confusion matrix with custom styling
fig,ax = plt.subplots(figsize=rf_settings['confusion_matrix_figsize'])
cm_display.plot(
    cmap=plt.cm.get_cmap(rf_settings['confusion_matrix_cmap']),
    ax=ax,
    values_format='d'
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.grid(rf_settings['grid_visible'])
plt.show()

# Feature importance analysis using Gini importance (built into Random Forest)
gini_importances = best_model.feature_importances_

# Create DataFrame for Gini importance and sort by importance
gini_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Gini Importance': gini_importances
})
gini_df = gini_df.sort_values(by='Gini Importance', ascending=False)

# Feature importance analysis using Mean Decrease in Accuracy (MDA)
# This method shuffles each feature and measures the drop in accuracy
mda_importances = []
initial_accuracy = accuracy_score(y_test, y_pred)

# For each feature, shuffle its values and measure accuracy drop
for i in range(X_data.shape[1]):
    X_test_copy = X_test.copy()
    # Shuffle the values in the specified feature column
    shuffled_column_values = X_test_copy.iloc[:, i].values.copy()
    np.random.shuffle(shuffled_column_values)
    X_test_copy.iloc[:, i] = shuffled_column_values

    # Calculate accuracy with shuffled feature
    shuff_accuracy = accuracy_score(y_test, best_model.predict(X_test_copy))
    # Importance is the drop in accuracy when feature is shuffled
    mda_importances.append(initial_accuracy - shuff_accuracy)

# Create DataFrame for MDA importance and sort by importance
mda_df = pd.DataFrame({'Feature': X_data.columns, 'Decrease in Accuracy': mda_importances}).sort_values('Decrease in Accuracy', ascending=False)

# Create side-by-side comparison of both feature importance methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=rf_settings['feature_importance_figsize'])

# Plot MDA feature importance
ax1.barh(mda_df['Feature'], mda_df['Decrease in Accuracy'])
ax1.set_title('Feature Importance (MDA)')
ax1.set_xlabel('Mean Decrease Accuracy')

# Plot Gini feature importance
ax2.barh(gini_df['Feature'], gini_df['Gini Importance'])
ax2.set_title('Feature Importance (Gini)')
ax2.set_xlabel('Mean Decrease Gini')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()