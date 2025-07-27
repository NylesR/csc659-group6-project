# Core Python utilities
import os
import time
import json
from math import sqrt
from collections import Counter

# Data handling
import pandas as pd
import numpy as np

# Data visualization 
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Hyperparameter tuning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Model Evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score

students_file = "./data/studentdras.csv"

# Modify pandas display options for better readability
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 0)            # no line-width limit
pd.set_option('display.expand_frame_repr', False)  # don't wrap wide frames


### 2. Original Database audit
students_df = pd.read_csv(students_file)

# 1. Basic counts (Samples, Features, Classes)
summary = pd.DataFrame([{
    "Samples":  students_df.shape[0],               # samples avaible
    "Features": students_df.shape[1] - 1,           # minus the label column
    "Classes":  students_df["Target"].nunique()     # number of unique classes
}])


print("\n## Original Set Summary")
print(summary.to_markdown(index=False))

# Display first few rows of the dataset
print("\n## Dataset Preview")
print(students_df.head())

# Prepare data
X_data = students_df.drop('Target', axis=1)
y_data = students_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)

print("\n## Data Split Information")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Hyperparameter tuning
n_trees = [400, 600, 800]
sqrt_n_features = sqrt(X_train.shape[1])
max_depths = [(int)(0.5*sqrt_n_features), (int)(sqrt_n_features), (int)(2*sqrt_n_features)]
best_model = None
best_oob = 0

print("\n## Hyperparameter Tuning")
print("============================================")
for trees in n_trees:
    for depth in max_depths:
        clf = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42, oob_score=True)
        clf.fit(X_train, y_train)
        oob_score = clf.oob_score_
        if best_oob < oob_score:
            best_model = clf
            best_oob = oob_score
        print(f"Trees: {trees}, Max Depth: {depth}, OOB Score: {oob_score:.3f}")

print("============================================")
print(f"Best OOB Score: {best_oob:.3f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X_data, y_data, cv=3)

print("\n## Model Evaluation")
print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.3f}")

y_pred = best_model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Dropout(0)","Enrolled(1)", "Graduate(2)"]
)

fig, ax = plt.subplots(figsize=(5,5))
cm_display.plot(
    cmap=plt.cm.Blues,
    ax=ax,
    values_format='d'
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.grid(False)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance Analysis
print("\n## Feature Importance Analysis")

# Gini Importance
gini_importances = best_model.feature_importances_

gini_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Gini Importance': gini_importances
})

gini_df = gini_df.sort_values(by='Gini Importance', ascending=False)

# Mean Decrease Accuracy (MDA)
mda_importances = []
initial_accuracy = accuracy_score(y_test, y_pred)
for i in range(X_data.shape[1]):
    X_test_copy = X_test.copy()
    # Shuffle the values in the specified feature column
    shuffled_column_values = X_test_copy.iloc[:, i].values.copy()
    np.random.shuffle(shuffled_column_values)
    X_test_copy.iloc[:, i] = shuffled_column_values

    shuff_accuracy = accuracy_score(y_test, best_model.predict(X_test_copy))
    mda_importances.append(initial_accuracy - shuff_accuracy)

mda_df = pd.DataFrame({'Feature': X_data.columns, 'Decrease in Accuracy': mda_importances}).sort_values('Decrease in Accuracy', ascending=False)

# Plot Feature Importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))

ax1.barh(mda_df['Feature'], mda_df['Decrease in Accuracy'])
ax1.set_title('Feature Importance (MDA)')
ax1.set_xlabel('Mean Decrease Accuracy')

ax2.barh(gini_df['Feature'], gini_df['Gini Importance'])
ax2.set_title('Feature Importance (Gini)')
ax2.set_xlabel('Mean Decrease Gini')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top features
print("\n## Top 10 Features by Gini Importance")
print(gini_df.head(10).to_string(index=False))

print("\n## Top 10 Features by Mean Decrease Accuracy")
print(mda_df.head(10).to_string(index=False))

print("\nLets go Team 6!")



