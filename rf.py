# Core Python utilities
import os
import time
import json
from math import sqrt

# Data handling
import pandas as pd
import numpy as np

# Data visualization 
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Model Evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

students_file = "./data/studentdras.csv"

# Modify pandas display options for better readability
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 0)            # no line-width limit
pd.set_option('display.expand_frame_repr', False)  # don’t wrap wide frames


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




# # Prepare data
# X_data = students_df.drop('Target', axis=1)
# y_data = students_df['Target']
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# # Hyperparameter tuning
# n_trees = [400, 500, 600]
# sqrt_n_features = sqrt(X_train.shape[1])
# max_depths = [(int)(0.5*sqrt_n_features), (int)(sqrt_n_features), (int)(2*sqrt_n_features)]
# best_model = None
# for trees in n_trees:
#     for depth in max_depths:
#         clf = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=0, oob_score=True)
#         clf.fit(X_train, y_train)
#         oob_score = clf.oob_score_
#         print(f"Trees: {trees}, Max Depth: {depth}, OOB Score: {oob_score}") 


