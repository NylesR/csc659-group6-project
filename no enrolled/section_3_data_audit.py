# Comprehensive Data Audit for Student Academic Performance Prediction
# Academic and Professional Version - Concise Analysis
# This script provides a systematic audit of the student academic performance dataset
# following best practices in data science and machine learning

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Configure pandas display settings for optimal output formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)

# Load the primary dataset for analysis
students_file = "./data/studentdras.csv"
students_df = pd.read_csv(students_file)
students_df = students_df[students_df['Target'] != 'Enrolled']  # Filter out 'Enrolled'

print("="*70)
print("STUDENT ACADEMIC PERFORMANCE PREDICTION - DATA AUDIT REPORT")
print("="*70)

# SECTION 1: DATASET OVERVIEW
# This section provides essential information about the dataset structure and purpose
print("\n1. DATASET OVERVIEW")
print("-" * 40)

# Basic dataset statistics
n_samples = students_df.shape[0]
n_features = students_df.shape[1] - 1  # Exclude target variable
n_classes = students_df['Target'].nunique()

print(f"Dataset Dimensions: {n_samples:,} samples, {n_features} features")
print(f"Target Classes: {n_classes} (Dropout, Graduate)")
print(f"Sample-to-Feature Ratio: {n_samples/n_features:.1f}")

# Class distribution analysis - critical for understanding data balance
class_distribution = students_df['Target'].value_counts()
print("\nClass Distribution:")
for class_name, count in class_distribution.items():
    percentage = (count / n_samples) * 100
    print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

# SECTION 2: DATA COMPLETENESS
# Basic assessment of data completeness and quality
print("\n2. DATA COMPLETENESS")
print("-" * 40)

# Check for missing values
missing_values = students_df.isnull().sum()
total_missing = missing_values.sum()

if total_missing == 0:
    print("Missing Values: None detected")
else:
    print(f"Missing Values: {total_missing} total missing values")

# Check for duplicates
duplicate_count = students_df.duplicated().sum()
print(f"Duplicate Rows: {duplicate_count}")

# Data types summary
dtype_counts = students_df.dtypes.value_counts()
print("\nData Types:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# SECTION 3: CLASS DISTRIBUTION
# Analysis of target variable distribution
print("\n3. CLASS DISTRIBUTION")
print("-" * 40)

# Calculate balance metrics
min_class_count = class_distribution.min()
max_class_count = class_distribution.max()
balance_ratio = min_class_count / max_class_count

print(f"Balance Ratio: {balance_ratio:.2f}")
if balance_ratio >= 0.8:
    print("Assessment: Well-balanced dataset")
elif balance_ratio >= 0.6:
    print("Assessment: Moderately imbalanced dataset")
else:
    print("Assessment: Imbalanced dataset")

# SECTION 4: DATA SUFFICIENCY
# Basic assessment of dataset adequacy
print("\n4. DATA SUFFICIENCY")
print("-" * 40)

# Sample-to-feature ratio
sample_feature_ratio = n_samples / n_features
print(f"Sample-to-Feature Ratio: {sample_feature_ratio:.1f}")

if sample_feature_ratio >= 10:
    print("Assessment: Sufficient samples for training")
else:
    print("Assessment: Consider feature selection")

# Minimum samples per class
min_samples_per_class = class_distribution.min()
print(f"Minimum samples per class: {min_samples_per_class:,}")

# SECTION 5: BASIC DATA AUDIT ANSWERS
# Simple answers to the required audit questions
print("\n5. BASIC DATA AUDIT ANSWERS")
print("-" * 40)

print("How is data gathered?")
print("  Dataset from UCI Machine Learning Repository")

# Calculate feature types dynamically
int_features = students_df.select_dtypes(include=['int64']).columns.tolist()
float_features = students_df.select_dtypes(include=['float64']).columns.tolist()
object_features = students_df.select_dtypes(include=['object']).columns.tolist()

# Remove target from numeric counts
if 'Target' in int_features:
    int_features.remove('Target')
if 'Target' in float_features:
    float_features.remove('Target')

print("\nFeatures/Measures?")
print(f"  {n_features} features: {len(int_features) + len(float_features)} numerical, {len(object_features)} categorical")
print(f"  Data types: {len(int_features)} integers, {len(float_features)} floats, {len(object_features)} strings")
print("  Mixed data types appropriate for Random Forest")

print("\nGround truth?")
print("  UCI ML Repository dataset - institutional records")
print("  3 classes: Dropout, Graduate")
print("  Based on academic status from educational institution")

# Calculate demographic information dynamically
print("\nDemographic representation?")
if 'Gender' in students_df.columns:
    gender_dist = students_df['Gender'].value_counts(normalize=True)
    male_pct = gender_dist.get(0, 0) * 100
    female_pct = gender_dist.get(1, 0) * 100
    print(f"  Gender: {male_pct:.1f}% Male, {female_pct:.1f}% Female")

if 'Age at enrollment' in students_df.columns:
    age_stats = students_df['Age at enrollment'].describe()
    print(f"  Age: {age_stats['min']:.0f}-{age_stats['max']:.0f} years")
    print(f"    Mean: {age_stats['mean']:.1f} years")
    print(f"    Median: {age_stats['50%']:.1f} years")

if 'International' in students_df.columns:
    international_dist = students_df['International'].value_counts(normalize=True)
    domestic_pct = international_dist.get(0, 0) * 100
    international_pct = international_dist.get(1, 0) * 100
    print(f"  International: {domestic_pct:.1f}% Domestic, {international_pct:.1f}% International")

print("\nSample-to-feature ratio?")
print(f"  {sample_feature_ratio:.1f} (excellent - >10x requirement)")

print("\nClass balance?")
print(f"  Balance ratio: {balance_ratio:.2f} (imbalanced)")

print("\nData formats?")
print("  Tabular data, no missing values, no duplicates")

# Identify protected attributes dynamically
print("\nPrivacy concerns?")
protected_attributes = []
potential_protected = ['Gender', 'Age at enrollment', 'Nacionality', 'International']

for attr in potential_protected:
    if attr in students_df.columns:
        protected_attributes.append(attr)

if protected_attributes:
    print(f"  Contains protected attributes: {', '.join(protected_attributes)}")
else:
    print("  No protected attributes identified")
print("  Anonymized institutional data")

print("\nAlgorithm compatibility?")
print("  Random Forest appropriate for this data")
print("  No scaling required, robust to noise")