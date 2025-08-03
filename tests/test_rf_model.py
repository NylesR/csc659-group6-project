"""
Comprehensive test suite for the Random Forest model implementation.
Tests cover data loading, preprocessing, model training, evaluation, and feature importance.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix)
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import warnings

# Import test configuration
from test_config import get_test_settings, check_data_file

# Get test settings
rf_settings = get_test_settings()

class TestRandomForestModel(unittest.TestCase):
    """Test suite for Random Forest model implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic target (3 classes: 0, 1, 2)
        y = np.random.randint(0, 3, n_samples)
        
        # Create DataFrame with feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X_df = pd.DataFrame(X, columns=feature_names)
        self.y_series = pd.Series(y, name='Target')
        
        # Combine into a single DataFrame
        self.test_data = pd.concat([self.X_df, self.y_series], axis=1)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_df, self.y_series, 
            test_size=0.2, 
            random_state=42,
            shuffle=True
        )
        
        # Initialize a basic Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            oob_score=True
        )
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Test that data can be loaded from CSV
        try:
            data_path = rf_settings['data_path']
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreater(len(df), 0)
                self.assertIn(rf_settings['target_column'], df.columns)
        except FileNotFoundError:
            self.skipTest("Data file not found, skipping data loading test")
    
    def test_data_preprocessing(self):
        """Test data preprocessing steps."""
        # Test feature extraction
        X_data = self.test_data.drop(rf_settings['target_column'], axis=1)
        y_data = self.test_data[rf_settings['target_column']]
        
        self.assertEqual(X_data.shape[1], 10)  # Should have 10 features
        self.assertEqual(len(y_data), 100)      # Should have 100 samples
        self.assertIsInstance(X_data, pd.DataFrame)
        self.assertIsInstance(y_data, pd.Series)
    
    def test_train_test_split(self):
        """Test train-test split functionality."""
        X_data = self.test_data.drop(rf_settings['target_column'], axis=1)
        y_data = self.test_data[rf_settings['target_column']]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data,
            test_size=rf_settings['test_size'],
            shuffle=rf_settings['shuffle'],
            random_state=rf_settings['random_state']
        )
        
        # Test split proportions
        expected_train_size = int(len(X_data) * (1 - rf_settings['test_size']))
        expected_test_size = len(X_data) - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)
        
        # Test that train and test sets are disjoint
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
    
    def test_model_initialization(self):
        """Test Random Forest model initialization."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rf_settings['random_state'],
            oob_score=rf_settings['oob_score']
        )
        
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.random_state, rf_settings['random_state'])
        self.assertTrue(model.oob_score)
    
    def test_model_training(self):
        """Test model training functionality."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Test that model has been fitted
        self.assertTrue(hasattr(self.model, 'feature_importances_'))
        self.assertTrue(hasattr(self.model, 'oob_score_'))
        self.assertEqual(len(self.model.feature_importances_), self.X_train.shape[1])
        
        # Test that OOB score is between 0 and 1
        self.assertGreaterEqual(self.model.oob_score_, 0)
        self.assertLessEqual(self.model.oob_score_, 1)
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Test prediction output
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertIsInstance(y_pred, np.ndarray)
        
        # Test that predictions are valid class labels
        unique_classes = np.unique(self.y_train)
        for pred in y_pred:
            self.assertIn(pred, unique_classes)
    
    def test_model_probability_prediction(self):
        """Test model probability prediction functionality."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get probability predictions
        y_proba = self.model.predict_proba(self.X_test)
        
        # Test probability output
        self.assertEqual(y_proba.shape[0], len(self.y_test))
        self.assertEqual(y_proba.shape[1], len(np.unique(self.y_train)))
        
        # Test that probabilities sum to 1 for each sample
        np.testing.assert_array_almost_equal(
            y_proba.sum(axis=1), 
            np.ones(len(self.y_test)), 
            decimal=10
        )
        
        # Test that all probabilities are between 0 and 1
        self.assertTrue(np.all(y_proba >= 0))
        self.assertTrue(np.all(y_proba <= 1))
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=rf_settings['f1_average'])
        precision = precision_score(self.y_test, y_pred, average=rf_settings['f1_average'])
        recall = recall_score(self.y_test, y_pred, average=rf_settings['f1_average'])
        
        # Test metric ranges
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.X_train, 
            self.y_train, 
            cv=rf_settings['cv_folds']
        )
        
        # Test cross-validation output
        self.assertEqual(len(cv_scores), rf_settings['cv_folds'])
        self.assertGreater(cv_scores.mean(), 0)
        self.assertLess(cv_scores.mean(), 1)
        
        # Test that all CV scores are between 0 and 1
        for score in cv_scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Test confusion matrix properties
        self.assertEqual(cm.shape[0], len(np.unique(self.y_test)))
        self.assertEqual(cm.shape[1], len(np.unique(self.y_test)))
        self.assertTrue(np.all(cm >= 0))  # All values should be non-negative
        
        # Test that sum of confusion matrix equals number of test samples
        self.assertEqual(cm.sum(), len(self.y_test))
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Test feature importance properties
        self.assertEqual(len(importances), self.X_train.shape[1])
        self.assertTrue(np.all(importances >= 0))  # All importances should be non-negative
        self.assertAlmostEqual(importances.sum(), 1.0, places=10)  # Should sum to 1
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning functionality."""
        n_trees_options = rf_settings['n_trees_options']
        sqrt_n_features = np.sqrt(self.X_train.shape[1])
        max_depths = [int(multiplier * sqrt_n_features) for multiplier in rf_settings['max_depth_multipliers']]
        
        best_model = None
        best_oob = 0
        
        # Test different hyperparameter combinations
        for trees in n_trees_options:
            for depth in max_depths:
                model = RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=depth,
                    random_state=rf_settings['random_state'],
                    oob_score=rf_settings['oob_score']
                )
                model.fit(self.X_train, self.y_train)
                
                oob_score = model.oob_score_
                
                # Test OOB score properties
                self.assertGreaterEqual(oob_score, 0)
                self.assertLessEqual(oob_score, 1)
                
                # Update best model
                if oob_score > best_oob:
                    best_model = model
                    best_oob = oob_score
        
        # Test that we found a best model
        self.assertIsNotNone(best_model)
        self.assertGreater(best_oob, 0)
    
    def test_data_validation(self):
        """Test data validation checks."""
        # Test with valid data
        self.assertIsInstance(self.X_train, pd.DataFrame)
        self.assertIsInstance(self.y_train, pd.Series)
        
        # Test that X and y have same number of samples
        self.assertEqual(len(self.X_train), len(self.y_train))
        
        # Test that there are no missing values in features
        self.assertFalse(self.X_train.isnull().any().any())
        
        # Test that target variable has valid values
        unique_targets = self.y_train.unique()
        self.assertGreater(len(unique_targets), 0)
    
    def test_model_persistence(self):
        """Test model saving and loading functionality."""
        import joblib
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            joblib.dump(self.model, model_path)
            
            # Load model
            loaded_model = joblib.load(model_path)
            
            # Test that loaded model makes same predictions
            original_pred = self.model.predict(self.X_test)
            loaded_pred = loaded_model.predict(self.X_test)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty dataset
        with self.assertRaises(ValueError):
            empty_model = RandomForestClassifier()
            empty_model.fit(pd.DataFrame(), pd.Series())
        
        # Test with invalid hyperparameters
        with self.assertRaises(ValueError):
            invalid_model = RandomForestClassifier(n_estimators=-1)
            invalid_model.fit(self.X_train, self.y_train)
    
    def test_settings_configuration(self):
        """Test that settings configuration is valid."""
        # Test required settings
        required_settings = [
            'data_path', 'target_column', 'test_size', 'shuffle', 
            'random_state', 'n_trees_options', 'max_depth_multipliers',
            'oob_score', 'cv_folds', 'f1_average'
        ]
        
        for setting in required_settings:
            self.assertIn(setting, rf_settings)
        
        # Test setting value ranges
        self.assertGreater(rf_settings['test_size'], 0)
        self.assertLess(rf_settings['test_size'], 1)
        self.assertGreaterEqual(rf_settings['random_state'], 0)
        self.assertGreater(rf_settings['cv_folds'], 0)
        
        # Test that n_trees_options contains positive values
        for n_trees in rf_settings['n_trees_options']:
            self.assertGreater(n_trees, 0)
        
        # Test that max_depth_multipliers contains positive values
        for multiplier in rf_settings['max_depth_multipliers']:
            self.assertGreater(multiplier, 0)


class TestModelPerformance(unittest.TestCase):
    """Test suite for model performance validation."""
    
    def setUp(self):
        """Set up test fixtures for performance tests."""
        # Create a larger synthetic dataset for performance testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic features with some correlation structure
        X = np.random.randn(n_samples, n_features)
        
        # Create a more realistic target distribution
        # 60% class 0, 25% class 1, 15% class 2
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X_df = pd.DataFrame(X, columns=feature_names)
        self.y_series = pd.Series(y, name='Target')
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_df, self.y_series, 
            test_size=0.2, 
            random_state=42,
            stratify=self.y_series
        )
    
    def test_model_performance_baseline(self):
        """Test that model performance is above random chance."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            oob_score=True
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Test that accuracy is above random chance (1/3 for 3 classes)
        self.assertGreater(accuracy, 0.33)
        
        # Test that OOB score is reasonable
        min_oob_score = rf_settings.get('min_oob_score', 0.2)
        self.assertGreater(model.oob_score_, min_oob_score)
    
    def test_model_consistency(self):
        """Test that model produces consistent results with same random state."""
        model1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model1.fit(self.X_train, self.y_train)
        model2.fit(self.X_train, self.y_train)
        
        y_pred1 = model1.predict(self.X_test)
        y_pred2 = model2.predict(self.X_test)
        
        # Test that predictions are identical
        np.testing.assert_array_equal(y_pred1, y_pred2)
        
        # Test that feature importances are identical
        np.testing.assert_array_equal(
            model1.feature_importances_, 
            model2.feature_importances_
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 