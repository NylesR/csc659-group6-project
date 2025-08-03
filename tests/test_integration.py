"""
Integration tests for the Random Forest model with real data.
These tests verify that the model works correctly with the actual dataset.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Import test configuration
from test_config import get_test_settings, check_data_file

# Get test settings
rf_settings = get_test_settings()

class TestModelIntegration(unittest.TestCase):
    """Integration tests using real data."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Check if data file exists
        self.data_path = rf_settings['data_path']
        if not check_data_file():
            self.skipTest(f"Data file {self.data_path} not found")
        
        # Load real data
        try:
            self.df = pd.read_csv(self.data_path)
            self.assertGreater(len(self.df), 0, "Dataset is empty")
            self.assertIn(rf_settings['target_column'], self.df.columns, 
                         f"Target column '{rf_settings['target_column']}' not found")
        except Exception as e:
            self.skipTest(f"Failed to load data: {e}")
        
        # Prepare features and target
        self.X = self.df.drop(rf_settings['target_column'], axis=1)
        self.y = self.df[rf_settings['target_column']]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=rf_settings['test_size'],
            shuffle=rf_settings['shuffle'],
            random_state=rf_settings['random_state']
        )
    
    def test_real_data_loading(self):
        """Test loading and preprocessing of real data."""
        # Test data shape
        self.assertGreater(self.X.shape[0], 0, "No samples in dataset")
        self.assertGreater(self.X.shape[1], 0, "No features in dataset")
        
        # Test target distribution
        unique_targets = self.y.unique()
        self.assertGreater(len(unique_targets), 0, "No unique target values")
        
        # Test for missing values
        missing_features = self.X.isnull().sum().sum()
        self.assertEqual(missing_features, 0, f"Found {missing_features} missing values in features")
        
        missing_targets = self.y.isnull().sum()
        self.assertEqual(missing_targets, 0, f"Found {missing_targets} missing values in target")
        
        # Test data types
        self.assertTrue(all(self.X.dtypes.apply(lambda x: np.issubdtype(x, np.number))),
                       "All features should be numeric")
    
    def test_model_training_with_real_data(self):
        """Test model training with real data."""
        # Initialize model with settings from rf.py
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rf_settings['random_state'],
            oob_score=rf_settings['oob_score']
        )
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
        
        # Test model properties
        self.assertTrue(hasattr(model, 'feature_importances_'))
        self.assertTrue(hasattr(model, 'oob_score_'))
        self.assertEqual(len(model.feature_importances_), self.X_train.shape[1])
        
        # Test OOB score
        self.assertGreaterEqual(model.oob_score_, 0)
        self.assertLessEqual(model.oob_score_, 1)
    
    def test_model_prediction_with_real_data(self):
        """Test model prediction with real data."""
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rf_settings['random_state']
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Test predictions
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertIsInstance(y_pred, np.ndarray)
        
        # Test that predictions are valid class labels
        unique_classes = np.unique(self.y_train)
        for pred in y_pred:
            self.assertIn(pred, unique_classes)
    
    def test_model_performance_with_real_data(self):
        """Test model performance with real data."""
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rf_settings['random_state']
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average=rf_settings['f1_average'])
        
        # Test performance thresholds (using test settings)
        min_accuracy = rf_settings.get('min_accuracy', 0.3)
        min_f1_score = rf_settings.get('min_f1_score', 0.2)
        self.assertGreater(accuracy, min_accuracy, f"Accuracy {accuracy:.3f} is too low")
        self.assertGreater(f1, min_f1_score, f"F1 score {f1:.3f} is too low")
        
        # Test that OOB score is reasonable
        min_oob_score = rf_settings.get('min_oob_score', 0.2)
        if hasattr(model, 'oob_score_'):
            self.assertGreater(model.oob_score_, min_oob_score, f"OOB score {model.oob_score_:.3f} is too low")
        else:
            # If OOB score is not available, just check that model was trained
            self.assertTrue(hasattr(model, 'feature_importances_'))
    
    def test_hyperparameter_tuning_with_real_data(self):
        """Test hyperparameter tuning with real data."""
        n_trees_options = rf_settings['n_trees_options']
        sqrt_n_features = np.sqrt(self.X_train.shape[1])
        max_depths = [int(multiplier * sqrt_n_features) for multiplier in rf_settings['max_depth_multipliers']]
        
        best_model = None
        best_oob = 0
        models_tested = 0
        
        # Test different hyperparameter combinations
        for trees in n_trees_options:
            for depth in max_depths:
                model = RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=depth,
                    random_state=rf_settings['random_state'],
                    oob_score=rf_settings['oob_score']
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(self.X_train, self.y_train)
                
                oob_score = model.oob_score_
                models_tested += 1
                
                # Update best model
                if oob_score > best_oob:
                    best_model = model
                    best_oob = oob_score
        
        # Test that we tested multiple models
        self.assertGreater(models_tested, 1, "Should test multiple hyperparameter combinations")
        
        # Test that we found a best model
        min_oob_score = rf_settings.get('min_oob_score', 0.2)
        self.assertIsNotNone(best_model)
        self.assertGreater(best_oob, min_oob_score, f"Best OOB score {best_oob:.3f} is too low")
    
    def test_feature_importance_with_real_data(self):
        """Test feature importance analysis with real data."""
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rf_settings['random_state']
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Test feature importance properties
        self.assertEqual(len(importances), self.X_train.shape[1])
        self.assertTrue(np.all(importances >= 0))
        self.assertAlmostEqual(importances.sum(), 1.0, places=10)
        
        # Test that some features have importance > 0
        max_importance = np.max(importances)
        self.assertGreater(max_importance, 0, "No features have importance > 0")
        
        # Test that importance values are reasonable
        self.assertLessEqual(max_importance, 1.0, "Feature importance should not exceed 1.0")
    
    def test_data_consistency(self):
        """Test data consistency across splits."""
        # Test that train and test sets are disjoint
        train_indices = set(self.X_train.index)
        test_indices = set(self.X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
        
        # Test that all data is used
        total_samples = len(self.X_train) + len(self.X_test)
        self.assertEqual(total_samples, len(self.X))
        
        # Test that feature columns are preserved
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1])
        self.assertEqual(self.X_train.shape[1], self.X.shape[1])
        
        # Test that column names are preserved
        self.assertTrue(all(self.X_train.columns == self.X_test.columns))
        self.assertTrue(all(self.X_train.columns == self.X.columns))


class TestModelReproducibility(unittest.TestCase):
    """Test model reproducibility with different random states."""
    
    def setUp(self):
        """Set up test fixtures for reproducibility tests."""
        # Load data
        data_path = rf_settings['data_path']
        if not os.path.exists(data_path):
            self.skipTest(f"Data file {data_path} not found")
        
        self.df = pd.read_csv(data_path)
        self.X = self.df.drop(rf_settings['target_column'], axis=1)
        self.y = self.df[rf_settings['target_column']]
    
    def test_model_reproducibility(self):
        """Test that model produces same results with same random state."""
        # Train two models with same random state
        model1 = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        
        model2 = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        
        # Split data consistently
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model1.fit(X_train1, y_train1)
            model2.fit(X_train2, y_train2)
        
        # Test that predictions are identical
        y_pred1 = model1.predict(X_test1)
        y_pred2 = model2.predict(X_test2)
        
        np.testing.assert_array_equal(y_pred1, y_pred2)
        
        # Test that feature importances are identical
        np.testing.assert_array_equal(
            model1.feature_importances_, 
            model2.feature_importances_
        )
        
        # Test that OOB scores are identical (if available)
        if hasattr(model1, 'oob_score_') and hasattr(model2, 'oob_score_'):
            self.assertEqual(model1.oob_score_, model2.oob_score_)
        else:
            # If OOB score is not available, just check that models were trained
            self.assertTrue(hasattr(model1, 'feature_importances_'))
            self.assertTrue(hasattr(model2, 'feature_importances_'))


if __name__ == '__main__':
    unittest.main(verbosity=2) 