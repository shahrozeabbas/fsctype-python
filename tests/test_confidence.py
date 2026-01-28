"""Unit tests for confidence calculation functions."""

import pytest
import numpy as np
from fsctype.confidence import gap_confidence_batch, entropy_confidence_batch, predict_labels
import pandas as pd


class TestGapConfidence:
    """Tests for gap_confidence_batch function."""
    
    def test_single_cell_type(self):
        """Single cell type should return confidence of 1.0."""
        scores = np.array([[5.0]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == 1.0
        assert conf.shape == (1,)
    
    def test_clear_winner(self):
        """Clear winner should have high confidence."""
        scores = np.array([[10.0, 1.0, 0.5]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == pytest.approx(0.9, rel=0.01)
        assert 0.0 <= conf[0] <= 1.0
    
    def test_tied_scores(self):
        """Tied scores should return 0.0 confidence."""
        scores = np.array([[5.0, 5.0, 5.0]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == 0.0
    
    def test_close_scores(self):
        """Close scores should have low confidence."""
        scores = np.array([[10.0, 9.5, 1.0]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == pytest.approx(0.05, rel=0.01)
    
    def test_batch_processing(self):
        """Should process multiple cells correctly."""
        scores = np.array([
            [10.0, 1.0],
            [5.0, 5.0],
            [8.0, 2.0],
        ])
        conf = gap_confidence_batch(scores)
        assert len(conf) == 3
        assert conf[0] > conf[1]  # First has clear winner
        assert conf[1] == 0.0  # Second is tied
        assert conf[2] > conf[1]  # Third has some separation
    
    def test_negative_scores(self):
        """Negative best score should return 0.0 confidence."""
        scores = np.array([[-5.0, -10.0]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == 0.0
    
    def test_zero_best_score(self):
        """Zero best score should return 0.0 confidence."""
        scores = np.array([[0.0, 0.0, 0.0]])
        conf = gap_confidence_batch(scores)
        assert conf[0] == 0.0


class TestEntropyConfidence:
    """Tests for entropy_confidence_batch function."""
    
    def test_single_cell_type(self):
        """Single cell type should return confidence of 1.0."""
        scores = np.array([[5.0]])
        conf = entropy_confidence_batch(scores)
        assert conf[0] == 1.0
        assert conf.shape == (1,)
    
    def test_uniform_distribution(self):
        """Uniform distribution should have low confidence."""
        scores = np.array([[1.0, 1.0, 1.0]])
        conf = entropy_confidence_batch(scores)
        assert conf[0] == pytest.approx(0.0, abs=0.01)
    
    def test_clear_winner(self):
        """Clear winner should have high confidence."""
        scores = np.array([[10.0, 1.0, 0.5]])
        conf = entropy_confidence_batch(scores)
        assert conf[0] > 0.7
        assert 0.0 <= conf[0] <= 1.0
    
    def test_temperature_effect(self):
        """Lower temperature should increase confidence."""
        scores = np.array([[10.0, 5.0, 1.0]])
        conf_low = entropy_confidence_batch(scores, temperature=0.5)
        conf_high = entropy_confidence_batch(scores, temperature=2.0)
        assert conf_low > conf_high
    
    def test_batch_processing(self):
        """Should process multiple cells correctly."""
        scores = np.array([
            [10.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [8.0, 2.0, 0.1],
        ])
        conf = entropy_confidence_batch(scores)
        assert len(conf) == 3
        assert conf[0] > conf[1]  # First has clear winner
        assert conf[1] == pytest.approx(0.0, abs=0.01)  # Second is uniform
        assert conf[2] > conf[1]  # Third has some separation
    
    def test_epsilon_parameter(self):
        """Epsilon should prevent numerical issues."""
        scores = np.array([[10.0, 0.0, 0.0]])
        conf = entropy_confidence_batch(scores, epsilon=1e-10)
        assert np.isfinite(conf[0])
        assert 0.0 <= conf[0] <= 1.0


class TestPredictLabels:
    """Tests for predict_labels function."""
    
    def test_basic_prediction(self):
        """Basic prediction should work correctly."""
        scores = pd.DataFrame({
            'TypeA': [10.0, 1.0],
            'TypeB': [1.0, 10.0],
        }, index=['CELL_0', 'CELL_1'])
        
        predictions = predict_labels(
            scores,
            ['TypeA', 'TypeB'],
            'gap',
            0.1,
            1.0,
            1e-10
        )
        
        assert len(predictions) == 2
        assert predictions.loc['CELL_0', 'predicted_type'] == 'TypeA'
        assert predictions.loc['CELL_1', 'predicted_type'] == 'TypeB'
        assert 'confidence' in predictions.columns
        assert 'score' in predictions.columns
    
    def test_confidence_threshold(self):
        """Low confidence predictions should be marked as Unknown."""
        scores = pd.DataFrame({
            'TypeA': [5.0, 5.0],  # Tied scores = low confidence
            'TypeB': [5.0, 5.0],
        }, index=['CELL_0', 'CELL_1'])
        
        predictions = predict_labels(
            scores,
            ['TypeA', 'TypeB'],
            'gap',
            0.5,  # High threshold
            1.0,
            1e-10
        )
        
        assert (predictions['predicted_type'] == 'Unknown').all()
    
    def test_gap_vs_entropy(self):
        """Both confidence methods should work."""
        scores = pd.DataFrame({
            'TypeA': [10.0, 1.0],
            'TypeB': [1.0, 10.0],
        }, index=['CELL_0', 'CELL_1'])
        
        pred_gap = predict_labels(scores, ['TypeA', 'TypeB'], 'gap', 0.1, 1.0, 1e-10)
        pred_entropy = predict_labels(scores, ['TypeA', 'TypeB'], 'entropy', 0.1, 1.0, 1e-10)
        
        assert len(pred_gap) == len(pred_entropy) == 2
        assert pred_gap.loc['CELL_0', 'predicted_type'] == pred_entropy.loc['CELL_0', 'predicted_type']
    
    def test_single_cell_type(self):
        """Single cell type should always predict that type."""
        scores = pd.DataFrame({
            'TypeA': [5.0, 3.0],
        }, index=['CELL_0', 'CELL_1'])
        
        predictions = predict_labels(
            scores,
            ['TypeA'],
            'gap',
            0.1,
            1.0,
            1e-10
        )
        
        assert (predictions['predicted_type'] == 'TypeA').all()
        assert (predictions['confidence'] == 1.0).all()
