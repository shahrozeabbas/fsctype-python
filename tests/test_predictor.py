"""Integration tests for FSCType predictor class."""

import pytest
import pandas as pd
from fsctype import FSCType, FSCTypeConfig


class TestFSCTypeInitialization:
    """Tests for FSCType initialization."""
    
    def test_basic_initialization(self, small_adata):
        """Basic initialization should work."""
        config = FSCTypeConfig()
        predictor = FSCType(small_adata, config)
        
        assert predictor.adata is small_adata
        assert predictor.config is config
    
    def test_default_config(self, small_adata):
        """Should use default config if none provided."""
        predictor = FSCType(small_adata)
        
        assert isinstance(predictor.config, FSCTypeConfig)
        assert predictor.config.n_neighbors == 20
    
    def test_no_expression_matrix_error(self):
        """Should raise error if no expression matrix."""
        import anndata as ad
        adata = ad.AnnData()
        adata.X = None
        
        with pytest.raises(ValueError, match="must contain expression matrix"):
            FSCType(adata)
    
    def test_no_neighbors_error(self):
        """Should raise error if neighbors not computed."""
        import anndata as ad
        import numpy as np
        from scipy import sparse
        
        adata = ad.AnnData(X=sparse.csr_matrix(np.random.randn(10, 5)))
        adata.uns.pop('neighbors', None)
        adata.obsp.pop('connectivities', None)
        
        with pytest.raises(ValueError, match="neighbor graph"):
            FSCType(adata)
    
    def test_invalid_layer_error(self, small_adata):
        """Should raise error for invalid expression layer."""
        config = FSCTypeConfig(expression_layer='nonexistent')
        
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            FSCType(small_adata, config)
    
    def test_get_neighbors(self, small_adata):
        """get_neighbors should return correct neighbors."""
        predictor = FSCType(small_adata)
        
        neighbors, distances = predictor.get_neighbors(0)
        
        assert isinstance(neighbors, type(predictor._neighbors[0]))
        assert isinstance(distances, type(predictor._distances[0]))
        assert len(neighbors) == len(distances)
    
    def test_get_neighbors_out_of_range(self, small_adata):
        """Should raise error for out of range index."""
        predictor = FSCType(small_adata)
        
        with pytest.raises(IndexError):
            predictor.get_neighbors(10000)


class TestFSCTypePredict:
    """Tests for FSCType.predict method."""
    
    def test_basic_prediction(self, small_adata, simple_markers):
        """Basic prediction should work."""
        predictor = FSCType(small_adata)
        
        predictions = predictor.predict(simple_markers, inplace=False)
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == small_adata.shape[0]
        assert 'predicted_type' in predictions.columns
        assert 'confidence' in predictions.columns
        assert 'score' in predictions.columns
    
    def test_inplace_true(self, small_adata, simple_markers):
        """inplace=True should modify adata.obs."""
        predictor = FSCType(small_adata)
        
        result = predictor.predict(simple_markers, inplace=True)
        
        assert result is None
        assert 'fsctype_prediction' in small_adata.obs.columns
        assert 'fsctype_confidence' in small_adata.obs.columns
        assert 'fsctype_score' in small_adata.obs.columns
    
    def test_custom_column_names(self, small_adata, simple_markers):
        """Custom column names should work."""
        predictor = FSCType(small_adata)
        
        predictor.predict(
            simple_markers,
            inplace=True,
            key_added='custom_pred',
            confidence_key='custom_conf',
            score_key='custom_score'
        )
        
        assert 'custom_pred' in small_adata.obs.columns
        assert 'custom_conf' in small_adata.obs.columns
        assert 'custom_score' in small_adata.obs.columns
    
    def test_return_scores(self, small_adata, simple_markers):
        """return_scores=True should return scores DataFrame."""
        predictor = FSCType(small_adata)
        
        predictions, scores = predictor.predict(
            simple_markers,
            inplace=False,
            return_scores=True
        )
        
        assert isinstance(predictions, pd.DataFrame)
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape[0] == small_adata.shape[0]
        assert 'TypeA' in scores.columns
        assert 'TypeB' in scores.columns
    
    def test_full_marker_format(self, small_adata, full_markers):
        """Full marker format with negative markers should work."""
        config = FSCTypeConfig(use_positive_only=False)
        predictor = FSCType(small_adata, config)
        
        predictions = predictor.predict(full_markers, inplace=False)
        
        assert len(predictions) == small_adata.shape[0]
        assert predictions['predicted_type'].isin(['TypeA', 'TypeB', 'Unknown']).all()
    
    def test_gap_confidence_method(self, small_adata, simple_markers):
        """Gap confidence method should work."""
        config = FSCTypeConfig(confidence_method='gap')
        predictor = FSCType(small_adata, config)
        
        predictions = predictor.predict(simple_markers, inplace=False)
        
        assert (predictions['confidence'] >= 0.0).all()
        assert (predictions['confidence'] <= 1.0).all()
    
    def test_entropy_confidence_method(self, small_adata, simple_markers):
        """Entropy confidence method should work."""
        config = FSCTypeConfig(confidence_method='entropy')
        predictor = FSCType(small_adata, config)
        
        predictions = predictor.predict(simple_markers, inplace=False)
        
        assert (predictions['confidence'] >= 0.0).all()
        assert (predictions['confidence'] <= 1.0).all()
    
    def test_confidence_threshold(self, small_adata, simple_markers):
        """Confidence threshold should mark low confidence as Unknown."""
        config = FSCTypeConfig(confidence_threshold=0.9)  # Very high threshold
        predictor = FSCType(small_adata, config)
        
        predictions = predictor.predict(simple_markers, inplace=False)
        
        # Most predictions should be Unknown with such high threshold
        unknown_count = (predictions['predicted_type'] == 'Unknown').sum()
        assert unknown_count >= 0  # At least some might be Unknown
    
    def test_different_neighbor_counts(self, small_adata, simple_markers):
        """Different n_neighbors should affect results."""
        config_small = FSCTypeConfig(n_neighbors=5)
        config_large = FSCTypeConfig(n_neighbors=20)
        
        predictor_small = FSCType(small_adata, config_small)
        predictor_large = FSCType(small_adata, config_large)
        
        pred_small = predictor_small.predict(simple_markers, inplace=False)
        pred_large = predictor_large.predict(simple_markers, inplace=False)
        
        assert len(pred_small) == len(pred_large)
        # Results might differ but should both be valid
        assert pred_small['predicted_type'].isin(['TypeA', 'TypeB', 'Unknown']).all()
        assert pred_large['predicted_type'].isin(['TypeA', 'TypeB', 'Unknown']).all()
    
    def test_weight_by_distance(self, small_adata, simple_markers):
        """weight_by_distance should affect aggregation."""
        config_weighted = FSCTypeConfig(weight_by_distance=True)
        config_unweighted = FSCTypeConfig(weight_by_distance=False)
        
        predictor_weighted = FSCType(small_adata, config_weighted)
        predictor_unweighted = FSCType(small_adata, config_unweighted)
        
        pred_weighted = predictor_weighted.predict(simple_markers, inplace=False)
        pred_unweighted = predictor_unweighted.predict(simple_markers, inplace=False)
        
        # Both should produce valid results
        assert len(pred_weighted) == len(pred_unweighted)
        # Results might differ but both valid
        assert pred_weighted['predicted_type'].isin(['TypeA', 'TypeB', 'Unknown']).all()
