"""Unit tests for score calculation functions."""

import pytest
import numpy as np
import pandas as pd
from fsctype.scoring import compute_specificity, compute_scores


class TestComputeSpecificity:
    """Tests for compute_specificity function."""
    
    def test_basic_specificity(self):
        """Basic specificity calculation should work."""
        processed_markers = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1'],
                'negative': []
            },
            'TypeB': {
                'positive': ['GENE_2', 'GENE_3'],
                'negative': []
            },
        }
        
        specificity = compute_specificity(processed_markers, True)
        
        assert 'GENE_0' in specificity
        assert 'GENE_1' in specificity
        assert all(score >= 1.0 for score in specificity.values())
    
    def test_rare_genes_higher_weight(self):
        """Rare genes should get higher specificity weights."""
        processed_markers = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1'],  # GENE_0 appears once
                'negative': []
            },
            'TypeB': {
                'positive': ['GENE_1', 'GENE_2'],  # GENE_1 appears twice
                'negative': []
            },
        }
        
        specificity = compute_specificity(processed_markers, True)
        
        # GENE_0 (appears once) should have higher weight than GENE_1 (appears twice)
        assert specificity['GENE_0'] > specificity['GENE_1']
    
    def test_include_negative_markers(self):
        """Negative markers should be included when use_positive_only=False."""
        processed_markers = {
            'TypeA': {
                'positive': ['GENE_0'],
                'negative': ['GENE_1']
            },
            'TypeB': {
                'positive': ['GENE_2'],
                'negative': ['GENE_1']  # GENE_1 appears twice (once negative)
            },
        }
        
        specificity_pos_only = compute_specificity(processed_markers, True)
        specificity_with_neg = compute_specificity(processed_markers, False)
        
        # With negative markers, GENE_1 should have different count
        assert 'GENE_1' not in specificity_pos_only
        assert 'GENE_1' in specificity_with_neg
    
    def test_empty_markers_error(self):
        """Empty markers should raise error."""
        with pytest.raises(ValueError, match="No marker genes"):
            compute_specificity({}, True)


class TestComputeScores:
    """Tests for compute_scores function."""
    
    def test_basic_scoring(self, small_adata, processed_markers):
        """Basic score computation should work."""
        specificity_scores = {
            'GENE_0': 2.0,
            'GENE_1': 2.0,
            'GENE_2': 2.0,
            'GENE_10': 1.5,
            'GENE_11': 1.5,
            'GENE_12': 1.5,
        }
        
        gene_index_cache = {}
        
        scores = compute_scores(
            processed_markers,
            specificity_scores,
            small_adata,
            'X',
            True,
            True,
            gene_index_cache
        )
        
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape[0] == small_adata.shape[0]
        assert 'TypeA' in scores.columns
        assert 'TypeB' in scores.columns
    
    def test_positive_markers_increase_score(self, small_adata):
        """Positive markers should increase cell type scores."""
        # Create markers with known genes
        processed_markers = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1'],
                'negative': []
            },
        }
        
        # Set high expression for first cell, low for others
        small_adata.X[0, 0] = 10.0
        small_adata.X[0, 1] = 10.0
        small_adata.X[1:, 0] = 0.1
        small_adata.X[1:, 1] = 0.1
        
        specificity_scores = {'GENE_0': 1.0, 'GENE_1': 1.0}
        gene_index_cache = {}
        
        scores = compute_scores(
            processed_markers,
            specificity_scores,
            small_adata,
            'X',
            True,
            True,
            gene_index_cache
        )
        
        # First cell should have higher score
        assert scores.loc[small_adata.obs_names[0], 'TypeA'] > scores.loc[small_adata.obs_names[1], 'TypeA']
    
    def test_negative_markers_decrease_score(self, small_adata):
        """Negative markers should decrease cell type scores."""
        processed_markers = {
            'TypeA': {
                'positive': ['GENE_0'],
                'negative': ['GENE_10']
            },
        }
        
        # Set high expression for negative marker
        small_adata.X[0, 10] = 10.0
        small_adata.X[1:, 10] = 0.1
        
        specificity_scores = {'GENE_0': 1.0, 'GENE_10': 1.0}
        gene_index_cache = {}
        
        scores = compute_scores(
            processed_markers,
            specificity_scores,
            small_adata,
            'X',
            False,  # Enable negative markers
            True,
            gene_index_cache
        )
        
        # First cell should have lower score due to negative marker
        assert scores.loc[small_adata.obs_names[0], 'TypeA'] < scores.loc[small_adata.obs_names[1], 'TypeA']
    
    def test_normalization(self, small_adata):
        """Score normalization by sqrt(n_markers) should work."""
        processed_markers_small = {
            'TypeA': {
                'positive': ['GENE_0'],
                'negative': []
            },
        }
        processed_markers_large = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1', 'GENE_2', 'GENE_3'],
                'negative': []
            },
        }
        
        # Set same expression
        small_adata.X[:, 0] = 5.0
        small_adata.X[:, 1] = 5.0
        small_adata.X[:, 2] = 5.0
        small_adata.X[:, 3] = 5.0
        
        specificity_scores = {'GENE_0': 1.0, 'GENE_1': 1.0, 'GENE_2': 1.0, 'GENE_3': 1.0}
        gene_index_cache = {}
        
        scores_small = compute_scores(
            processed_markers_small,
            specificity_scores,
            small_adata,
            'X',
            True,
            True,  # Normalize
            gene_index_cache
        )
        
        scores_large = compute_scores(
            processed_markers_large,
            specificity_scores,
            small_adata,
            'X',
            True,
            True,  # Normalize
            gene_index_cache
        )
        
        # With normalization, scores should be more comparable
        # Large marker set should have higher score but not 4x higher
        assert scores_large.loc[small_adata.obs_names[0], 'TypeA'] > scores_small.loc[small_adata.obs_names[0], 'TypeA']
        # But not 4x higher due to normalization
        ratio = scores_large.loc[small_adata.obs_names[0], 'TypeA'] / scores_small.loc[small_adata.obs_names[0], 'TypeA']
        assert ratio < 4.0
    
    def test_no_normalization(self, small_adata):
        """Without normalization, more markers = higher scores."""
        processed_markers_small = {
            'TypeA': {
                'positive': ['GENE_0'],
                'negative': []
            },
        }
        processed_markers_large = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1'],
                'negative': []
            },
        }
        
        small_adata.X[:, 0] = 5.0
        small_adata.X[:, 1] = 5.0
        
        specificity_scores = {'GENE_0': 1.0, 'GENE_1': 1.0}
        gene_index_cache = {}
        
        scores_small = compute_scores(
            processed_markers_small,
            specificity_scores,
            small_adata,
            'X',
            True,
            False,  # No normalization
            gene_index_cache
        )
        
        scores_large = compute_scores(
            processed_markers_large,
            specificity_scores,
            small_adata,
            'X',
            True,
            False,  # No normalization
            gene_index_cache
        )
        
        # Without normalization, large should be ~2x higher
        ratio = scores_large.loc[small_adata.obs_names[0], 'TypeA'] / scores_small.loc[small_adata.obs_names[0], 'TypeA']
        assert ratio == pytest.approx(2.0, rel=0.1)
    
    def test_missing_genes_error(self, small_adata):
        """Missing all marker genes should raise error."""
        processed_markers = {
            'TypeA': {
                'positive': ['FAKE_GENE_1', 'FAKE_GENE_2'],
                'negative': []
            },
        }
        
        specificity_scores = {'FAKE_GENE_1': 1.0, 'FAKE_GENE_2': 1.0}
        gene_index_cache = {}
        
        with pytest.raises(ValueError, match="No marker genes found"):
            compute_scores(
                processed_markers,
                specificity_scores,
                small_adata,
                'X',
                True,
                True,
                gene_index_cache
            )
