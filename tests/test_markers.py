"""Unit tests for marker validation functions."""

import pytest
import warnings
from fsctype.markers import validate_markers, get_gene_indices, get_expression_matrix, should_densify, get_expression_subset
import anndata as ad
import numpy as np
from scipy import sparse


class TestValidateMarkers:
    """Tests for validate_markers function."""
    
    def test_simple_format(self, small_adata):
        """Simple format (list of genes) should work."""
        markers = {
            'TypeA': ['GENE_0', 'GENE_1', 'GENE_2'],
            'TypeB': ['GENE_10', 'GENE_11', 'GENE_12'],
        }
        
        result = validate_markers(
            markers,
            small_adata,
            'X',
            True,
            3
        )
        
        assert 'TypeA' in result
        assert 'TypeB' in result
        assert result['TypeA']['positive'] == ['GENE_0', 'GENE_1', 'GENE_2']
        assert result['TypeA']['negative'] == []
    
    def test_full_format(self, small_adata):
        """Full format (positive/negative dict) should work."""
        markers = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1', 'GENE_2'],
                'negative': ['GENE_10', 'GENE_11']
            },
        }
        
        result = validate_markers(
            markers,
            small_adata,
            'X',
            False,
            3
        )
        
        assert result['TypeA']['positive'] == ['GENE_0', 'GENE_1', 'GENE_2']
        assert result['TypeA']['negative'] == ['GENE_10', 'GENE_11']
    
    def test_gs_positive_format(self, small_adata):
        """gs_positive/gs_negative format should work."""
        markers = {
            'TypeA': {
                'gs_positive': ['GENE_0', 'GENE_1', 'GENE_2'],
                'gs_negative': ['GENE_10']
            },
        }
        
        result = validate_markers(
            markers,
            small_adata,
            'X',
            False,
            3
        )
        
        assert result['TypeA']['positive'] == ['GENE_0', 'GENE_1', 'GENE_2']
        assert result['TypeA']['negative'] == ['GENE_10']
    
    def test_missing_genes_warning(self, small_adata):
        """Missing genes should emit warnings."""
        markers = {
            'TypeA': ['GENE_0', 'FAKE_GENE_1', 'GENE_2', 'FAKE_GENE_2'],
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_markers(markers, small_adata, 'X', True, 2)
            
            assert len(w) > 0
            assert any('FAKE_GENE' in str(warning.message) for warning in w)
            assert 'GENE_0' in result['TypeA']['positive']
            assert 'FAKE_GENE_1' not in result['TypeA']['positive']
    
    def test_minimum_marker_requirement(self, small_adata):
        """Should raise error if not enough valid markers."""
        markers = {
            'TypeA': ['GENE_0'],  # Only 1 marker, need 3
        }
        
        with pytest.raises(ValueError, match="Minimum required"):
            validate_markers(markers, small_adata, 'X', True, 3)
    
    def test_empty_markers_error(self, small_adata):
        """Empty markers dict should raise error."""
        with pytest.raises(ValueError, match="non-empty dictionary"):
            validate_markers({}, small_adata, 'X', True, 3)
    
    def test_invalid_format_error(self, small_adata):
        """Invalid format should raise error."""
        markers = {
            'TypeA': 123,  # Not a list or dict
        }
        
        with pytest.raises(ValueError, match="Expected list or dict"):
            validate_markers(markers, small_adata, 'X', True, 3)
    
    def test_use_positive_only(self, small_adata):
        """use_positive_only=True should ignore negative markers."""
        markers = {
            'TypeA': {
                'positive': ['GENE_0', 'GENE_1', 'GENE_2'],
                'negative': ['GENE_10', 'GENE_11']
            },
        }
        
        result = validate_markers(markers, small_adata, 'X', True, 3)
        assert result['TypeA']['negative'] == []
        
        result = validate_markers(markers, small_adata, 'X', False, 3)
        assert len(result['TypeA']['negative']) > 0
    
    def test_raw_layer(self, small_adata):
        """Should work with raw layer."""
        small_adata.raw = small_adata.copy()
        markers = {
            'TypeA': ['GENE_0', 'GENE_1', 'GENE_2'],
        }
        
        result = validate_markers(markers, small_adata, 'raw', True, 3)
        assert 'TypeA' in result


class TestGetGeneIndices:
    """Tests for get_gene_indices function."""
    
    def test_basic_lookup(self, small_adata):
        """Basic gene lookup should work."""
        gene_names = ['GENE_0', 'GENE_1', 'GENE_2']
        cache = {}
        
        indices = get_gene_indices(gene_names, small_adata, 'X', cache)
        
        assert len(indices) == 3
        assert 'GENE_0' in indices
        assert isinstance(indices['GENE_0'], int)
    
    def test_missing_genes(self, small_adata):
        """Missing genes should be skipped."""
        gene_names = ['GENE_0', 'FAKE_GENE', 'GENE_1']
        cache = {}
        
        indices = get_gene_indices(gene_names, small_adata, 'X', cache)
        
        assert 'GENE_0' in indices
        assert 'GENE_1' in indices
        assert 'FAKE_GENE' not in indices
    
    def test_cache_reuse(self, small_adata):
        """Cache should be reused across calls."""
        gene_names = ['GENE_0', 'GENE_1']
        cache = {}
        
        indices1 = get_gene_indices(gene_names, small_adata, 'X', cache)
        indices2 = get_gene_indices(gene_names, small_adata, 'X', cache)
        
        assert indices1 == indices2
        assert 'X_index' in cache


class TestExpressionHelpers:
    """Tests for expression helper functions."""
    
    def test_get_expression_matrix(self, small_adata):
        """Should return correct expression matrix."""
        X = get_expression_matrix(small_adata, 'X')
        assert X.shape == small_adata.shape
    
    def test_should_densify_sparse(self):
        """Should densify small sparse matrices."""
        X_sparse = sparse.csr_matrix(np.random.randn(100, 10))
        assert should_densify(X_sparse) is True
    
    def test_should_densify_dense(self):
        """Should not densify already dense matrices."""
        X_dense = np.random.randn(100, 10)
        assert should_densify(X_dense) is False
    
    def test_get_expression_subset(self, small_adata):
        """Should return subset of expression matrix."""
        gene_indices = [0, 1, 2]
        X_subset = get_expression_subset(small_adata, gene_indices, 'X')
        assert X_subset.shape[1] == 3
