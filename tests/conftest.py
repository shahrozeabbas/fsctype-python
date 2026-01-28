"""Shared pytest fixtures for FSCType tests."""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse


@pytest.fixture
def small_adata():
    """Create minimal AnnData for fast unit tests."""
    np.random.seed(42)
    n_cells, n_genes = 100, 50
    
    # Generate expression matrix
    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    X = np.abs(X)  # Ensure non-negative
    
    # Create AnnData object
    adata = ad.AnnData(X=sparse.csr_matrix(X))
    adata.var_names = [f'GENE_{i}' for i in range(n_genes)]
    adata.obs_names = [f'CELL_{i:03d}' for i in range(n_cells)]
    
    # Minimal preprocessing for neighbor graph
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    
    return adata


@pytest.fixture
def simple_markers():
    """Simple marker dictionary with 2 cell types."""
    return {
        'TypeA': ['GENE_0', 'GENE_1', 'GENE_2'],
        'TypeB': ['GENE_10', 'GENE_11', 'GENE_12'],
    }


@pytest.fixture
def full_markers():
    """Full marker dictionary with positive and negative markers."""
    return {
        'TypeA': {
            'positive': ['GENE_0', 'GENE_1', 'GENE_2'],
            'negative': ['GENE_10', 'GENE_11']
        },
        'TypeB': {
            'positive': ['GENE_10', 'GENE_11', 'GENE_12'],
            'negative': ['GENE_0', 'GENE_1']
        },
    }


@pytest.fixture
def processed_markers():
    """Pre-validated marker dictionary."""
    return {
        'TypeA': {
            'positive': ['GENE_0', 'GENE_1', 'GENE_2'],
            'negative': []
        },
        'TypeB': {
            'positive': ['GENE_10', 'GENE_11', 'GENE_12'],
            'negative': []
        },
    }
