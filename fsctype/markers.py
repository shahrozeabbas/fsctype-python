"""Marker validation and gene index utilities."""

import warnings
from typing import Dict, List, Union, Set
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def validate_markers(
    markers: Dict[str, Union[List[str], Dict[str, List[str]]]],
    adata: AnnData,
    expression_layer: str,
    use_positive_only: bool,
    min_marker_genes: int
) -> Dict[str, Dict[str, List[str]]]:
    """
    Process and validate marker gene dictionaries.
    
    Supports flexible input formats:
    - Simple: {'T_cell': ['CD3D', 'CD3E']}  # Only positive markers
    - Full: {'T_cell': {'positive': ['CD3D'], 'negative': ['CD19']}}
    - Mixed: Combination of both formats
    
    Parameters
    ----------
    markers : dict
        Marker gene dictionary in flexible format.
    adata : AnnData
        Annotated data object.
    expression_layer : str
        Expression layer to use ('X', 'raw', or layer name).
    use_positive_only : bool
        Whether to use only positive markers.
    min_marker_genes : int
        Minimum number of marker genes required for a cell type.
        
    Returns
    -------
    processed_markers : dict
        Standardized marker dictionary with 'positive' and 'negative' keys.
        
    Raises
    ------
    ValueError
        If marker format is invalid or insufficient markers provided.
    """
    if not isinstance(markers, dict) or len(markers) == 0:
        raise ValueError("Markers must be a non-empty dictionary")
    
    # Get available genes for validation
    if expression_layer == 'raw':
        available_genes = set(adata.raw.var_names) if adata.raw is not None else set()
    else:
        available_genes = set(adata.var_names)
    
    if len(available_genes) == 0:
        raise ValueError("No genes found in the specified expression layer")
    
    processed_markers = {}
    
    for cell_type, marker_data in markers.items():
        if not isinstance(cell_type, str):
            raise ValueError(f"Cell type names must be strings, got {type(cell_type)}")
        
        processed_markers[cell_type] = {'positive': [], 'negative': []}
        
        # Handle different input formats
        if isinstance(marker_data, list):
            # Simple format: just a list of positive markers
            positive_markers = marker_data
            negative_markers = []
        elif isinstance(marker_data, dict):
            # Full format: dictionary with positive/negative keys
            positive_markers = marker_data.get('positive', marker_data.get('gs_positive', []))
            
            if not use_positive_only:
                negative_markers = marker_data.get('negative', marker_data.get('gs_negative', []))
            else:
                negative_markers = []
        else:
            raise ValueError(
                f"Invalid marker format for '{cell_type}'. "
                f"Expected list or dict, got {type(marker_data)}"
            )
        
        # Validate and filter positive markers
        if not isinstance(positive_markers, list):
            raise ValueError(f"Positive markers for '{cell_type}' must be a list")
        
        valid_positive = [gene for gene in positive_markers if gene in available_genes]
        missing_positive = [gene for gene in positive_markers if gene not in available_genes]
        
        if missing_positive:
            warnings.warn(
                f"Cell type '{cell_type}': {len(missing_positive)} positive marker genes not found: "
                f"{missing_positive[:5]}{'...' if len(missing_positive) > 5 else ''}"
            )
        
        # Check minimum marker requirement
        if len(valid_positive) < min_marker_genes:
            raise ValueError(
                f"Cell type '{cell_type}' has only {len(valid_positive)} valid positive markers. "
                f"Minimum required: {min_marker_genes}"
            )
        
        processed_markers[cell_type]['positive'] = valid_positive
        
        # Handle negative markers if enabled
        if not use_positive_only and negative_markers:
            if not isinstance(negative_markers, list):
                raise ValueError(f"Negative markers for '{cell_type}' must be a list")
            
            valid_negative = [gene for gene in negative_markers if gene in available_genes]
            missing_negative = [gene for gene in negative_markers if gene not in available_genes]
            
            if missing_negative:
                warnings.warn(
                    f"Cell type '{cell_type}': {len(missing_negative)} negative marker genes not found: "
                    f"{missing_negative[:3]}{'...' if len(missing_negative) > 3 else ''}"
                )
            
            processed_markers[cell_type]['negative'] = valid_negative
    
    return processed_markers


def get_gene_indices(
    gene_names: List[str],
    adata: AnnData,
    expression_layer: str,
    gene_index_cache: Dict[str, pd.Index]
) -> Dict[str, int]:
    """
    Get gene indices using efficient O(1) lookups with pandas Index.
    
    Uses pandas Index for fast gene name to index mapping instead of slow list.index() calls.
    Includes caching for optimal performance across repeated calls.
    
    Parameters
    ----------
    gene_names : list
        List of gene names to get indices for.
    adata : AnnData
        Annotated data object.
    expression_layer : str
        Expression layer to use ('X', 'raw', or layer name).
    gene_index_cache : dict
        Cache dictionary to store pandas Index (mutated in place).
        
    Returns
    -------
    gene_indices : dict
        Dictionary mapping gene names to their indices in the expression matrix.
    """
    # Get gene names from appropriate source
    if expression_layer == 'raw':
        available_genes = adata.raw.var_names if adata.raw is not None else adata.var_names
    else:
        available_genes = adata.var_names
    
    # Create pandas Index for O(1) lookups (cache this for efficiency)
    cache_key = f'{expression_layer}_index'
    if cache_key not in gene_index_cache:
        gene_index_cache[cache_key] = pd.Index(available_genes)
    
    gene_index = gene_index_cache[cache_key]
    
    # Batch lookup - handles missing genes gracefully (returns -1 for not found)
    indices = gene_index.get_indexer(gene_names)
    
    # Build result dictionary (skip -1 = not found)
    gene_indices = {}
    for i, gene in enumerate(gene_names):
        if indices[i] != -1:
            gene_indices[gene] = indices[i]
    
    return gene_indices


def get_expression_matrix(adata: AnnData, expression_layer: str) -> Union[np.ndarray, sp.spmatrix]:
    """Get expression matrix based on config layer."""
    if expression_layer == 'X':
        return adata.X
    elif expression_layer == 'raw':
        return adata.raw.X
    else:
        return adata.layers[expression_layer]


def should_densify(X_subset: Union[np.ndarray, sp.spmatrix]) -> bool:
    """Decide whether to convert sparse matrix to dense."""
    if not sp.issparse(X_subset):
        return False  # Already dense
    
    n_cells, n_genes = X_subset.shape
    sparsity = X_subset.nnz / (n_cells * n_genes)
    memory_dense = n_cells * n_genes * 4  # 4 bytes per float32
    
    # Convert if dense memory < 100MB OR sparsity > 30%
    return memory_dense < 100_000_000 or sparsity > 0.3


def get_expression_subset(
    adata: AnnData,
    gene_indices: List[int],
    expression_layer: str
) -> Union[np.ndarray, sp.spmatrix]:
    """Get expression subset with adaptive sparse/dense handling."""
    X = get_expression_matrix(adata, expression_layer)
    X_subset = X[:, gene_indices]
    
    if should_densify(X_subset):
        return X_subset.toarray() if sp.issparse(X_subset) else X_subset
    else:
        return X_subset
