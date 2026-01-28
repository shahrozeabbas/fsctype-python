"""Score calculation and marker specificity."""

from typing import Dict, List
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .markers import get_gene_indices, get_expression_subset


def compute_specificity(
    processed_markers: Dict[str, Dict[str, List[str]]],
    use_positive_only: bool
) -> Dict[str, float]:
    """
    Calculate marker gene specificity scores.
    
    More specific markers (appearing in fewer cell types) get higher scores,
    which makes intuitive sense for cell type classification.
    
    When use_positive_only=False, negative markers are included in frequency
    counting, allowing them to receive proper specificity weights instead of
    defaulting to 1.0.
    
    Parameters
    ----------
    processed_markers : dict
        Processed marker dictionary with standardized format.
    use_positive_only : bool
        Whether to include negative markers in specificity calculation.
        
    Returns
    -------
    specificity_scores : dict
        Dictionary mapping gene names to specificity scores (≥1.0).
        Higher values = more specific markers.
    """
    # Collect marker genes for specificity calculation
    marker_genes = []
    for cell_type_markers in processed_markers.values():
        marker_genes.extend(cell_type_markers['positive'])
        if not use_positive_only:
            marker_genes.extend(cell_type_markers['negative'])
    
    if not marker_genes:
        raise ValueError("No marker genes found across all cell types")
    
    # Count gene frequencies across collected markers
    gene_counts = Counter(marker_genes)
    
    # Calculate specificity scores (inverse of frequency)
    # Higher scores for MORE specific (less frequent) genes
    max_count = max(gene_counts.values())
    specificity_scores = {
        gene: max_count / count  # Inverse relationship
        for gene, count in gene_counts.items()
    }
    
    return specificity_scores


def compute_scores(
    processed_markers: Dict[str, Dict[str, List[str]]],
    specificity_scores: Dict[str, float],
    adata: AnnData,
    expression_layer: str,
    use_positive_only: bool,
    normalize_scores: bool,
    gene_index_cache: Dict[str, pd.Index]
) -> pd.DataFrame:
    """
    Calculate raw cell type scores for each cell using vectorized operations.
    
    This applies the corrected algorithm with full vectorization:
    1. Single sparse-to-dense conversion (if needed) for efficiency
    2. Pre-compute all weights and indices for batch processing
    3. Vectorized operations across all cell types
    4. Weight expression by marker specificity (higher weight = more specific)
    5. Sum weighted expression for positive markers
    6. Normalize by sqrt(number of markers) to handle varying marker set sizes
    7. Subtract negative marker scores if enabled
    
    Parameters
    ----------
    processed_markers : dict
        Processed marker dictionary with standardized format.
    specificity_scores : dict
        Gene specificity scores (higher = more specific).
    adata : AnnData
        Annotated data object.
    expression_layer : str
        Expression layer to use.
    use_positive_only : bool
        Whether to use only positive markers.
    normalize_scores : bool
        Whether to normalize scores by sqrt(n_markers).
    gene_index_cache : dict
        Cache for gene index lookups.
        
    Returns
    -------
    cell_scores : pd.DataFrame
        DataFrame with cells as rows and cell types as columns.
        Values are raw cell type scores.
    """
    n_cells = adata.shape[0]
    cell_types = list(processed_markers.keys())
    n_cell_types = len(cell_types)
    
    # Initialize score matrix
    scores = np.zeros((n_cells, n_cell_types), dtype=np.float32)
    
    # Get all unique marker genes for efficient expression extraction
    all_marker_genes = set()
    for markers in processed_markers.values():
        all_marker_genes.update(markers['positive'])
        if not use_positive_only:
            all_marker_genes.update(markers['negative'])
    
    all_marker_genes = list(all_marker_genes)
    
    # Get gene indices and expression data
    gene_indices = get_gene_indices(all_marker_genes, adata, expression_layer, gene_index_cache)
    if not gene_indices:
        raise ValueError("No marker genes found in expression data")
    
    # Extract expression matrix for marker genes (single operation)
    marker_gene_indices = list(gene_indices.values())
    X_markers = get_expression_subset(adata, marker_gene_indices, expression_layer)
    
    # OPTIMIZATION: Single sparse-to-dense conversion (if needed)
    if sp.issparse(X_markers):
        X_markers = X_markers.toarray()
    
    # Create mapping from gene name to column index in X_markers
    gene_to_col = {gene: i for i, gene in enumerate(all_marker_genes) if gene in gene_indices}
    
    # OPTIMIZATION: Pre-compute all marker data for vectorized processing
    pos_marker_data = []  # (cell_type_idx, gene_cols, weights)
    neg_marker_data = []  # (cell_type_idx, gene_cols, weights)
    
    for ct_idx, cell_type in enumerate(cell_types):
        ct_markers = processed_markers[cell_type]
        
        # Pre-compute positive marker data
        pos_genes = ct_markers['positive']
        if pos_genes:
            pos_cols = [gene_to_col[gene] for gene in pos_genes if gene in gene_to_col]
            if pos_cols:
                weights = np.array([specificity_scores.get(pos_genes[i], 1.0) 
                                  for i, _ in enumerate(pos_cols)], dtype=np.float32)
                pos_marker_data.append((ct_idx, pos_cols, weights))
        
        # Pre-compute negative marker data if enabled
        if not use_positive_only:
            neg_genes = ct_markers['negative']
            if neg_genes:
                neg_cols = [gene_to_col[gene] for gene in neg_genes if gene in gene_to_col]
                if neg_cols:
                    weights = np.array([specificity_scores.get(neg_genes[i], 1.0) 
                                      for i, _ in enumerate(neg_cols)], dtype=np.float32)
                    neg_marker_data.append((ct_idx, neg_cols, weights))
    
    # OPTIMIZATION: Vectorized positive marker scoring
    for ct_idx, pos_cols, weights in pos_marker_data:
        # Extract expression for positive markers (already dense)
        pos_expr = X_markers[:, pos_cols]
        
        # Vectorized weight application and scoring
        weighted_expr = pos_expr * weights[np.newaxis, :]
        
        # Calculate positive scores: sum(weighted_expr) / sqrt(n_genes)
        if normalize_scores:
            pos_scores = np.sum(weighted_expr, axis=1) / np.sqrt(len(pos_cols))
        else:
            pos_scores = np.sum(weighted_expr, axis=1)
        
        scores[:, ct_idx] += pos_scores
    
    # OPTIMIZATION: Vectorized negative marker scoring (if enabled)
    for ct_idx, neg_cols, weights in neg_marker_data:
        # Extract expression for negative markers (already dense)
        neg_expr = X_markers[:, neg_cols]
        
        # Vectorized weight application and scoring
        weighted_expr = neg_expr * weights[np.newaxis, :]
        
        # Calculate negative scores: sum(weighted_expr) / sqrt(n_genes)
        if normalize_scores:
            neg_scores = np.sum(weighted_expr, axis=1) / np.sqrt(len(neg_cols))
        else:
            neg_scores = np.sum(weighted_expr, axis=1)
        
        # Subtract negative marker contribution
        scores[:, ct_idx] -= neg_scores
    
    # Convert to DataFrame with proper labels
    cell_scores = pd.DataFrame(
        scores,
        index=adata.obs_names,
        columns=cell_types
    )
    
    return cell_scores
