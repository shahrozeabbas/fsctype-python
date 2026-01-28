"""Neighbor graph handling and score aggregation."""

import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp


def prepare_neighbors(
    conn_matrix: sp.spmatrix,
    n_neighbors: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract neighbors to list of dense arrays for efficient access.
    
    Parameters
    ----------
    conn_matrix : sp.spmatrix
        Connectivity matrix from scanpy neighbors (CSR format preferred).
    n_neighbors : int
        Number of neighbors to extract per cell.
        
    Returns
    -------
    neighbors : list of np.ndarray
        List of neighbor indices for each cell.
    distances : list of np.ndarray
        List of neighbor distances/weights for each cell.
    """
    # Ensure CSR format for efficient row access
    conn_matrix = conn_matrix.tocsr()
    n_cells = conn_matrix.shape[0]
    k = n_neighbors
    
    # Pre-allocate result arrays for better memory efficiency
    neighbors: List[np.ndarray] = []
    distances: List[np.ndarray] = []
    
    for i in range(n_cells):
        # Direct CSR access - much faster than getrow()
        start_idx = conn_matrix.indptr[i]
        end_idx = conn_matrix.indptr[i + 1]
        
        if end_idx > start_idx:
            # Get neighbors and weights for this cell
            neighbor_idx = conn_matrix.indices[start_idx:end_idx]
            neighbor_weights = conn_matrix.data[start_idx:end_idx]
            
            if len(neighbor_idx) > k:
                # Use argpartition for top-k: O(m) instead of O(m log m)
                top_k_indices = np.argpartition(-neighbor_weights, k)[:k]
                actual_neighbors = neighbor_idx[top_k_indices].astype(np.int32)
                actual_distances = neighbor_weights[top_k_indices].astype(np.float32)
            else:
                # Fewer neighbors than k - take all
                actual_neighbors = neighbor_idx.astype(np.int32)
                actual_distances = neighbor_weights.astype(np.float32)
        else:
            # No neighbors found
            actual_neighbors = np.array([], dtype=np.int32)
            actual_distances = np.array([], dtype=np.float32)
            warnings.warn(f"Cell {i} has no neighbors in the graph")
        
        neighbors.append(actual_neighbors)
        distances.append(actual_distances)
    
    return neighbors, distances


def aggregate_scores(
    cell_scores: pd.DataFrame,
    neighbors: List[np.ndarray],
    distances: List[np.ndarray],
    weight_by_distance: bool
) -> pd.DataFrame:
    """
    Aggregate cell type scores across k-nearest neighbors using vectorized operations.
    
    Uses pre-computed neighbor arrays and vectorized numpy operations for maximum efficiency.
    Eliminates Python loops and pandas indexing overhead.
    
    Parameters
    ----------
    cell_scores : pd.DataFrame
        Raw cell type scores with cells as rows and cell types as columns.
    neighbors : list of np.ndarray
        List of neighbor indices for each cell.
    distances : list of np.ndarray
        List of neighbor distances/weights for each cell.
    weight_by_distance : bool
        Whether to weight neighbor contributions by distance.
        
    Returns
    -------
    aggregated_scores : pd.DataFrame
        Neighborhood-aggregated scores with same structure as input.
    """
    n_cells, n_cell_types = cell_scores.shape
    
    # Convert to numpy for faster indexing
    scores_array = cell_scores.values.astype(np.float32)
    
    # Pre-allocate result array
    aggregated_scores = np.zeros((n_cells, n_cell_types), dtype=np.float32)
    
    # Track cells with no neighbors
    isolated_count = 0
    
    # Vectorized processing using pre-computed neighbor arrays
    for i in range(n_cells):
        neighbor_indices = neighbors[i]
        neighbor_weights = distances[i]
        
        if len(neighbor_indices) == 0:
            # Cell has no neighbors - use its own scores
            aggregated_scores[i] = scores_array[i]
            isolated_count += 1
            continue
        
        # Vectorized neighbor score extraction - much faster than pandas
        neighbor_scores = scores_array[neighbor_indices]  # Shape: (n_neighbors, n_cell_types)
        
        if weight_by_distance and len(neighbor_weights) > 0:
            # Vectorized weight normalization
            weight_sum = np.sum(neighbor_weights)
            if weight_sum > 0:
                normalized_weights = neighbor_weights / weight_sum
            else:
                # Fallback to equal weights
                normalized_weights = np.ones(len(neighbor_weights)) / len(neighbor_weights)
            
            # Vectorized weighted average: (n_neighbors, n_cell_types) -> (n_cell_types,)
            aggregated_scores[i] = np.average(neighbor_scores, axis=0, weights=normalized_weights)
        else:
            # Vectorized simple average
            aggregated_scores[i] = np.mean(neighbor_scores, axis=0)
    
    # Warn about isolated cells if any
    if isolated_count > 0:
        warnings.warn(
            f"Found {isolated_count} cells with no neighbors. "
            f"Using individual cell scores for these cells."
        )
    
    # Convert back to DataFrame with original labels
    aggregated_df = pd.DataFrame(
        aggregated_scores,
        index=cell_scores.index,
        columns=cell_scores.columns
    )
    
    return aggregated_df
