"""Vectorized prediction and confidence calculation."""

import warnings
from typing import List
import numpy as np
import pandas as pd


def gap_confidence_batch(scores: np.ndarray) -> np.ndarray:
    """
    Calculate confidence using gap between top two scores (vectorized).
    
    Processes all cells in one numpy call.
    
    Parameters
    ----------
    scores : np.ndarray
        Cell type scores for all cells, shape (n_cells, n_cell_types).
        
    Returns
    -------
    confidence : np.ndarray
        Confidence scores in [0, 1] range, shape (n_cells,).
    """
    n_cells, n_cell_types = scores.shape
    
    if n_cell_types == 1:
        return np.ones(n_cells, dtype=np.float32)
    
    # Sort scores in descending order along cell type axis
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    best = sorted_scores[:, 0]
    second_best = sorted_scores[:, 1] if n_cell_types > 1 else np.zeros(n_cells)
    
    # Vectorized confidence calculation: (max - second_max) / max
    confidence = np.where(
        best <= 0,
        0.0,
        (best - second_best) / np.maximum(best, 1e-10)
    )
    
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def entropy_confidence_batch(
    scores: np.ndarray,
    temperature: float = 1.0,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Calculate confidence using entropy (vectorized).
    
    Processes all cells in one numpy call.
    
    Parameters
    ----------
    scores : np.ndarray
        Cell type scores for all cells, shape (n_cells, n_cell_types).
    temperature : float, default=1.0
        Temperature parameter for softmax.
    epsilon : float, default=1e-10
        Small value to prevent log(0).
        
    Returns
    -------
    confidence : np.ndarray
        Confidence scores in [0, 1] range, shape (n_cells,).
    """
    n_cells, n_cell_types = scores.shape
    
    if n_cell_types == 1:
        return np.ones(n_cells, dtype=np.float32)
    
    # Stable softmax (vectorized)
    shifted = (scores - scores.max(axis=1, keepdims=True)) / temperature
    shifted = np.clip(shifted, -500, 500)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # Entropy (vectorized)
    safe_probs = np.clip(probs, epsilon, 1.0 - epsilon)
    entropy = -np.sum(probs * np.log(safe_probs), axis=1)
    
    # Normalize to [0, 1] range
    max_entropy = np.log(n_cell_types)
    confidence = 1.0 - (entropy / max_entropy)
    
    # Handle edge cases
    # All scores identical (within tolerance)
    identical_mask = np.allclose(scores, scores[:, 0:1], rtol=1e-9, axis=1)
    confidence = np.where(identical_mask, 0.0, confidence)
    
    # Non-finite scores
    finite_mask = np.isfinite(scores).all(axis=1)
    confidence = np.where(finite_mask, confidence, 0.0)
    
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def predict_labels(
    aggregated_scores: pd.DataFrame,
    cell_types: List[str],
    confidence_method: str,
    confidence_threshold: float,
    softmax_temperature: float = 1.0,
    entropy_epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Convert aggregated scores to final predictions with confidence scores (vectorized).
    
    For all cells at once, finds the highest-scoring cell type and calculates confidence
    scores using either gap-based or entropy-based method.
    
    Parameters
    ----------
    aggregated_scores : pd.DataFrame
        Neighborhood-aggregated cell type scores.
    cell_types : list
        List of cell type names (must match DataFrame columns).
    confidence_method : str
        Method for calculating confidence ('gap' or 'entropy').
    confidence_threshold : float
        Minimum confidence score for predictions.
    softmax_temperature : float, default=1.0
        Temperature parameter for entropy method.
    entropy_epsilon : float, default=1e-10
        Small value to prevent log(0) in entropy calculation.
        
    Returns
    -------
    predictions : pd.DataFrame
        DataFrame with columns: ['predicted_type', 'score', 'confidence'].
        Index matches aggregated_scores index.
    """
    scores_array = aggregated_scores.values.astype(np.float32)
    n_cells, n_cell_types = scores_array.shape
    
    # Vectorized argmax and max
    best_indices = np.argmax(scores_array, axis=1)
    best_scores = np.max(scores_array, axis=1)
    
    # Vectorized confidence calculation
    if confidence_method == 'gap':
        confidences = gap_confidence_batch(scores_array)
    else:  # entropy
        confidences = entropy_confidence_batch(
            scores_array, softmax_temperature, entropy_epsilon
        )
    
    # Map indices to cell type names
    cell_types_array = np.array(cell_types)
    predicted_types = cell_types_array[best_indices]
    
    # Apply threshold (vectorized)
    low_confidence_mask = confidences < confidence_threshold
    predicted_types = np.where(low_confidence_mask, 'Unknown', predicted_types)
    
    # Build result DataFrame
    predictions_df = pd.DataFrame({
        'predicted_type': predicted_types,
        'score': best_scores,
        'confidence': confidences
    }, index=aggregated_scores.index)
    
    return predictions_df
