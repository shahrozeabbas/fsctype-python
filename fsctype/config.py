"""Configuration classes for FSCType algorithm."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FSCTypeConfig:
    """
    Configuration for FSCType algorithm.
    
    Parameters
    ----------
    n_neighbors : int, default=20
        Number of nearest neighbors to consider for each cell.
    use_positive_only : bool, default=True
        Whether to use only positive markers or include negative markers.
    weight_by_distance : bool, default=True
        Whether to weight neighbor contributions by their distance/similarity.
    normalize_scores : bool, default=True
        Whether to normalize scores by the square root of number of markers.
    min_marker_genes : int, default=3
        Minimum number of marker genes required for a cell type.
    confidence_threshold : float, default=0.1
        Minimum confidence score for predictions (0-1 scale).
    random_state : int, optional
        Random state for reproducible results.
    expression_layer : str, default='X'
        Expression layer to use ('X', 'raw', or layer name).
    """
    
    n_neighbors: int = 20
    use_positive_only: bool = True
    weight_by_distance: bool = True
    normalize_scores: bool = True
    min_marker_genes: int = 3
    confidence_threshold: float = 0.1
    random_state: Optional[int] = None
    expression_layer: str = 'X'
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.min_marker_genes < 1:
            raise ValueError("min_marker_genes must be positive") 