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
    confidence_method : str, default='entropy'
        Method for calculating confidence scores. Options: 'gap', 'entropy'.
        - 'gap': (max_score - second_max_score) / max_score
        - 'entropy': 1 - (normalized_shannon_entropy)
    softmax_temperature : float, default=1.0
        Temperature parameter for softmax in entropy calculation.
        Lower values make distributions sharper (higher confidence).
        Higher values make distributions smoother (lower confidence).
    entropy_epsilon : float, default=1e-10
        Small value to prevent log(0) in entropy calculation.
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
    confidence_method: str = 'entropy'
    softmax_temperature: float = 1.0
    entropy_epsilon: float = 1e-10
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
        
        if self.confidence_method not in ['gap', 'entropy']:
            raise ValueError(f"confidence_method must be 'gap' or 'entropy', got '{self.confidence_method}'")
        
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        
        if self.entropy_epsilon <= 0:
            raise ValueError("entropy_epsilon must be positive") 