"""Main FSCType predictor class."""

from typing import Optional, Dict, Union, List
import pandas as pd
from anndata import AnnData

from .config import FSCTypeConfig
from .markers import validate_markers
from .scoring import compute_specificity, compute_scores
from .neighbors import prepare_neighbors, aggregate_scores
from .confidence import predict_labels


class FSCType:
    """
    Fast Single-Cell Type annotation using k-nearest neighbors.
    
    A Python implementation of the fsctype algorithm that operates on
    AnnData objects and leverages sparse matrices for efficiency.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression data and neighbor graph.
    config : FSCTypeConfig, optional
        Configuration object with algorithm parameters.
        
    Examples
    --------
    >>> import scanpy as sc
    >>> import fsctype as fsc
    >>> 
    >>> # Preprocessing (user's responsibility)
    >>> sc.pp.neighbors(adata, n_neighbors=20)
    >>> 
    >>> # Cell type annotation
    >>> config = fsc.FSCTypeConfig(n_neighbors=20, weight_by_distance=True)
    >>> predictor = fsc.FSCType(adata, config)
    >>> predictions = predictor.predict(markers)
    """
    
    def __init__(self, adata: AnnData, config: Optional[FSCTypeConfig] = None) -> None:
        """Initialize FSCType with AnnData object."""
        self.adata = adata
        self.config = config or FSCTypeConfig()
        
        # Basic validation
        if adata.X is None:
            raise ValueError("AnnData object must contain expression matrix")
        
        # Validate expression layer at init
        self._validate_expression_layer()
        
        # Require neighbors to be pre-computed
        if 'neighbors' not in adata.uns:
            raise ValueError(
                "No neighbor graph found in adata.uns['neighbors']. "
                "Please run sc.pp.neighbors() first."
            )
        
        # Pre-compute neighbor arrays for performance
        if 'connectivities' not in adata.obsp:
            raise ValueError(
                "No 'connectivities' matrix found in adata.obsp. "
                "Please run sc.pp.neighbors() first."
            )
        
        self._neighbors, self._distances = prepare_neighbors(
            adata.obsp['connectivities'],
            self.config.n_neighbors
        )
        
        # Initialize caches
        self._gene_index_cache: Dict[str, pd.Index] = {}
    
    def _validate_expression_layer(self) -> None:
        """Validate that the specified expression layer exists."""
        layer = self.config.expression_layer
        
        if layer == 'X':
            # Always valid
            return
        elif layer == 'raw':
            if self.adata.raw is None:
                raise ValueError("expression_layer='raw' but adata.raw is None")
        else:
            # Custom layer
            if layer not in self.adata.layers:
                available = list(self.adata.layers.keys())
                raise ValueError(
                    f"Layer '{layer}' not found. Available layers: {available}"
                )
    
    def get_neighbors(self, cell_idx: int) -> tuple:
        """
        Get neighbors and distances for a specific cell.
        
        Parameters
        ----------
        cell_idx : int
            Index of the cell.
            
        Returns
        -------
        neighbors : np.ndarray
            Array of neighbor cell indices.
        distances : np.ndarray
            Array of corresponding distances/weights.
        """
        if not 0 <= cell_idx < len(self._neighbors):
            raise IndexError(f"Cell index {cell_idx} out of range")
        
        return self._neighbors[cell_idx], self._distances[cell_idx]
    
    def predict(
        self,
        markers: Dict[str, Union[List[str], Dict[str, List[str]]]],
        inplace: bool = True,
        key_added: str = 'fsctype_prediction',
        confidence_key: str = 'fsctype_confidence',
        score_key: str = 'fsctype_score',
        return_scores: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Predict cell types using FSCType algorithm.
        
        This is the main method that orchestrates the entire prediction pipeline:
        1. Prepare and validate marker genes
        2. Calculate marker specificity scores  
        3. Calculate raw cell type scores for each cell
        4. Aggregate scores across k-nearest neighbors
        5. Make final predictions with confidence scores
        
        Parameters
        ----------
        markers : dict
            Marker gene dictionary. Supports flexible formats:
            - Simple: {'T_cell': ['CD3D', 'CD3E']}
            - Full: {'T_cell': {'positive': ['CD3D'], 'negative': ['CD19']}}
        inplace : bool, default=True
            If True, add predictions to adata.obs. If False, return DataFrame.
        key_added : str, default='fsctype_prediction'
            Key name for predictions in adata.obs (if inplace=True).
        confidence_key : str, default='fsctype_confidence'  
            Key name for confidence scores in adata.obs (if inplace=True).
        score_key : str, default='fsctype_score'
            Key name for raw scores in adata.obs (if inplace=True).
        return_scores : bool, default=False
            If True, return tuple of (predictions, aggregated_scores).
            
        Returns
        -------
        predictions : pd.DataFrame or None
            If inplace=False, returns DataFrame with predictions.
            If inplace=True, returns None and modifies adata.obs.
        aggregated_scores : pd.DataFrame, optional
            If return_scores=True, also returns the aggregated cell type scores.
            
        Examples
        --------
        >>> # Simple usage
        >>> predictions = fsc.predict(markers, inplace=False)
        >>> 
        >>> # Add to AnnData object
        >>> fsc.predict(markers, inplace=True)
        >>> print(adata.obs['fsctype_prediction'].value_counts())
        >>> 
        >>> # Get detailed scores
        >>> predictions, scores = fsc.predict(markers, inplace=False, return_scores=True)
        """
        
        # Step 1: Prepare and validate markers
        processed_markers = validate_markers(
            markers,
            self.adata,
            self.config.expression_layer,
            self.config.use_positive_only,
            self.config.min_marker_genes
        )
        
        # Step 2: Calculate marker specificity scores
        specificity_scores = compute_specificity(
            processed_markers,
            self.config.use_positive_only
        )
        
        # Step 3: Calculate raw cell type scores
        cell_scores = compute_scores(
            processed_markers,
            specificity_scores,
            self.adata,
            self.config.expression_layer,
            self.config.use_positive_only,
            self.config.normalize_scores,
            self._gene_index_cache
        )
        
        # Step 4: Aggregate scores across neighborhoods (core innovation)
        aggregated_scores = aggregate_scores(
            cell_scores,
            self._neighbors,
            self._distances,
            self.config.weight_by_distance
        )
        
        # Step 5: Make final predictions with confidence (vectorized)
        predictions = predict_labels(
            aggregated_scores,
            list(processed_markers.keys()),
            self.config.confidence_method,
            self.config.confidence_threshold,
            self.config.softmax_temperature,
            self.config.entropy_epsilon
        )
        
        # Step 6: Handle output format
        if inplace:
            # Add predictions to AnnData object
            self.adata.obs[key_added] = predictions['predicted_type']
            self.adata.obs[confidence_key] = predictions['confidence']
            self.adata.obs[score_key] = predictions['score']
            
            if return_scores:
                return predictions, aggregated_scores
            else:
                return None
        else:
            # Return predictions DataFrame
            if return_scores:
                return predictions, aggregated_scores
            else:
                return predictions
