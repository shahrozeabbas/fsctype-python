"""Core FSCType implementation."""

import warnings
from typing import Optional, List, Union, Dict, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .config import FSCTypeConfig


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
        self._prepare_neighbors()
        
        # Initialize caches
        self._gene_indices_cache: Dict[str, int] = {}
        self._marker_specificity_cache: Dict[str, float] = {}
    
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
    
    def _prepare_neighbors(self) -> None:
        """Extract neighbors to list of dense arrays for efficient access."""
        # Get connectivity matrix from scanpy neighbors
        if 'connectivities' not in self.adata.obsp:
            raise ValueError(
                "No 'connectivities' matrix found in adata.obsp. "
                "Please run sc.pp.neighbors() first."
            )
        
        conn_matrix = self.adata.obsp['connectivities']
        n_cells = conn_matrix.shape[0]
        
        # Initialize lists to store neighbors and distances
        self._neighbors: List[np.ndarray] = []
        self._distances: List[np.ndarray] = []
        
        for i in range(n_cells):
            # Get neighbors for cell i
            row = conn_matrix.getrow(i)
            neighbor_idx = row.indices
            neighbor_weights = row.data
            
            if len(neighbor_idx) > 0:
                # Sort by weight (descending) and take top k
                sorted_indices = np.argsort(-neighbor_weights)[:self.config.n_neighbors]
                actual_neighbors = neighbor_idx[sorted_indices].astype(np.int32)
                actual_distances = neighbor_weights[sorted_indices].astype(np.float32)
            else:
                # No neighbors found
                actual_neighbors = np.array([], dtype=np.int32)
                actual_distances = np.array([], dtype=np.float32)
                warnings.warn(f"Cell {i} has no neighbors in the graph")
            
            self._neighbors.append(actual_neighbors)
            self._distances.append(actual_distances)
    
    def _get_expression_matrix(self) -> Union[np.ndarray, sp.spmatrix]:
        """Get expression matrix based on config layer."""
        layer = self.config.expression_layer
        
        if layer == 'X':
            return self.adata.X
        elif layer == 'raw':
            return self.adata.raw.X
        else:
            return self.adata.layers[layer]
    
    def _should_densify(self, X_subset: Union[np.ndarray, sp.spmatrix]) -> bool:
        """Decide whether to convert sparse matrix to dense."""
        if not sp.issparse(X_subset):
            return False  # Already dense
        
        n_cells, n_genes = X_subset.shape
        sparsity = X_subset.nnz / (n_cells * n_genes)
        memory_dense = n_cells * n_genes * 4  # 4 bytes per float32
        
        # Convert if dense memory < 100MB OR sparsity > 30%
        return memory_dense < 100_000_000 or sparsity > 0.3
    
    def _get_expression_subset(self, gene_indices: List[int]) -> Union[np.ndarray, sp.spmatrix]:
        """Get expression subset with adaptive sparse/dense handling."""
        X = self._get_expression_matrix()
        X_subset = X[:, gene_indices]
        
        if self._should_densify(X_subset):
            return X_subset.toarray() if sp.issparse(X_subset) else X_subset
        else:
            return X_subset
    
    def get_neighbors(self, cell_idx: int) -> tuple[np.ndarray, np.ndarray]:
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
    
    def _prepare_markers(self, markers: Dict[str, Union[List[str], Dict[str, List[str]]]]) -> Dict[str, Dict[str, List[str]]]:
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
        if self.config.expression_layer == 'raw':
            available_genes = set(self.adata.raw.var_names) if self.adata.raw is not None else set()
        else:
            available_genes = set(self.adata.var_names)
        
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
                
                if not self.config.use_positive_only:
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
            if len(valid_positive) < self.config.min_marker_genes:
                raise ValueError(
                    f"Cell type '{cell_type}' has only {len(valid_positive)} valid positive markers. "
                    f"Minimum required: {self.config.min_marker_genes}"
                )
            
            processed_markers[cell_type]['positive'] = valid_positive
            
            # Handle negative markers if enabled
            if not self.config.use_positive_only and negative_markers:
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
    
    def _calculate_marker_sensitivity(self, processed_markers: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
        """
        Calculate marker gene specificity scores.
        
        This fixes two issues from the R implementation:
        1. Bug: Used all markers (pos+neg) for counting but only positive count for rescaling
        2. Logic: Now gives higher weights to MORE specific (less frequent) markers
        
        More specific markers (appearing in fewer cell types) get higher scores,
        which makes intuitive sense for cell type classification.
        
        Parameters
        ----------
        processed_markers : dict
            Processed marker dictionary with standardized format.
            
        Returns
        -------
        specificity_scores : dict
            Dictionary mapping gene names to specificity scores (≥1.0).
            Higher values = more specific markers.
        """
        # Extract ONLY positive markers for specificity calculation (bug fix)
        positive_genes = []
        for cell_type_markers in processed_markers.values():
            positive_genes.extend(cell_type_markers['positive'])
        
        if not positive_genes:
            raise ValueError("No positive marker genes found across all cell types")
        
        # Count gene frequencies across positive markers only
        gene_counts = Counter(positive_genes)
        
        # Calculate specificity scores (inverse of frequency)
        # Higher scores for MORE specific (less frequent) genes
        max_count = max(gene_counts.values())
        specificity_scores = {
            gene: max_count / count  # Inverse relationship
            for gene, count in gene_counts.items()
        }
        
        # Cache for potential reuse
        self._marker_specificity_cache.update(specificity_scores)
        
        return specificity_scores
    
    def _get_gene_indices(self, gene_names: List[str]) -> Dict[str, int]:
        """
        Get gene indices with caching for performance.
        
        Parameters
        ----------
        gene_names : list
            List of gene names to get indices for.
            
        Returns
        -------
        gene_indices : dict
            Dictionary mapping gene names to their indices in the expression matrix.
        """
        # Get gene names from appropriate source
        if self.config.expression_layer == 'raw':
            available_genes = self.adata.raw.var_names if self.adata.raw is not None else self.adata.var_names
        else:
            available_genes = self.adata.var_names
        
        # Build index mapping with caching
        gene_indices = {}
        for gene in gene_names:
            if gene not in self._gene_indices_cache:
                try:
                    self._gene_indices_cache[gene] = list(available_genes).index(gene)
                except ValueError:
                    # Gene not found (should have been caught in _prepare_markers)
                    continue
            
            gene_indices[gene] = self._gene_indices_cache[gene]
        
        return gene_indices
    
    def _calculate_cell_scores(self, 
                              processed_markers: Dict[str, Dict[str, List[str]]], 
                              specificity_scores: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate raw cell type scores for each cell.
        
        This applies the corrected algorithm:
        1. Weight expression by marker specificity (higher weight = more specific)
        2. Sum weighted expression for positive markers
        3. Normalize by sqrt(number of markers) to handle varying marker set sizes
        4. Subtract negative marker scores if enabled
        
        Parameters
        ----------
        processed_markers : dict
            Processed marker dictionary with standardized format.
        specificity_scores : dict
            Gene specificity scores (higher = more specific).
            
        Returns
        -------
        cell_scores : pd.DataFrame
            DataFrame with cells as rows and cell types as columns.
            Values are raw cell type scores.
        """
        n_cells = self.adata.shape[0]
        cell_types = list(processed_markers.keys())
        
        # Initialize score matrix
        scores = np.zeros((n_cells, len(cell_types)), dtype=np.float32)
        
        # Get all unique marker genes for efficient expression extraction
        all_marker_genes = set()
        for markers in processed_markers.values():
            all_marker_genes.update(markers['positive'])
            if not self.config.use_positive_only:
                all_marker_genes.update(markers['negative'])
        
        all_marker_genes = list(all_marker_genes)
        
        # Get gene indices and expression data
        gene_indices = self._get_gene_indices(all_marker_genes)
        if not gene_indices:
            raise ValueError("No marker genes found in expression data")
        
        # Extract expression matrix for marker genes
        marker_gene_indices = list(gene_indices.values())
        X_markers = self._get_expression_subset(marker_gene_indices)
        
        # Create mapping from gene name to column index in X_markers
        gene_to_col = {gene: i for i, gene in enumerate(all_marker_genes) if gene in gene_indices}
        
        # Calculate scores for each cell type
        for ct_idx, cell_type in enumerate(cell_types):
            ct_markers = processed_markers[cell_type]
            
            # Calculate positive marker scores
            pos_genes = ct_markers['positive']
            if pos_genes:
                # Get column indices for positive markers
                pos_cols = [gene_to_col[gene] for gene in pos_genes if gene in gene_to_col]
                
                if pos_cols:
                    # Extract expression for positive markers
                    if sp.issparse(X_markers):
                        pos_expr = X_markers[:, pos_cols].toarray()
                    else:
                        pos_expr = X_markers[:, pos_cols]
                    
                    # Apply specificity weights
                    weights = np.array([specificity_scores.get(pos_genes[i], 1.0) 
                                      for i, _ in enumerate(pos_cols)], dtype=np.float32)
                    
                    # Weight expression by specificity
                    weighted_expr = pos_expr * weights[np.newaxis, :]
                    
                    # Calculate positive scores: sum(weighted_expr) / sqrt(n_genes)
                    if self.config.normalize_scores:
                        pos_scores = np.sum(weighted_expr, axis=1) / np.sqrt(len(pos_cols))
                    else:
                        pos_scores = np.sum(weighted_expr, axis=1)
                    
                    scores[:, ct_idx] += pos_scores
            
            # Calculate negative marker scores if enabled
            if not self.config.use_positive_only:
                neg_genes = ct_markers['negative']
                if neg_genes:
                    # Get column indices for negative markers
                    neg_cols = [gene_to_col[gene] for gene in neg_genes if gene in gene_to_col]
                    
                    if neg_cols:
                        # Extract expression for negative markers
                        if sp.issparse(X_markers):
                            neg_expr = X_markers[:, neg_cols].toarray()
                        else:
                            neg_expr = X_markers[:, neg_cols]
                        
                        # Apply specificity weights
                        weights = np.array([specificity_scores.get(neg_genes[i], 1.0) 
                                          for i, _ in enumerate(neg_cols)], dtype=np.float32)
                        
                        # Weight expression by specificity
                        weighted_expr = neg_expr * weights[np.newaxis, :]
                        
                        # Calculate negative scores: sum(weighted_expr) / sqrt(n_genes)
                        if self.config.normalize_scores:
                            neg_scores = np.sum(weighted_expr, axis=1) / np.sqrt(len(neg_cols))
                        else:
                            neg_scores = np.sum(weighted_expr, axis=1)
                        
                        # Subtract negative marker contribution
                        scores[:, ct_idx] -= neg_scores
        
        # Convert to DataFrame with proper labels
        cell_scores = pd.DataFrame(
            scores,
            index=self.adata.obs_names,
            columns=cell_types
        )
        
        return cell_scores
    
    def _aggregate_neighborhood_scores(self, cell_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate cell type scores across k-nearest neighbors.
        
        This is the core innovation of FSCType: instead of using individual cell scores,
        we aggregate scores across each cell's neighborhood to reduce noise and 
        improve classification accuracy.
        
        Parameters
        ----------
        cell_scores : pd.DataFrame
            Raw cell type scores with cells as rows and cell types as columns.
            
        Returns
        -------
        aggregated_scores : pd.DataFrame
            Neighborhood-aggregated scores with same structure as input.
        """
        n_cells, n_cell_types = cell_scores.shape
        aggregated_scores = np.zeros_like(cell_scores.values, dtype=np.float32)
        
        # Track cells with no neighbors for reporting
        isolated_cells = []
        
        for i in range(n_cells):
            # Get neighbors and their weights for cell i
            neighbor_indices, neighbor_weights = self.get_neighbors(i)
            
            if len(neighbor_indices) == 0:
                # Cell has no neighbors - use its own scores
                aggregated_scores[i] = cell_scores.iloc[i].values
                isolated_cells.append(i)
                continue
            
            # Get scores for all neighbors
            neighbor_scores = cell_scores.iloc[neighbor_indices].values  # Shape: (n_neighbors, n_cell_types)
            
            if self.config.weight_by_distance and len(neighbor_weights) > 0:
                # Weight by connectivity/similarity from the graph
                
                # Normalize weights to sum to 1 (for proper averaging)
                if np.sum(neighbor_weights) > 0:
                    normalized_weights = neighbor_weights / np.sum(neighbor_weights)
                else:
                    # Fallback to equal weights if all weights are zero
                    normalized_weights = np.ones_like(neighbor_weights) / len(neighbor_weights)
                
                # Weighted average across neighbors for each cell type
                # neighbor_scores: (n_neighbors, n_cell_types)
                # normalized_weights: (n_neighbors,)
                # Result: (n_cell_types,)
                aggregated_scores[i] = np.average(neighbor_scores, axis=0, weights=normalized_weights)
                
            else:
                # Simple average (all neighbors weighted equally)
                aggregated_scores[i] = np.mean(neighbor_scores, axis=0)
        
        # Warn about isolated cells
        if isolated_cells:
            warnings.warn(
                f"Found {len(isolated_cells)} cells with no neighbors. "
                f"Using individual cell scores for these cells."
            )
        
        # Convert back to DataFrame with original labels
        aggregated_df = pd.DataFrame(
            aggregated_scores,
            index=cell_scores.index,
            columns=cell_scores.columns
        )
        
        return aggregated_df
    
    def _make_predictions(self, aggregated_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Convert aggregated scores to final predictions with confidence scores.
        
        For each cell, finds the highest-scoring cell type and calculates a confidence
        score using either gap-based or entropy-based method.
        
        Parameters
        ----------
        aggregated_scores : pd.DataFrame
            Neighborhood-aggregated cell type scores.
            
        Returns
        -------
        predictions : pd.DataFrame
            DataFrame with columns: ['cell_id', 'predicted_type', 'score', 'confidence']
        """
        predictions = []
        
        for cell_id in aggregated_scores.index:
            cell_scores = aggregated_scores.loc[cell_id]
            scores_array = cell_scores.values
            
            # Sort scores in descending order
            sorted_scores = cell_scores.sort_values(ascending=False)
            
            # Get top prediction
            best_cell_type = sorted_scores.index[0]
            best_score = sorted_scores.iloc[0]
            
            # Calculate confidence score using selected method
            if self.config.confidence_method == "gap":
                confidence = self._calculate_gap_confidence(scores_array)
            else:  # entropy
                confidence = self._calculate_entropy_confidence(scores_array)
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                # Low confidence - mark as uncertain
                final_prediction = "Unknown"
                final_confidence = confidence
                final_score = best_score
            else:
                # High confidence - use prediction
                final_prediction = best_cell_type
                final_confidence = confidence
                final_score = best_score
            
            predictions.append({
                'cell_id': cell_id,
                'predicted_type': final_prediction,
                'score': final_score,
                'confidence': final_confidence
            })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Set cell_id as index for easier manipulation
        predictions_df = predictions_df.set_index('cell_id')
        
        return predictions_df
    
    def _stable_softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Numerically stable softmax implementation.
        
        Parameters
        ----------
        scores : np.ndarray
            Input scores for cell types.
        temperature : float, default=1.0
            Temperature parameter controlling distribution sharpness.
            
        Returns
        -------
        probs : np.ndarray
            Normalized probabilities that sum to 1.0.
        """
        # Shift by max to prevent overflow
        shifted = (scores - np.max(scores)) / temperature
        
        # Clip to prevent extreme values
        shifted = np.clip(shifted, -500, 500)
        
        # Calculate exponentials
        exp_scores = np.exp(shifted)
        
        # Normalize to probabilities
        return exp_scores / np.sum(exp_scores)
    
    def _safe_entropy(self, probs: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Shannon entropy with numerical protection.
        
        Parameters
        ----------
        probs : np.ndarray
            Probability distribution.
        epsilon : float, default=1e-10
            Small value to prevent log(0).
            
        Returns
        -------
        entropy : float
            Shannon entropy value.
        """
        # Ensure probabilities sum to 1
        probs = probs / np.sum(probs)
        
        # Clip to prevent log(0)
        safe_probs = np.clip(probs, epsilon, 1.0 - epsilon)
        
        # Calculate entropy
        return -np.sum(probs * np.log(safe_probs))
    
    def _calculate_entropy_confidence(self, scores: np.ndarray) -> float:
        """
        Calculate confidence using entropy with robust edge case handling.
        
        Returns confidence in [0, 1] where:
        - 1.0 = Perfect confidence (low entropy)
        - 0.0 = No confidence (high entropy)
        
        Parameters
        ----------
        scores : np.ndarray
            Cell type scores for a single cell.
            
        Returns
        -------
        confidence : float
            Confidence score in [0, 1] range.
        """
        n_cell_types = len(scores)
        
        # Edge case 1: Single cell type
        if n_cell_types == 1:
            return 1.0
        
        # Edge case 2: All scores identical (within tolerance)
        if np.allclose(scores, scores[0], rtol=1e-9):
            return 0.0  # Maximum uncertainty
        
        # Edge case 3: Check for valid scores
        if not np.isfinite(scores).all():
            warnings.warn("Non-finite scores detected, using uniform confidence")
            return 0.0
        
        # Main calculation with stable softmax
        try:
            probs = self._stable_softmax(scores, self.config.softmax_temperature)
            entropy = self._safe_entropy(probs, self.config.entropy_epsilon)
            
            # Normalize to [0, 1] range
            max_entropy = np.log(n_cell_types)
            confidence = 1.0 - (entropy / max_entropy)
            
            # Ensure valid range
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Entropy calculation failed: {e}, falling back to gap method")
            return self._calculate_gap_confidence(scores)
    
    def _calculate_gap_confidence(self, scores: np.ndarray) -> float:
        """
        Calculate confidence using gap between top two scores (original method).
        
        Parameters
        ----------
        scores : np.ndarray
            Cell type scores for a single cell.
            
        Returns
        -------
        confidence : float
            Confidence score in [0, 1] range.
        """
        n_cell_types = len(scores)
        
        # Sort scores in descending order
        sorted_scores = np.sort(scores)[::-1]
        
        best_score = sorted_scores[0]
        
        # Calculate confidence score
        if n_cell_types == 1:
            # Only one cell type - confidence is 1.0
            confidence = 1.0
        elif best_score <= 0:
            # No positive evidence - confidence is 0.0
            confidence = 0.0
        else:
            # Standard confidence: (max - second_max) / max
            second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            confidence = (best_score - second_best_score) / best_score
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return confidence
    
    def predict(self, 
                markers: Dict[str, Union[List[str], Dict[str, List[str]]]], 
                inplace: bool = True,
                key_added: str = 'fsctype_prediction',
                confidence_key: str = 'fsctype_confidence',
                score_key: str = 'fsctype_score',
                return_scores: bool = False) -> Optional[pd.DataFrame]:
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
        processed_markers = self._prepare_markers(markers)
        
        # Step 2: Calculate marker specificity scores (fixed algorithm)
        specificity_scores = self._calculate_marker_sensitivity(processed_markers)
        
        # Step 3: Calculate raw cell type scores
        cell_scores = self._calculate_cell_scores(processed_markers, specificity_scores)
        
        # Step 4: Aggregate scores across neighborhoods (core innovation)
        aggregated_scores = self._aggregate_neighborhood_scores(cell_scores)
        
        # Step 5: Make final predictions with confidence
        predictions = self._make_predictions(aggregated_scores)
        
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