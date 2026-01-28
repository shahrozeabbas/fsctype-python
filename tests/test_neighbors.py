"""Unit tests for neighbor graph handling functions."""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from fsctype.neighbors import prepare_neighbors, aggregate_scores


class TestPrepareNeighbors:
    """Tests for prepare_neighbors function."""
    
    def test_basic_preparation(self):
        """Basic neighbor preparation should work."""
        n_cells = 10
        k = 3
        
        # Create simple connectivity matrix
        conn_matrix = sparse.csr_matrix(np.eye(n_cells, dtype=np.float32))
        # Add some connections
        conn_matrix[0, 1] = 0.8
        conn_matrix[0, 2] = 0.6
        conn_matrix[0, 3] = 0.4
        
        neighbors, distances = prepare_neighbors(conn_matrix, k)
        
        assert len(neighbors) == n_cells
        assert len(distances) == n_cells
        assert len(neighbors[0]) <= k
    
    def test_top_k_selection(self):
        """Should select top k neighbors."""
        n_cells = 5
        k = 2
        
        # Create matrix with multiple connections
        conn_matrix = sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
        conn_matrix[0, 1] = 0.9
        conn_matrix[0, 2] = 0.8
        conn_matrix[0, 3] = 0.7
        conn_matrix[0, 4] = 0.6
        
        neighbors, distances = prepare_neighbors(conn_matrix, k)
        
        assert len(neighbors[0]) == k
        # Top k should be neighbors 1 and 2 (highest weights)
        assert 1 in neighbors[0]
        assert 2 in neighbors[0]
    
    def test_fewer_than_k_neighbors(self):
        """Should take all neighbors if fewer than k."""
        n_cells = 5
        k = 10  # More than available
        
        conn_matrix = sparse.csr_matrix(np.eye(n_cells, dtype=np.float32))
        conn_matrix[0, 1] = 0.5
        
        neighbors, distances = prepare_neighbors(conn_matrix, k)
        
        # Should have at least 1 neighbor (itself + connection)
        assert len(neighbors[0]) >= 1
    
    def test_no_neighbors(self):
        """Cells with no neighbors should have empty arrays."""
        n_cells = 3
        k = 2
        
        # Identity matrix only (self-connections)
        conn_matrix = sparse.csr_matrix(np.eye(n_cells, dtype=np.float32))
        
        neighbors, distances = prepare_neighbors(conn_matrix, k)
        
        # Should handle gracefully
        assert len(neighbors) == n_cells
        assert len(distances) == n_cells


class TestAggregateScores:
    """Tests for aggregate_scores function."""
    
    def test_basic_aggregation(self):
        """Basic score aggregation should work."""
        n_cells = 5
        n_types = 2
        
        # Create simple scores
        cell_scores = pd.DataFrame(
            np.random.randn(n_cells, n_types),
            columns=['TypeA', 'TypeB'],
            index=[f'CELL_{i}' for i in range(n_cells)]
        )
        
        # Create simple neighbors (each cell neighbors with next)
        neighbors = []
        distances = []
        for i in range(n_cells):
            if i < n_cells - 1:
                neighbors.append(np.array([i + 1], dtype=np.int32))
                distances.append(np.array([1.0], dtype=np.float32))
            else:
                neighbors.append(np.array([], dtype=np.int32))
                distances.append(np.array([], dtype=np.float32))
        
        aggregated = aggregate_scores(cell_scores, neighbors, distances, False)
        
        assert aggregated.shape == cell_scores.shape
        assert list(aggregated.columns) == list(cell_scores.columns)
        assert list(aggregated.index) == list(cell_scores.index)
    
    def test_distance_weighting(self):
        """Distance weighting should affect aggregation."""
        cell_scores = pd.DataFrame({
            'TypeA': [10.0, 5.0, 1.0],
            'TypeB': [1.0, 5.0, 10.0],
        }, index=['CELL_0', 'CELL_1', 'CELL_2'])
        
        # CELL_0 neighbors with CELL_1 (weight 0.8) and CELL_2 (weight 0.2)
        neighbors = [
            np.array([1, 2], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]
        distances = [
            np.array([0.8, 0.2], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]
        
        aggregated_weighted = aggregate_scores(cell_scores, neighbors, distances, True)
        aggregated_unweighted = aggregate_scores(cell_scores, neighbors, distances, False)
        
        # Weighted should be closer to CELL_1's scores (higher weight)
        assert aggregated_weighted.loc['CELL_0', 'TypeA'] > aggregated_unweighted.loc['CELL_0', 'TypeA']
    
    def test_isolated_cells(self):
        """Isolated cells should use their own scores."""
        cell_scores = pd.DataFrame({
            'TypeA': [10.0, 5.0],
            'TypeB': [1.0, 5.0],
        }, index=['CELL_0', 'CELL_1'])
        
        # CELL_0 has no neighbors
        neighbors = [
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]
        distances = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]
        
        aggregated = aggregate_scores(cell_scores, neighbors, distances, False)
        
        # Should use own scores
        assert aggregated.loc['CELL_0', 'TypeA'] == cell_scores.loc['CELL_0', 'TypeA']
        assert aggregated.loc['CELL_1', 'TypeB'] == cell_scores.loc['CELL_1', 'TypeB']
    
    def test_uniform_weighting(self):
        """Uniform weighting should average neighbors."""
        cell_scores = pd.DataFrame({
            'TypeA': [10.0, 5.0, 1.0],
        }, index=['CELL_0', 'CELL_1', 'CELL_2'])
        
        # CELL_0 neighbors with CELL_1 and CELL_2
        neighbors = [
            np.array([1, 2], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]
        distances = [
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]
        
        aggregated = aggregate_scores(cell_scores, neighbors, distances, False)
        
        # Should be average of neighbors
        expected = (cell_scores.loc['CELL_1', 'TypeA'] + cell_scores.loc['CELL_2', 'TypeA']) / 2
        assert aggregated.loc['CELL_0', 'TypeA'] == pytest.approx(expected)
    
    def test_zero_weight_sum(self):
        """Zero weight sum should fallback to equal weights."""
        cell_scores = pd.DataFrame({
            'TypeA': [10.0, 5.0],
        }, index=['CELL_0', 'CELL_1'])
        
        neighbors = [
            np.array([1], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]
        distances = [
            np.array([0.0], dtype=np.float32),  # Zero weight
            np.array([], dtype=np.float32),
        ]
        
        # Should not crash
        aggregated = aggregate_scores(cell_scores, neighbors, distances, True)
        assert aggregated.shape == cell_scores.shape
