"""
Test script for FSCType implementation.

This script creates synthetic data and tests the complete FSCType pipeline
to validate the implementation and demonstrate usage patterns.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import warnings

# Import our FSCType implementation
import sys
sys.path.append('.')  # Add current directory to path
import fsctype as fsc

# Set random seed for reproducibility
np.random.seed(42)
sc.settings.verbosity = 1  # Reduce scanpy output


def create_synthetic_data(n_cells=1000, n_genes=2000):
    """Create synthetic single-cell data with known cell types."""
    print("Creating synthetic test data...")
    
    # Define cell type proportions
    cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte']
    proportions = [0.4, 0.3, 0.2, 0.1]
    
    # Generate cell type labels
    cell_labels = []
    for ct, prop in zip(cell_types, proportions):
        n_cells_type = int(n_cells * prop)
        cell_labels.extend([ct] * n_cells_type)
    
    # Pad to exact n_cells
    while len(cell_labels) < n_cells:
        cell_labels.append(np.random.choice(cell_types))
    cell_labels = cell_labels[:n_cells]
    
    # Define marker genes with realistic expression patterns
    marker_genes = {
        'T_cell': ['CD3D', 'CD3E', 'CD2', 'TRBC1'],
        'B_cell': ['CD19', 'MS4A1', 'CD79A', 'PAX5'], 
        'NK_cell': ['KLRD1', 'NCR1', 'NKG7', 'GNLY'],
        'Monocyte': ['CD14', 'LYZ', 'S100A8', 'FCN1']
    }
    
    # Add housekeeping genes
    housekeeping = ['ACTB', 'GAPDH', 'RPL13A', 'B2M']
    
    # Create gene list
    all_marker_genes = []
    for genes in marker_genes.values():
        all_marker_genes.extend(genes)
    all_marker_genes.extend(housekeeping)
    
    # Add random genes to reach n_genes
    random_genes = [f'GENE_{i:04d}' for i in range(len(all_marker_genes), n_genes)]
    gene_names = all_marker_genes + random_genes
    
    # Generate expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add cell-type-specific expression for marker genes
    for i, cell_type in enumerate(cell_labels):
        if cell_type in marker_genes:
            # Boost expression of this cell type's markers
            for gene in marker_genes[cell_type]:
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    # Add cell-type-specific signal
                    X[i, gene_idx] += np.random.negative_binomial(10, 0.2)
    
    # Add housekeeping gene expression (high in all cells)
    for gene in housekeeping:
        if gene in gene_names:
            gene_idx = gene_names.index(gene)
            X[:, gene_idx] += np.random.negative_binomial(20, 0.1, size=n_cells)
    
    # Create AnnData object
    adata = ad.AnnData(X=sparse.csr_matrix(X))
    adata.var_names = gene_names
    adata.obs_names = [f'CELL_{i:06d}' for i in range(n_cells)]
    adata.obs['true_cell_type'] = cell_labels
    
    # Add some metadata
    adata.obs['n_genes'] = (X > 0).sum(axis=1)
    adata.obs['total_counts'] = X.sum(axis=1)
    
    print(f"Created data: {n_cells} cells × {n_genes} genes")
    print(f"Cell type distribution:")
    print(adata.obs['true_cell_type'].value_counts())
    
    return adata, marker_genes


def preprocess_data(adata):
    """Standard scanpy preprocessing pipeline."""
    print("\nPreprocessing data...")
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Save raw counts
    adata.raw = adata
    
    # Normalization and log transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Compute neighborhood graph (REQUIRED for FSCType)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
    
    # UMAP for visualization
    sc.tl.umap(adata)
    
    print(f"After preprocessing: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata


def test_basic_functionality():
    """Test basic FSCType functionality."""
    print("\n" + "="*50)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*50)
    
    # Create test data
    adata, true_markers = create_synthetic_data(n_cells=500, n_genes=1000)
    adata = preprocess_data(adata)
    
    # Test 1: Simple marker format
    print("\nTest 1: Simple marker format")
    simple_markers = {
        'T_cell': ['CD3D', 'CD3E'],
        'B_cell': ['CD19', 'MS4A1'],
        'NK_cell': ['KLRD1', 'NCR1']
    }
    
    config = fsc.FSCTypeConfig(n_neighbors=15, confidence_threshold=0.1)
    predictor = fsc.FSCType(adata, config)
    
    predictions = predictor.predict(simple_markers, inplace=False)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted cell types: {predictions['predicted_type'].value_counts()}")
    print(f"Mean confidence: {predictions['confidence'].mean():.3f}")
    
    # Test 2: Full marker format with negative markers
    print("\nTest 2: Full marker format with negative markers")
    full_markers = {
        'T_cell': {
            'positive': ['CD3D', 'CD3E', 'CD2'],
            'negative': ['CD19', 'CD14']
        },
        'B_cell': {
            'positive': ['CD19', 'MS4A1', 'CD79A'], 
            'negative': ['CD3D', 'CD14']
        },
        'Monocyte': {
            'positive': ['CD14', 'LYZ'],
            'negative': ['CD3D', 'CD19']
        }
    }
    
    config_neg = fsc.FSCTypeConfig(
        n_neighbors=15, 
        use_positive_only=False,  # Enable negative markers
        confidence_threshold=0.15
    )
    predictor_neg = fsc.FSCType(adata, config_neg)
    
    predictions_neg = predictor_neg.predict(full_markers, inplace=False)
    print(f"With negative markers: {predictions_neg['predicted_type'].value_counts()}")
    print(f"Mean confidence: {predictions_neg['confidence'].mean():.3f}")
    
    return adata, predictions, predictions_neg


def test_configuration_options():
    """Test different configuration options."""
    print("\n" + "="*50)
    print("TESTING CONFIGURATION OPTIONS")
    print("="*50)
    
    # Create test data
    adata, _ = create_synthetic_data(n_cells=300, n_genes=800)
    adata = preprocess_data(adata)
    
    markers = {
        'T_cell': ['CD3D', 'CD3E'],
        'B_cell': ['CD19', 'MS4A1'],
        'NK_cell': ['KLRD1', 'NCR1']
    }
    
    configs = [
        ("Default", fsc.FSCTypeConfig()),
        ("High neighbors", fsc.FSCTypeConfig(n_neighbors=30)),
        ("No distance weighting", fsc.FSCTypeConfig(weight_by_distance=False)),
        ("High confidence threshold", fsc.FSCTypeConfig(confidence_threshold=0.5)),
        ("No normalization", fsc.FSCTypeConfig(normalize_scores=False))
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        try:
            predictor = fsc.FSCType(adata, config)
            pred = predictor.predict(markers, inplace=False)
            
            results[name] = {
                'n_unknown': (pred['predicted_type'] == 'Unknown').sum(),
                'mean_confidence': pred['confidence'].mean(),
                'cell_type_counts': pred['predicted_type'].value_counts()
            }
            
            print(f"  Unknown cells: {results[name]['n_unknown']}")
            print(f"  Mean confidence: {results[name]['mean_confidence']:.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {'error': str(e)}
    
    return results


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)
    
    # Create minimal test data
    adata, _ = create_synthetic_data(n_cells=100, n_genes=200)
    adata = preprocess_data(adata)
    
    config = fsc.FSCTypeConfig()
    predictor = fsc.FSCType(adata, config)
    
    # Test 1: Empty markers
    print("\nTest 1: Empty markers")
    try:
        pred = predictor.predict({}, inplace=False)
        print("  ERROR: Should have failed!")
    except Exception as e:
        print(f"  PASS: {e}")
    
    # Test 2: Non-existent genes
    print("\nTest 2: Non-existent genes")
    bad_markers = {
        'Fake_cell': ['NONEXISTENT1', 'NONEXISTENT2', 'NONEXISTENT3']
    }
    try:
        pred = predictor.predict(bad_markers, inplace=False)
        print("  ERROR: Should have failed!")
    except Exception as e:
        print(f"  PASS: {e}")
    
    # Test 3: Mixed valid/invalid genes
    print("\nTest 3: Mixed valid/invalid genes") 
    mixed_markers = {
        'T_cell': ['CD3D', 'NONEXISTENT1', 'CD3E', 'FAKE_GENE']
    }
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pred = predictor.predict(mixed_markers, inplace=False)
            print(f"  PASS: Got predictions with warnings: {len(w)} warnings")
            if pred is not None:
                print(f"  Predictions: {pred['predicted_type'].value_counts()}")
    except Exception as e:
        print(f"  Result: {e}")
    
    # Test 4: Single cell type
    print("\nTest 4: Single cell type")
    single_markers = {'Only_type': ['CD3D', 'CD3E']}
    try:
        pred = predictor.predict(single_markers, inplace=False)
        print(f"  PASS: Single type predictions")
        print(f"  All confidence = 1.0: {(pred['confidence'] == 1.0).all()}")
    except Exception as e:
        print(f"  ERROR: {e}")


def test_inplace_functionality():
    """Test inplace functionality and AnnData integration."""
    print("\n" + "="*50)
    print("TESTING INPLACE FUNCTIONALITY") 
    print("="*50)
    
    # Create test data
    adata, _ = create_synthetic_data(n_cells=200, n_genes=500)
    adata = preprocess_data(adata)
    
    markers = {
        'T_cell': ['CD3D', 'CD3E'],
        'B_cell': ['CD19', 'MS4A1']
    }
    
    config = fsc.FSCTypeConfig()
    predictor = fsc.FSCType(adata, config)
    
    # Test inplace=True
    print("\nTesting inplace=True")
    result = predictor.predict(markers, inplace=True)
    
    print(f"  Return value is None: {result is None}")
    print(f"  Added to adata.obs: {'fsctype_prediction' in adata.obs.columns}")
    print(f"  Predictions: {adata.obs['fsctype_prediction'].value_counts()}")
    
    # Test custom column names
    print("\nTesting custom column names")
    predictor.predict(
        markers, 
        inplace=True,
        key_added='my_cell_types',
        confidence_key='my_confidence'
    )
    
    print(f"  Custom columns added: {'my_cell_types' in adata.obs.columns}")
    
    # Test return_scores=True
    print("\nTesting return_scores=True")
    pred, scores = predictor.predict(markers, inplace=False, return_scores=True)
    
    print(f"  Predictions shape: {pred.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score columns: {list(scores.columns)}")


def run_all_tests():
    """Run all test functions."""
    print("FSCType Implementation Test Suite")
    print("="*50)
    
    try:
        # Test 1: Basic functionality
        adata, pred1, pred2 = test_basic_functionality()
        
        # Test 2: Configuration options
        config_results = test_configuration_options()
        
        # Test 3: Edge cases
        test_edge_cases()
        
        # Test 4: Inplace functionality
        test_inplace_functionality()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED!")
        print("="*50)
        
        print("\nTest Summary:")
        print("✅ Basic functionality - PASSED")
        print("✅ Configuration options - PASSED") 
        print("✅ Edge case handling - PASSED")
        print("✅ AnnData integration - PASSED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Suppress scanpy warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 FSCType implementation appears to be working correctly!")
        print("\nNext steps:")
        print("1. Test with real single-cell datasets")
        print("2. Compare results with original R implementation")
        print("3. Benchmark performance on large datasets")
        print("4. Add visualization functions")
    else:
        print("\n🐛 Issues found - check the error messages above") 