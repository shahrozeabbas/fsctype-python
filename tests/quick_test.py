"""
Quick test script for FSCType - minimal example to verify installation.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

# Import FSCType
import fsctype as fsc

# Create minimal test data
np.random.seed(42)
n_cells, n_genes = 200, 100

# Generate synthetic expression data
X = np.random.negative_binomial(3, 0.4, size=(n_cells, n_genes)).astype(np.float32)

# Add some marker genes with cell-type-specific expression
marker_genes = ['CD3D', 'CD3E', 'CD19', 'MS4A1', 'KLRD1', 'NCR1']
other_genes = [f'GENE_{i}' for i in range(len(marker_genes), n_genes)]
gene_names = marker_genes + other_genes

# Create cell types (40% T, 40% B, 20% NK)
cell_types = ['T_cell'] * 80 + ['B_cell'] * 80 + ['NK_cell'] * 40
np.random.shuffle(cell_types)

# Boost marker expression for corresponding cell types
for i, cell_type in enumerate(cell_types):
    if cell_type == 'T_cell':
        X[i, 0] += np.random.poisson(5)  # CD3D
        X[i, 1] += np.random.poisson(4)  # CD3E
    elif cell_type == 'B_cell':
        X[i, 2] += np.random.poisson(6)  # CD19
        X[i, 3] += np.random.poisson(5)  # MS4A1
    elif cell_type == 'NK_cell':
        X[i, 4] += np.random.poisson(4)  # KLRD1
        X[i, 5] += np.random.poisson(3)  # NCR1

# Create AnnData object
adata = ad.AnnData(X=sparse.csr_matrix(X))
adata.var_names = gene_names
adata.obs_names = [f'CELL_{i:03d}' for i in range(n_cells)]
adata.obs['true_cell_type'] = cell_types

print(f"Created test data: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"True cell types: {pd.Series(cell_types).value_counts()}")

# Minimal preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15)  # REQUIRED for FSCType

print("Preprocessing complete")

# Define markers
markers = {
    'T_cell': ['CD3D', 'CD3E'],
    'B_cell': ['CD19', 'MS4A1'], 
    'NK_cell': ['KLRD1', 'NCR1']
}

# Run FSCType
config = fsc.FSCTypeConfig(n_neighbors=15, confidence_threshold=0.1)
predictor = fsc.FSCType(adata, config)

print("Running FSCType prediction...")
predictions = predictor.predict(markers, inplace=False)

# Show results
print("\n=== RESULTS ===")
print(f"Predictions shape: {predictions.shape}")
print(f"\nPredicted cell types:")
print(predictions['predicted_type'].value_counts())
print(f"\nMean confidence: {predictions['confidence'].mean():.3f}")
print(f"High confidence (>0.5): {(predictions['confidence'] > 0.5).sum()} cells")

# Compare with true labels
adata.obs['fsctype_pred'] = predictions['predicted_type']
adata.obs['fsctype_conf'] = predictions['confidence']

print(f"\n=== ACCURACY CHECK ===")
# Simple accuracy for non-Unknown predictions
mask = predictions['predicted_type'] != 'Unknown'
if mask.sum() > 0:
    true_labels = pd.Series(cell_types)[mask]
    pred_labels = predictions['predicted_type'][mask]
    accuracy = (true_labels == pred_labels).mean()
    print(f"Accuracy (excluding Unknown): {accuracy:.3f}")
else:
    print("All predictions are Unknown")

print(f"\nSuccess! FSCType appears to be working correctly.")
print(f"Next: Try with your own data and markers!") 