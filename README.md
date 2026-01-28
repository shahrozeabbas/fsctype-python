# FSCType: Fast Single-Cell Type Annotation

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)
![Build](https://img.shields.io/badge/build-passing-green.svg)

A Python implementation of the FSCType algorithm for automated cell type annotation in single-cell RNA sequencing data using k-nearest neighbors.

## Features

- **Fast & Efficient**: Optimized for large single-cell datasets with sparse matrix support
- **Scanpy Integration**: Seamless integration with AnnData objects and scanpy workflow
- **Flexible Markers**: Supports multiple marker input formats (positive/negative genes)
- **Neighborhood Aggregation**: Uses k-nearest neighbors for robust predictions
- **Confidence Scoring**: Provides prediction confidence scores for quality control

## Installation

### From PyPI (coming soon)
```bash
pip install fsctype
```

### From Source
```bash
git clone https://github.com/shahrozeabbas/fsctype-python.git
cd fsctype-python
pip install -e .
```

## Quick Start

```python
import scanpy as sc
import fsctype as fsc

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Preprocessing (user responsibility)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=20)

# Define cell type markers
markers = {
    'T_cell': {
        'positive': ['CD3D', 'CD3E', 'CD3G'],
        'negative': ['CD19', 'CD79A']
    },
    'B_cell': {
        'positive': ['CD19', 'MS4A1', 'CD79A'],
        'negative': ['CD3D']
    },
    'NK_cell': {
        'positive': ['GNLY', 'NKG7', 'KLRD1'],
        'negative': ['CD3D', 'CD19']
    }
}

# Configure and run FSCType
config = fsc.FSCTypeConfig(
    n_neighbors=20,
    weight_by_distance=True,
    confidence_threshold=0.5
)

model = fsc.FSCType(adata, config)
predictions = model.predict(markers)

# Results are stored in adata.obs
print(adata.obs[['fsctype_prediction', 'fsctype_confidence']].head())
```

## API Documentation

### FSCTypeConfig

Configuration class for FSCType parameters:

```python
config = fsc.FSCTypeConfig(
    n_neighbors=20,           # Number of neighbors for aggregation
    weight_by_distance=True,  # Weight neighbors by distance
    confidence_threshold=0.5, # Minimum confidence for predictions
    confidence_method='entropy',    # 'gap' or 'entropy'
    softmax_temperature=1.0,        # For entropy method
    expression_layer='X'      # AnnData layer to use
)
```

### FSCType

Main prediction class:

```python
model = fsc.FSCType(adata, config)

# Basic prediction (returns predictions DataFrame)
predictions = model.predict(markers, inplace=False)

# Get both predictions and detailed cell type scores
predictions, scores = model.predict(markers, inplace=False, return_scores=True)

# Store results directly in adata.obs (default behavior)
model.predict(markers, inplace=True)  # Adds to adata.obs['fsctype_prediction']
```

## Confidence Methods

FSCType supports two confidence scoring methods:

- **gap** (simple): `(best_score - second_best) / best_score`. Only considers top two candidates.
- **entropy** (default): Uses normalized Shannon entropy across all cell types. More nuanced for multi-class scenarios.

Set via `confidence_method` parameter. For entropy, `softmax_temperature` controls distribution sharpness (lower = sharper/higher confidence).

## Marker Format

FSCType supports flexible marker definitions:

```python
# Full format (recommended)
markers = {
    'cell_type': {
        'positive': ['GENE1', 'GENE2'],
        'negative': ['GENE3', 'GENE4']
    }
}

# Simple format (positive only)
markers = {
    'cell_type': ['GENE1', 'GENE2', 'GENE3']
}
```

## Requirements

- Python >= 3.10
- numpy >= 1.23.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- anndata >= 0.9.0

Optional:
- scanpy ≥ 1.8.0 (for neighbor computation)
- matplotlib ≥ 3.4.0 (for plotting)

## Citation

If you use FSCType in your research, please cite:

```bibtex
@software{fsctype_python,
  title={FSCType: Fast Single-Cell Type Annotation},
  author={Abbas, Shahroze},
  year={2024},
  url={https://github.com/shahrozeabbas/fsctype-python}
}
```

## License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines and submit pull requests.

## Support

- 🐛 [Report bugs](https://github.com/shahrozeabbas/fsctype-python/issues)
- 💡 [Request features](https://github.com/shahrozeabbas/fsctype-python/issues)
- 📖 [Documentation](https://fsctype-python.readthedocs.io) 
