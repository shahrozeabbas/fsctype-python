"""
FSCType: Fast Single-Cell Type Annotation

A Python implementation of the fsctype algorithm for automated cell type annotation
in single-cell RNA sequencing data using k-nearest neighbors.

This package provides a lightweight, efficient implementation that works with 
AnnData objects and requires users to handle their own preprocessing pipeline.

Examples
--------
>>> import scanpy as sc
>>> import fsctype as fsc
>>> 
>>> # User handles preprocessing
>>> sc.pp.normalize_total(adata, target_sum=1e4)
>>> sc.pp.log1p(adata)
>>> sc.pp.scale(adata, max_value=10)
>>> sc.pp.neighbors(adata, n_neighbors=20)
>>> 
# FSCType handles annotation
>>> markers = {
>>>     'T_cell': {'positive': ['CD3D', 'CD3E'], 'negative': ['CD19']},
>>>     'B_cell': {'positive': ['CD19', 'MS4A1'], 'negative': ['CD3D']}
>>> }
>>> 
>>> # Configure with entropy-based confidence (default)
>>> config = fsc.FSCTypeConfig(
>>>     n_neighbors=20, 
>>>     confidence_method='entropy',
>>>     softmax_temperature=1.0
>>> )
>>> predictor = fsc.FSCType(adata, config)
>>> predictions = predictor.predict(markers)
"""

from .core import FSCType
from .config import FSCTypeConfig

__version__ = "0.1.0"
__author__ = "Shahroze Abbas"
__email__ = "shahroze@datatecnica.com"

__all__ = [
    "FSCType",
    "FSCTypeConfig",
] 