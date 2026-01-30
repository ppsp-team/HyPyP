#!/usr/bin/env python
# coding=utf-8

"""
Synchrony and connectivity metrics for hyperscanning analysis.

This module provides a collection of connectivity metrics that can be used
to measure neural synchronization between participants.
"""

from .base import BaseMetric, detect_backend, multiply_conjugate, multiply_conjugate_time, multiply_product
from .plv import PLV
from .ccorr import CCorr
from .accorr import ACorr
from .coh import Coh
from .imaginary_coh import ImCoh
from .pli import PLI
from .wpli import WPLI
from .envelope_corr import EnvCorr
from .pow_corr import PowCorr

# Metric registry: maps mode names to metric classes
METRICS = {
    'plv': PLV,
    'ccorr': CCorr,
    'accorr': ACorr,
    'coh': Coh,
    'imcoh': ImCoh,
    'pli': PLI,
    'wpli': WPLI,
    'envcorr': EnvCorr,
    'powcorr': PowCorr,
}

__all__ = [
    # Base classes and utilities
    'BaseMetric',
    'detect_backend',
    'multiply_conjugate',
    'multiply_conjugate_time',
    'multiply_product',
    # Metric classes
    'PLV',
    'CCorr',
    'ACorr',
    'Coh',
    'ImCoh',
    'PLI',
    'WPLI',
    'EnvCorr',
    'PowCorr',
    # Registry
    'METRICS',
    'get_metric',
]


def get_metric(mode: str, backend: str = 'numpy') -> BaseMetric:
    """
    Get a connectivity metric instance by name.
    
    Parameters
    ----------
    mode : str
        Name of the connectivity metric. One of: 'plv', 'ccorr', 'accorr',
        'coh', 'imcoh', 'pli', 'wpli', 'envcorr', 'powcorr'.
        
    backend : str, optional
        Computation backend to use. One of: 'numpy', 'numba', 'torch'.
        Default is 'numpy'.
    
    Returns
    -------
    metric : BaseMetric
        An instance of the requested metric class.
    
    Raises
    ------
    ValueError
        If the mode is not recognized.
    
    Examples
    --------
    >>> from hypyp.sync import get_metric
    >>> plv = get_metric('plv', backend='numpy')
    >>> result = plv.compute(complex_signal, n_samp, transpose_axes)
    """
    mode_lower = mode.lower()
    if mode_lower not in METRICS:
        available = ', '.join(METRICS.keys())
        raise ValueError(f"Unknown metric mode '{mode}'. Available: {available}")
    
    return METRICS[mode_lower](backend=backend)
