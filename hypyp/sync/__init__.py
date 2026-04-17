#!/usr/bin/env python
# coding=utf-8

"""
Synchrony and connectivity metrics for hyperscanning analysis.

This module provides a collection of connectivity metrics that can be used
to measure neural synchronization between participants.
"""

from typing import Optional

from .base import (
    BaseMetric, multiply_conjugate, multiply_conjugate_time, multiply_product,
    multiply_conjugate_torch, multiply_conjugate_time_torch,
)
from .plv import PLV
from .ccorr import CCorr
from .accorr import ACCorr
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
    'accorr': ACCorr,
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
    'multiply_conjugate',
    'multiply_conjugate_time',
    'multiply_product',
    'multiply_conjugate_torch',
    'multiply_conjugate_time_torch',
    # Metric classes
    'PLV',
    'CCorr',
    'ACCorr',
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


def get_metric(mode: str, optimization: Optional[str] = None,
               priority: Optional[list] = None) -> BaseMetric:
    """
    Get a connectivity metric instance by name.

    Parameters
    ----------
    mode : str
        Name of the connectivity metric. One of: 'plv', 'ccorr', 'accorr',
        'coh', 'imcoh', 'pli', 'wpli', 'envcorr', 'powcorr'.
    optimization : str, optional
        Optimization strategy. Options: None, 'auto', 'numba', 'torch',
        'metal', 'cuda_kernel'. See BaseMetric for fallback behavior.
    priority : list of str, optional
        Custom backend priority for ``'auto'`` mode. Overrides the default
        ``AUTO_PRIORITY`` table. Example: ``['metal', 'torch', 'numba']``.

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
    >>> plv = get_metric('plv', optimization='auto')          # benchmark-driven
    >>> pli = get_metric('pli', optimization='auto',
    ...                  priority=['numba', 'metal'])          # custom priority
    """
    mode_lower = mode.lower()
    if mode_lower not in METRICS:
        available = ', '.join(METRICS.keys())
        raise ValueError(f"Unknown metric mode '{mode}'. Available: {available}")

    return METRICS[mode_lower](optimization=optimization, priority=priority)
