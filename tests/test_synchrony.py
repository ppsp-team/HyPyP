import pytest

import numpy as np

from hypyp.dataclasses.synchrony import SynchronyTimeSeries, SynchronyForCondition


def test_synchrony_class():
    synchronies = SynchronyTimeSeries()

    with pytest.raises(KeyError):
        synchronies.by_condition['non_existing_task']

def test_synchrony_for_task():
    mat = np.zeros((2, 100))
    mat[1,:] = 1
    synchronies = SynchronyTimeSeries([SynchronyForCondition(mat, 'my_task', ['low', 'high'], 0.1)])
    assert synchronies.by_condition['my_task'].time_series_per_range.shape == mat.shape

    assert len(synchronies.by_condition['my_task'].by_freq_band) == 2
    assert list(synchronies.by_condition['my_task'].by_freq_band.keys())[0] == 'low'
    assert np.nanmean(synchronies.by_condition['my_task'].by_freq_band['low']) == 0
    assert np.nanmean(synchronies.by_condition['my_task'].by_freq_band['high']) == 1

def test_synchrony_times():
    s = SynchronyForCondition(np.ones((2, 101)), 'my_task', ['low', 'high'], 0.1)
    assert s.times[0] == 0
    assert s.times[-1] == 10