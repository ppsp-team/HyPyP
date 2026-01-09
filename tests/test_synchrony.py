import pytest

import numpy as np

from hypyp.dataclasses.synchrony import SynchronyTimeSeries, SynchronyForATask


def test_synchrony_class():
    synchronies = SynchronyTimeSeries()

    with pytest.raises(KeyError):
        synchronies.by_task['non_existing_task']

def test_synchrony_for_task():
    mat = np.zeros((2, 100))
    mat[1,:] = 1
    synchronies = SynchronyTimeSeries([SynchronyForATask(mat, 'my_task', ['low', 'high'], 0.1)])
    assert synchronies.by_task['my_task'].time_series_per_range.shape == mat.shape

    assert len(synchronies.by_task['my_task'].by_freq_range) == 2
    assert list(synchronies.by_task['my_task'].by_freq_range.keys())[0] == 'low'
    assert np.nanmean(synchronies.by_task['my_task'].by_freq_range['low']) == 0
    assert np.nanmean(synchronies.by_task['my_task'].by_freq_range['high']) == 1

def test_synchrony_times():
    s = SynchronyForATask(np.ones((2, 101)), 'my_task', ['low', 'high'], 0.1)
    assert s.times[0] == 0
    assert s.times[-1] == 10