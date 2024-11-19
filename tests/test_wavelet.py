import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

def test_instanciate():
    wavelet_name = 'cgau1'
    wavelet = PywaveletsWavelet(wavelet_name=wavelet_name)
    assert wavelet.wtc_smoothing_smooth_factor != 0
    assert wavelet.wtc_smoothing_boxcar_size > 0
    assert wavelet.wavelet_name == wavelet_name
    assert len(wavelet.psi) == 2 ** wavelet.precision
    assert isinstance(wavelet.cwt_params, dict)

def test_resolution():
    wavelet = PywaveletsWavelet(evaluate=False)
    # need to be evaluated before calling .dt
    with pytest.raises(Exception):
        wavelet.psi_dx
    
    wavelet.evaluate_psi()
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1

def test_default_domain():
    wavelet = PywaveletsWavelet()
    assert min(wavelet.psi_x) == wavelet.domain[0]
    assert max(wavelet.psi_x) == wavelet.domain[1]
    
def test_domain():
    lower_bound = -1
    upper_bound = 1
    wavelet = PywaveletsWavelet(lower_bound=lower_bound, upper_bound=upper_bound)

    assert min(wavelet.psi_x) == lower_bound
    assert max(wavelet.psi_x) == upper_bound
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1
    
def test_precision():
    precision = 12
    wavelet = PywaveletsWavelet(precision=precision)
    assert len(wavelet.psi_x) == 2 ** precision
    assert len(wavelet.psi) == 2 ** precision
    
def test_psi():
    wavelet = PywaveletsWavelet(wavelet_name='cmor2,1')
    assert len(wavelet.psi_x) == len(wavelet.psi)
    assert len(wavelet.psi_x) == 2 ** wavelet.precision

    assert np.sum(np.real(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.imag(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.abs(wavelet.psi) * wavelet.psi_dx) == pytest.approx(1)


def test_cwt():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=100).add_sin(0.05)
    res = wavelet.cwt(signal.y, signal.period)
    assert len(res.scales) > 0
    assert len(res.scales) == len(res.frequencies)
    # TODO test something on res.W

def test_wtc():
    wavelet = PywaveletsWavelet()
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    # check that we have a metric for significance
    assert res.sig_metric > 0
    assert res.sig_metric < 1
    
def test_cache():
    wavelet = PywaveletsWavelet(cache_dict=dict())
    assert len(list(wavelet.cache_dict.keys())) == 0
    wavelet.add_cache_item('foo', 'bar')
    assert len(list(wavelet.cache_dict.keys())) == 1
    assert wavelet.get_cache_item('foo') == 'bar'
    wavelet.clear_cache()
    assert len(list(wavelet.cache_dict.keys())) == 0
    assert wavelet.get_cache_item('foo') == None

    time_range = (0,1)
    other_time_range = (0,2)
    zeros = np.zeros((10,))
    pair1 = PairSignals(zeros, zeros, zeros, ch_name1='ch1', ch_name2='ch2', label_s1='subject1', label_s2='subject2', task='my_task', range=time_range)
    pair2 = PairSignals(zeros, zeros, zeros, ch_name1='ch1', ch_name2='ch2', label_s1='subject1', label_s2='subject2', task='my_task', range=other_time_range)
    pair3 = PairSignals(zeros, zeros, zeros, ch_name1='ch1', ch_name2='ch2', label_s1='subject1', label_s2='subject2', task='my_other_task', range=time_range)
    pair4 = PairSignals(zeros, zeros, zeros, ch_name1='ch3', ch_name2='ch4', label_s1='subject1', label_s2='subject2', task='my_task', range=time_range)
    pair5 = PairSignals(zeros, zeros, zeros, ch_name1='ch1', ch_name2='ch2', label_s1='subject5.1', label_s2='subject5.2', task='my_task', range=time_range)
    # add all the keys to a list, then use a set to remove duplicates and make sure we still have the same count
    keys = []
    keys.append(wavelet.get_cache_key_pair(pair1, 0, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair1, 1, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair2, 0, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair3, 0, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair4, 0, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair5, 0, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair5, 1, 'cwt'))
    keys.append(wavelet.get_cache_key_pair(pair5, 1, 'something_else'))
    #print(keys)
    assert len(keys) == len(set(keys))


    