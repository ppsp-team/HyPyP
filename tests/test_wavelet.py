import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.wavelet_implementations.pywavelets_wavelet import PywaveletsWavelet

def test_instanciate():
    wavelet_name = 'cgau1'
    wavelet = PywaveletsWavelet(wavelet_name=wavelet_name)
    assert wavelet.wtc_smoothing_boxcar_size is None
    assert wavelet.wavelet_name == wavelet_name
    assert len(wavelet.psi) == 2 ** 10
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
    
def test_psi():
    wavelet = PywaveletsWavelet(wavelet_name='cmor2,1')
    assert len(wavelet.psi_x) == len(wavelet.psi)
    assert len(wavelet.psi_x) == 2 ** 10

    assert np.sum(np.real(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.imag(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.abs(wavelet.psi) * wavelet.psi_dx) == pytest.approx(1)

def test_number_of_scales():
    assert len(PywaveletsWavelet(periods_range=(2, 4)).get_periods()) == 12
    assert len(PywaveletsWavelet(periods_range=(4, 8)).get_periods()) == 12
    assert len(PywaveletsWavelet(periods_range=(2, 8)).get_periods()) == 24
    

def test_cwt():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=100).add_sin(0.05)
    res = wavelet.cwt(signal.y, signal.period)
    assert len(res.scales) > 0
    assert len(res.scales) == len(res.frequencies)
    # TODO test something on res.W

def test_wtc():
    wavelet = PywaveletsWavelet(disable_caching=True)
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert res.coherence_metric > 0
    assert res.coherence_metric < 1
    
def test_cache_key():
    assert 'key_foo_bar' in PywaveletsWavelet().get_cache_key('foo', 'bar')
    assert PywaveletsWavelet(disable_caching=True).get_cache_key('foo', 'bar') is None
    
def test_cache():
    wavelet = PywaveletsWavelet(cache=dict())
    assert len(list(wavelet.cache.keys())) == 0
    wavelet.add_cache_item('foo', 'bar')
    assert len(list(wavelet.cache.keys())) == 1
    assert wavelet.get_cache_item('foo') == 'bar'
    wavelet.clear_cache()
    assert len(list(wavelet.cache.keys())) == 0
    assert wavelet.get_cache_item('foo') == None

    time_range = (0,1)
    other_time_range = (0,2)
    zeros = np.zeros((10,))
    x = np.arange(len(zeros))
    pair1 = PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', task='my_task', time_range=time_range)
    pair2 = PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', task='my_task', time_range=other_time_range)
    pair3 = PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', task='my_other_task', time_range=time_range)
    pair4 = PairSignals(x, zeros, zeros, label_ch1='ch3', label_ch2='ch4', label_s1='subject1', label_s2='subject2', task='my_task', time_range=time_range)
    pair5 = PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject5.1', label_s2='subject5.2', task='my_task', time_range=time_range)
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


def test_wtc_coi_masked():
    wavelet = PywaveletsWavelet(disable_caching=True)
    signal = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal.x, signal.y, signal.y))
    assert res.wtc_masked is not None
    assert res.wtc_masked.mask[0,0] == True
    assert res.wtc_masked.mask[0,len(signal.x)//2] == False

def test_periods_frequencies_range():    
    frequencies_range = np.array([5., 1.])
    periods_range = 1 / frequencies_range
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()

    wavelet1 = PywaveletsWavelet(periods_range=tuple(periods_range), disable_caching=True)
    res1 = wavelet1.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert np.all(res1.frequencies[[0,-1]] == pytest.approx(frequencies_range))

    wavelet2 = PywaveletsWavelet(frequencies_range=tuple(frequencies_range), disable_caching=True)
    res2 = wavelet2.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert np.all(res2.frequencies[[0,-1]] == pytest.approx(frequencies_range))
    
def test_smooth_in_scale_window():
    assert np.sum(BaseWavelet.get_boxcar_window(0.6, 10)) == 1#pytest.approx(1)
    assert len(BaseWavelet.get_boxcar_window(1, 1/2)) == 2
    assert len(BaseWavelet.get_boxcar_window(1.1, 1/2)) == 3

    # dirac
    assert len(BaseWavelet.get_boxcar_window(1/12, 1/12)) == 1

    win10 = BaseWavelet.get_boxcar_window(10, 1)
    assert len(win10) == 10
    assert np.mean(win10) == win10[0]

    win10_plus = BaseWavelet.get_boxcar_window(10.1, 1)
    assert len(win10_plus) == 11
    assert np.mean(win10_plus) > win10_plus[0]

def test_fft_kwargs():
    d = BaseWavelet.get_fft_kwargs(np.zeros((10,)))
    assert d['n'] == 16

    d = BaseWavelet.get_fft_kwargs(np.zeros((10,)), extra='foo')
    assert d['extra'] == 'foo'
    
def test_smoothing():
    dt = 0.1
    dj = 0.1
    wavelet = PywaveletsWavelet()
    scales = wavelet.get_scales(dt, dj)
    W = np.zeros((len(scales), 200))
    W[:, np.arange(0, W.shape[1], 2)] = 1
    smoothed = wavelet.smoothing(W, dt, dj, scales)
    assert np.all(smoothed > 0)
    assert np.all(smoothed < 1)
    # TODO Should test more
    
def test_to_pandas_df():
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    wavelet = PywaveletsWavelet(disable_caching=True)
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    df = res.to_frame()

    assert df['coherence'][0] > 0

def test_downsampling():
    wavelet = PywaveletsWavelet(disable_caching=True)
    signal = SynteticSignal(n_points=2000).add_sin(1)
    pair = PairSignals(signal.x, signal.y, signal.y)
    wtc = wavelet.wtc(pair)

    dt = signal.x[1] - signal.x[0]
    nyquist = (1 / dt) / 2
    assert wtc.sfreq == 1 / dt
    assert wtc.nyquist == nyquist
    wtc.downsample_in_time(100)

    assert len(wtc.times) == 100
    # These should not change
    assert wtc.sfreq == 1 / dt
    assert wtc.nyquist == nyquist

def test_wtc_time_slicing():
    tmax = 100
    n = 1000
    signal1 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    signal2 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    wavelet = PywaveletsWavelet(disable_caching=True)
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), bin_seconds=10)
    df = res.to_frame()

    assert df.shape[0] == 10

    # first and last bins should be excluded (nan) since they do not have enough unmasked values
    masked = df['coherence_masked']
    assert masked[0] > masked[1]
    assert np.isnan(df.at[0, 'coherence'])
    assert np.isnan(df.at[df.shape[0]-1, 'coherence'])
    
def test_wtc_period_slicing():
    tmax = 100
    n = 1000
    signal1 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    signal2 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    wavelet = PywaveletsWavelet(disable_caching=True)
    period_cuts = [3, 5, 10]
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), period_cuts=period_cuts)
    df = res.to_frame()

    assert df.shape[0] == 4

    # first and last bins should be excluded (nan) since they do not have enough unmasked values
    masked = df['coherence_masked']
    assert masked[0] < masked[1]

def test_wtc_period_slicing_edge_cases():
    tmax = 100
    n = 1000
    signal1 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    signal2 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    wavelet = PywaveletsWavelet(disable_caching=True)
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    assert wavelet.wtc(pair, period_cuts=[99999]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[0]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[0, 1, 1.5]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[3, 10]).to_frame().shape[0] == 3
    assert wavelet.wtc(pair, period_cuts=[3, 3.001, 3.002, 10]).to_frame().shape[0] == 3
    
def test_wtc_period_time_combined_slicing():
    tmax = 100
    n = 1000
    signal1 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    signal2 = SynteticSignal(tmax=tmax, n_points=n).add_noise()
    wavelet = PywaveletsWavelet(disable_caching=True)
    bin_seconds = 20
    period_cuts = [3, 5, 10]
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), bin_seconds=bin_seconds, period_cuts=period_cuts)
    df = res.to_frame()

    assert df.shape[0] == 5 * 4
    #print(df)
    
def test_wtc_wavelet_info():
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    wavelet = PywaveletsWavelet(disable_caching=True)
    res = wavelet.wtc(pair)
    df = res.to_frame()

    assert df.at[0, 'wavelet_library'] == 'pywavelets'
    assert df.at[0, 'wavelet_name'] == wavelet.wavelet_name
    
@pytest.mark.parametrize("wavelet", [
   PywaveletsWavelet(), 
   PywaveletsWavelet(wavelet_name='cmor10,1'), 
   PywaveletsWavelet(wavelet_name='cgau1'), 
   PywaveletsWavelet(wavelet_name='cgau2'), 
   PywaveletsWavelet(wavelet_name='cgau3'), 
])
def test_cone_of_influence(wavelet):
    n = 11
    dt = 1
    coi = wavelet.get_cone_of_influence(n, dt)
    assert coi[0] == coi[-1]
    assert coi[0] < coi[1]
    assert np.argmax(coi) == n // 2
    assert len(coi) == n

    
    