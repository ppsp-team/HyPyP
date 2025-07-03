import pytest

import numpy as np

from hypyp.signal import SyntheticSignal
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexGaussianWavelet, ComplexMorletWavelet

def test_instanciate_complex_gaussian_wavelet():
    wavelet = ComplexGaussianWavelet(degree=2)
    assert wavelet.wavelet_name == 'cgau2'
    assert wavelet.degree == 2
    assert len(wavelet.psi) == 2 ** 10
    assert isinstance(wavelet.cwt_params, dict)

def test_instanciate_complex_morlet_wavelet():
    wavelet = ComplexMorletWavelet(2, 1)
    assert wavelet.wavelet_name == 'cmor2,1'
    assert wavelet.bandwidth_frequency == 2
    assert wavelet.center_frequency == 1

def test_resolution():
    wavelet = ComplexMorletWavelet(evaluate=False)
    # need to be evaluated before calling .dt
    with pytest.raises(Exception):
        wavelet.psi_dx
    
    wavelet.evaluate_psi()
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1

def test_default_domain():
    wavelet = ComplexMorletWavelet()
    assert min(wavelet.psi_x) == wavelet.domain[0]
    assert max(wavelet.psi_x) == wavelet.domain[1]
    
def test_domain():
    lower_bound = -1
    upper_bound = 1
    wavelet = ComplexMorletWavelet(lower_bound=lower_bound, upper_bound=upper_bound)

    assert min(wavelet.psi_x) == lower_bound
    assert max(wavelet.psi_x) == upper_bound
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1
    
def test_psi():
    wavelet = ComplexMorletWavelet(bandwidth_frequency=2, center_frequency=1)
    assert len(wavelet.psi_x) == len(wavelet.psi)
    assert len(wavelet.psi_x) == 2 ** 10

    assert np.sum(np.real(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.imag(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.abs(wavelet.psi) * wavelet.psi_dx) == pytest.approx(1)

def test_number_of_scales():
    assert len(ComplexMorletWavelet(period_range=(2, 4)).get_periods()) == 12
    assert len(ComplexMorletWavelet(period_range=(4, 8)).get_periods()) == 12
    assert len(ComplexMorletWavelet(period_range=(2, 8)).get_periods()) == 24
    assert len(ComplexMorletWavelet(dj=1/100, period_range=(2, 4)).get_periods()) == 100

    with pytest.raises(Exception):
        # must be < 1
        ComplexMorletWavelet(dj=2)
    
@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_cwt(wavelet_class):
    wavelet = wavelet_class()
    signal = SyntheticSignal(duration=100).add_sin(0.05)
    res = wavelet.cwt(signal.y, signal.period)
    assert len(res.scales) > 0
    assert len(res.scales) == len(res.frequencies)
    # TODO test something on res.W

def test_pair_signals():
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    sub = pair.sub((0, 1))
    assert sub.x[0] == 0
    assert sub.x[-1] < 1.1
    assert sub.x[-1] > 0.9
    assert sub.section_idx == pair.section_idx
    
def test_pair_signals_epoch():
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    sub = pair.sub((0, 1), section_idx=pair.section_idx+1)
    assert sub.section_idx == pair.section_idx+1
    

@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_wtc(wavelet_class):
    wavelet = wavelet_class(disable_caching=True)
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert res.coherence_metric > 0
    assert res.coherence_metric < 1
    
def test_cache_key():
    assert 'key_foo_bar' in ComplexMorletWavelet()._get_cache_key('foo', 'bar')
    assert ComplexMorletWavelet(disable_caching=True)._get_cache_key('foo', 'bar') is None
    
def test_cache():
    wavelet = ComplexMorletWavelet(cache=dict())
    assert len(list(wavelet.cache.keys())) == 0
    wavelet._add_cache_item('foo', 'bar')
    assert len(list(wavelet.cache.keys())) == 1
    assert wavelet._get_cache_item('foo') == 'bar'
    wavelet.clear_cache()
    assert len(list(wavelet.cache.keys())) == 0
    assert wavelet._get_cache_item('foo') == None

    zeros = np.zeros((10,))
    x = np.arange(len(zeros))

    # all these should yield a unique key
    pairs = [
        PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', label_task='my_task'),
        PairSignals(x[2:], zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', label_task='my_task'), # other time range
        PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject1', label_s2='subject2', label_task='my_other_task'),
        PairSignals(x, zeros, zeros, label_ch1='ch3', label_ch2='ch4', label_s1='subject1', label_s2='subject2', label_task='my_task'),
        PairSignals(x, zeros, zeros, label_ch1='ch1', label_ch2='ch2', label_s1='subject5.1', label_s2='subject5.2', label_task='my_task'),
    ]
    # add all the keys to a list, then use a set to remove duplicates and make sure we still have the same count
    keys = []
    for pair in pairs:
        keys.append(wavelet._get_cache_key_pair(pair, 0, 'cwt'))
        keys.append(wavelet._get_cache_key_pair(pair, 1, 'cwt'))

    keys.append(wavelet._get_cache_key_pair(pairs[0], 0, 'another_suffix'))

    #print(keys)
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_wtc_coi_masked(wavelet_class):
    wavelet = wavelet_class(disable_caching=True)
    signal = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal.x, signal.y, signal.y))
    assert res.wtc_masked is not None
    assert res.wtc_masked.mask[0,0] == True
    assert res.wtc_masked.mask[0,len(signal.x)//2] == False

@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_period_frequency_range(wavelet_class):    
    frequency_range = np.array([5., 1.])
    period_range = 1 / frequency_range
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()

    wavelet1 = wavelet_class(period_range=tuple(period_range), disable_caching=True)
    res1 = wavelet1.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert np.all(res1.frequencies[[0,-1]] == pytest.approx(frequency_range))

    wavelet2 = wavelet_class(frequency_range=tuple(frequency_range), disable_caching=True)
    res2 = wavelet2.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    assert np.all(res2.frequencies[[0,-1]] == pytest.approx(frequency_range))
    
def test_smooth_in_scale_window():
    assert np.sum(BaseWavelet._get_smoothing_window(0.6, 10)) == 1
    assert len(BaseWavelet._get_smoothing_window(1, 1/2)) == 4
    assert len(BaseWavelet._get_smoothing_window(1.1, 1/2)) == 5

    # dirac
    assert len(BaseWavelet._get_smoothing_window(1/12, 1/12)) == 2

    win10 = BaseWavelet._get_smoothing_window(5, 1)
    assert len(win10) == 10
    assert np.mean(win10) == win10[0]

    win10_plus = BaseWavelet._get_smoothing_window(5.1, 1)
    assert len(win10_plus) == 11
    assert np.mean(win10_plus) > win10_plus[0]

def test_fft_kwargs():
    d = BaseWavelet._get_fft_kwargs(np.zeros((10,)))
    assert d['n'] == 16

    d = BaseWavelet._get_fft_kwargs(np.zeros((10,)), extra='foo')
    assert d['extra'] == 'foo'
    
@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_smoothing(wavelet_class):
    dt = 0.1
    dj = 0.1
    wavelet = wavelet_class(dj=dj)
    scales = wavelet.get_scales(dt)
    W = np.zeros((len(scales), 200))
    W[:, np.arange(0, W.shape[1], 2)] = 1
    smoothed = wavelet.smoothing(W, dt, scales)
    assert np.all(smoothed > 0)
    assert np.all(smoothed < 1)
    # TODO Should test more
    
@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_to_pandas_df(wavelet_class):
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    wavelet = wavelet_class(disable_caching=True)
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))
    df = res.to_frame()

    assert df['coherence'][0] > 0

@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_downsampling(wavelet_class):
    wavelet = wavelet_class(disable_caching=True)
    signal = SyntheticSignal(n_points=2000).add_sin(1)
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
    signal1 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    signal2 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    wavelet = ComplexMorletWavelet(disable_caching=True)
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
    signal1 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    signal2 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    wavelet = ComplexMorletWavelet(disable_caching=True)
    period_cuts = [3, 5, 10]
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), period_cuts=period_cuts)
    df = res.to_frame()

    assert df.shape[0] == 4

    # first and last bins should be excluded (nan) since they do not have enough unmasked values
    masked = df['coherence_masked']
    assert masked[0] < masked[1]

def test_wtc_time_series():
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    wavelet = ComplexMorletWavelet(disable_caching=True)
    period_cuts = [3, 5, 10]
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), period_cuts=period_cuts)
    time_series = res.get_as_time_series()
    assert time_series.shape[0] == 4
    assert time_series.shape[1] == len(signal1.x)
    

def test_wtc_period_slicing_edge_cases():
    tmax = 100
    n = 1000
    signal1 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    signal2 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    wavelet = ComplexMorletWavelet(disable_caching=True)
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    assert wavelet.wtc(pair, period_cuts=[99999]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[0]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[0, 1, 1.5]).to_frame().shape[0] == 1
    assert wavelet.wtc(pair, period_cuts=[3, 10]).to_frame().shape[0] == 3
    assert wavelet.wtc(pair, period_cuts=[3, 3.001, 3.002, 10]).to_frame().shape[0] == 3
    
def test_wtc_period_time_combined_slicing():
    tmax = 100
    n = 1000
    signal1 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    signal2 = SyntheticSignal(duration=tmax, n_points=n).add_noise()
    wavelet = ComplexMorletWavelet(disable_caching=True)
    bin_seconds = 20
    period_cuts = [3, 5, 10]
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y), bin_seconds=bin_seconds, period_cuts=period_cuts)
    df = res.to_frame()

    assert df.shape[0] == 5 * 4
    #print(df)
    
@pytest.mark.parametrize("wavelet_class", [
   ComplexMorletWavelet, 
   ComplexGaussianWavelet, 
])
def test_wtc_wavelet_info(wavelet_class):
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    pair = PairSignals(signal1.x, signal1.y, signal2.y)
    wavelet = wavelet_class(disable_caching=True)
    res = wavelet.wtc(pair)
    df = res.to_frame()

    assert df.at[0, 'wavelet_library'] == 'pywavelets'
    assert df.at[0, 'wavelet_name'] == wavelet.wavelet_name_with_args
    
@pytest.mark.parametrize("wavelet", [
   ComplexMorletWavelet(), 
   ComplexMorletWavelet(bandwidth_frequency=10, center_frequency=1), 
   ComplexGaussianWavelet(degree=1), 
   ComplexGaussianWavelet(degree=2), 
   ComplexGaussianWavelet(degree=3), 
])
def test_cone_of_influence(wavelet):
    n = 11
    dt = 1
    coi = wavelet._get_cone_of_influence(n, dt)
    assert coi[0] == coi[-1]
    assert coi[0] < coi[1]
    assert np.argmax(coi) == n // 2
    assert len(coi) == n

    
    