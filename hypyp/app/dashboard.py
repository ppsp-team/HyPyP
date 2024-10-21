import os
from pathlib import Path
import sys

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pywt
import pycwt
from scipy import fft
import mne

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots
from hypyp.fnirs_tools import (
    xwt_coherence_morl
)

default_plot_signal_height = 150

matplotlib.use('Agg')

def ui_option_row(label, ui_element, sizes=(6, 6)):
    return ui.row(
        ui.column(sizes[0], label),
        ui.column(sizes[1], ui_element),
    ),

# UI
app_ui = ui.page_fluid(
    ui.page_navbar(
        ui.nav_spacer(),
        ui.nav_panel(
            "Wavelet Coherence",
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.tags.strong('Signal'),
                        ui.output_plot('plot_signals', height=default_plot_signal_height),
                        ui.output_plot('plot_signals_spectrum', height=default_plot_signal_height),
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.tags.strong('Wavelet'),
                        ui.output_plot('plot_mother_wavelet'),
                        ui.output_plot('plot_daughter_wavelet'),
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.tags.strong('Wavelet Coherence'),
                        ui.output_plot('plot_wct'),
                        ui.output_plot('plot_wct_half_time'),
                    ),
                ),
            ),
        ),
        selected='Wavelet Coherence',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.tags.strong('Signal parameters'),
            ui.input_select(
                "signal_choice",
                "",
                choices={
                    'chirp': 'Crossing chirps',
                    'chirp_multiple': 'Multiple crossing chirps',
                    'sinusoid': 'Sinusoid different',
                    'sinusoid_almost_similar': 'Sinusoid almost similar',
                    'sinusoid_dephased': 'Sinusoid dephased',
                    'fnirs_sub_110_session_1_pre': 'fNIRS sub 110 session 1',
                },
            ),
            ui_option_row("Sampling freq. (Hz)", ui.input_numeric("signal_sampling_frequency", "", value=5)),
            ui_option_row("Nb. points", ui.input_numeric("signal_n", "", value=2000)),
            ui_option_row("Noise level", ui.input_numeric("signal_noise_level", "", value=0.01)),

            ui.tags.strong('Wavelet parameters'),
            ui.input_select(
                "wavelet_type",
                "",
                choices={
                    'cmor': 'Complex Morlet',
                    'cmor_pycwt': 'Complex Morlet (pywct/Matlab)',
                    'cgau1': 'Complex Gaussian 1',
                    'cgau2': 'Complex Gaussian 2',
                    'cgau3': 'Complex Gaussian 3',
                    'cgau4': 'Complex Gaussian 4',
                    'cgau5': 'Complex Gaussian 5',
                    'cgau6': 'Complex Gaussian 6',
                    'cgau7': 'Complex Gaussian 7',
                    'cgau8': 'Complex Gaussian 8',
                    'fbsp': 'Fbsp',
                },
            ),
            ui.output_ui('ui_input_wavelet_options'),

            ui.tags.strong('Smoothing parameters'),
            ui.output_ui('ui_input_smoothing_options'),

            ui.input_action_button("button_action_compute_wct", label="Compute WCT"),

            ui.tags.strong('Display parameters'),
            ui_option_row("Daughter wavelet id", ui.input_numeric("display_daughter_wavelet_id", "", value=0)),
            ui_option_row("WCT frequencies at time", ui.input_numeric("display_wct_frequencies_at_time", "", value=-1)),

            ui_option_row("Downsample", ui.input_checkbox("display_downsample", "", value=True), sizes=(8,4)),
            ui_option_row("Show COI", ui.input_checkbox("display_show_coif", "", value=True), sizes=(8,4)),
            ui_option_row("Show Nyquist", ui.input_checkbox("display_show_nyquist", "", value=True), sizes=(8,4)),
            open="always",
        ),
        title="Wavelet Explorer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc()
    def signals():
        fs = input.signal_sampling_frequency()
        N = input.signal_n()
        T = 1.0 / fs
        x = np.linspace(0, N/fs, N)
        def make_low_freq_chirp(t, t0, base_freq):
            frequencies = (base_freq * np.log(t + t0)) ** 2
            chirp = np.sin(2 * np.pi * frequencies * t)
            return chirp, frequencies

        chirp_frequencies = np.zeros_like(x)
        if input.signal_choice() == 'sinusoid':
            freq1 = 1
            freq2 = 0.8
            y1 = np.sin(x * 2 * np.pi * freq1)
            y2 = np.sin(x * 2 * np.pi * freq2)
        elif input.signal_choice() == 'sinusoid_almost_similar':
            freq1 = 0.02
            freq2 = freq1 - 0.0001
            y1 = np.sin(x * 2 * np.pi * freq1)
            y2 = np.sin(x * 2 * np.pi * freq2)
        elif input.signal_choice() == 'sinusoid_dephased':
            freq1 = 0.02
            y1 = np.sin(x * 2 * np.pi * freq1)
            y1_fft = np.fft.fft(y1)
            y1_magnitude = np.abs(y1_fft)
            y1_phase = np.angle(y1_fft)
            y1_random_phase = np.random.uniform(-np.pi, np.pi, len(y1_phase))
            y1_randomized_fft = y1_magnitude * np.exp(1j * y1_random_phase)
            y2 = np.fft.ifft(y1_randomized_fft).real
            y2 = y2 * 1 / (np.abs(np.max(y2)-np.min(y2)) / 2)
        elif input.signal_choice() == 'chirp':
            y1, chirp_frequencies = make_low_freq_chirp(x, 10, 0.04)
            y2 = np.flip(y1)
        elif input.signal_choice() == 'chirp_multiple':
            sum_y1 = np.zeros_like(x)
            sum_y2 = np.zeros_like(x)
            y1, chirp_frequencies = make_low_freq_chirp(x, 10, 0.04)
            y2 = np.flip(y1)
            for i in range(100):
                y1_fft = np.fft.fft(y1)
                y1_magnitude = np.abs(y1_fft)
                y1_phase = np.angle(y1_fft)
                y1_random_phase = np.random.uniform(-np.pi, np.pi, len(y1_phase))
                y1_randomized_fft = y1_magnitude * np.exp(1j * y1_random_phase)
                sum_y1 = sum_y1 + np.fft.ifft(y1_randomized_fft).real

                y2_fft = np.fft.fft(y2)
                y2_magnitude = np.abs(y2_fft)
                y2_phase = np.angle(y2_fft)
                y2_random_phase = np.random.uniform(-np.pi, np.pi, len(y2_phase))
                y2_randomized_fft = y2_magnitude * np.exp(1j * y2_random_phase)
                sum_y2 = sum_y2 + np.fft.ifft(y2_randomized_fft).real

            y1 = sum_y1
            y2 = sum_y2
        elif input.signal_choice() == 'fnirs_sub_110_session_1_pre':

            prep_path = './data/'
            fname1 = 'sub-110_session-1_pre_raw.fif'
            fname2 = fname1
            p1 = prep_path + fname1
            p2 = prep_path + fname2

            chs1 = ["S4_D4 hbo"] 
            chs2 = ["S7_D6 hbo"]

            #set events
            tmin = 0 
            tmax = 300
            baseline = (0, 0)

            # read in data 1
            p1 = mne.io.read_raw_fif(p1, verbose=True, preload=True)
            p2 = mne.io.read_raw_fif(p2, verbose=True, preload=True)

            # select channels from csv of best channels
            ch_list1 = chs1
            ch_list2 = chs2
            ch_picks1 = mne.pick_channels(p1.ch_names, include = ch_list1)
            ch_picks2 = mne.pick_channels(p2.ch_names, include = ch_list2)
            p1_best_ch = p1.copy().pick(ch_picks1)
            p2_best_ch = p2.copy().pick(ch_picks2)

            # get events
            events1, event_dict1 = mne.events_from_annotations(p1_best_ch)
            events2, event_dict2 = mne.events_from_annotations(p2_best_ch)
            epo1 = mne.Epochs(p1_best_ch, events1,
                            event_id = event_dict1, tmin = tmin, tmax = tmax,
                            baseline = baseline, reject_by_annotation=False)
            epo2 = mne.Epochs(p2_best_ch, events2,
                            event_id = event_dict2, tmin = tmin, tmax = tmax,
                            baseline = baseline, reject_by_annotation=False)
            y1 = epo1.get_data()[0,0,:]
            y2 = epo2.get_data()[0,0,:]
            
            # Force N for this signal
            N = len(y1)
            x = np.linspace(0, N/fs, N)


        noise_level = input.signal_noise_level()
        #noise_level = 0

        y1 +=  noise_level * np.random.normal(0, 1, len(x))
        y2 +=  noise_level * np.random.normal(0, 1, len(x))

        return (x, y1, y2, chirp_frequencies)

    @reactive.calc()
    def wavelet_name():
        if input.wavelet_type() == 'cmor':
            return f'cmor{input.wavelet_bandwidth()}, {input.wavelet_center_frequency()}'
        else:
            return input.wavelet_type()

    @reactive.event(input.button_action_compute_wct)
    def compute_coherence():
        x, y1, y2, _ = signals()
        name = wavelet_name()
        tracer = dict(name=name)
        if name == 'cmor_pycwt':
            wct, _, coif_periods, freqs, _ = pycwt.wct(y1, y2, dt=x[1]-x[0], sig=False, tracer=tracer)
            coif = 1 / coif_periods
            return wct, x, freqs, coif, tracer
        else:
            cwt_params = dict()
            if input.wavelet_hack_compute_each_scale():
                cwt_params['hack_compute_each_scale'] = True

            wct, x, freqs, coif = xwt_coherence_morl(
                y1,
                y2,
                wavelet_name=wavelet_name(),
                dt=(x[1] - x[0]),
                normalize=True,
                smoothing_params=dict(
                    smooth_factor=input.smoothing_smooth_factor(),
                    boxcar_size=input.smoothing_boxcar_size(),
                ),
                cwt_params=cwt_params,
                tracer=tracer,
            )
            return wct, x, freqs, coif, tracer


    @render.plot(height=default_plot_signal_height)
    def plot_signals():
        fig, ax = plt.subplots()
        x, y1, y2, _ = signals()
        ax.plot(x, y1)
        ax.plot(x, y2)
        return fig

    @render.plot(height=default_plot_signal_height)
    def plot_signals_spectrum():
        fig, ax = plt.subplots()
        x, y1, y2, _ = signals()
        N = len(x)
        T = x[1] - x[0]

        yf = fft.fft(y1)
        xf = fft.fftfreq(N, T)[:N//2]
        yf2 = fft.fft(y2)

        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.plot(xf, 2.0/N * np.abs(yf2[0:N//2]))
        ax.set_xscale('log')
        ax.grid()
        return fig

    @render.plot()
    def plot_mother_wavelet():
        fig, ax = plt.subplots()
        name = wavelet_name()

        if name == 'cmor_pycwt':
            x = np.linspace(-8, 8, 1000)
            psi = pycwt.wavelet.Morlet().psi(x)
        else:
            wavelet = pywt.ContinuousWavelet(name)
            psi, x = wavelet.wavefun(10)


        ax.plot(x, np.real(psi))
        ax.plot(x, np.imag(psi))
        ax.plot(x, np.abs(psi))
        ax.title.set_text(f"mother wavelet ({name})")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_daughter_wavelet():
        fig, ax = plt.subplots()

        _, _, freqs, _, tracer = compute_coherence()
        id = input.display_daughter_wavelet_id()
        psi = tracer['psi_scales'][id]
        ax.plot(np.real(psi))
        ax.plot(np.imag(psi))
        ax.plot(np.abs(psi))
        ax.title.set_text(f"daughter wavelet {id}/{len(tracer['psi_scales'])} for {freqs[id]:.3f}Hz")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_wct():
        fig, ax = plt.subplots()

        wct, times, freqs, coif, tracer = compute_coherence()
        hypyp.plots.plot_wavelet_coherence(
            np.abs(wct),
            times,
            freqs,
            coif,
            ax=ax,
            colorbar=False,
            downsample=input.display_downsample(),
            show_coif=input.display_show_coif(),
            show_nyquist=input.display_show_nyquist(),
        )
        return fig

    @render.plot()
    def plot_wct_half_time():
        fig, ax = plt.subplots()

        x, y1, y2, chirp_frequencies = signals()
        wct, times, freqs, coif, tracer  = compute_coherence()

        if input.display_wct_frequencies_at_time() == -1:
            col = len(times)//2
        else:
            dt = x[1] - x[0]
            col = int(input.display_wct_frequencies_at_time() / dt)
            
        ax.plot(freqs, wct[:,col])
        ax.title.set_text(f'Frequencies at t={times[col]:.1f} (max expected: {chirp_frequencies[col]:.2f}Hz, found: {freqs[np.argmax(wct[:,col])]:.2f}Hz)')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        return fig

    @render.ui
    def ui_input_wavelet_options():
        options = []
        if input.wavelet_type() == 'cmor':
            options.append(ui_option_row("Bandwidth", ui.input_numeric("wavelet_bandwidth", "", value=2)))
            options.append(ui_option_row("Center frequency", ui.input_numeric("wavelet_center_frequency", "", value=1)))
        if input.wavelet_type() != 'cmor_pycwt':
            options.append(ui_option_row("Hack compute each scale", ui.input_checkbox("wavelet_hack_compute_each_scale", "", value=False), sizes=(8,4)))
        return options
    
    @render.ui
    def ui_input_smoothing_options():
        if input.wavelet_type() != 'cmor_pycwt':
            return [
                ui_option_row("Smooth factor", ui.input_numeric("smoothing_smooth_factor", "", value=-0.1)),
                ui_option_row("Boxcar size", ui.input_numeric("smoothing_boxcar_size", "", value=1)),
            ]


app = App(app_ui, server)