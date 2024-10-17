import os
from pathlib import Path
import sys

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pywt
from scipy import fft

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots
from hypyp.fnirs_tools import (
    xwt_coherence_morl
)

default_plot_signal_height = 150

matplotlib.use('Agg')

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
                "Signal choice",
                choices={
                    'chirp': 'Crossing chirps',
                    'chirp_multiple': 'Multiple crossing chirps',
                    'sinusoid': 'Sinusoid different',
                    'sinusoid_almost_similar': 'Sinusoid almost similar',
                    'sinusoid_dephased': 'Sinusoid dephased',
                },
            ),
            ui.input_numeric("signal_sampling_frequency", "Sampling frequency", value=5),
            ui.input_numeric("signal_n", "Number of points", value=2000),
            ui.input_numeric("signal_noise_level", "Noise level", value=0.01),

            ui.tags.strong('Wavelet parameters'),
            ui.input_select(
                "wavelet_type",
                "Wavelet type",
                choices={'cmor': 'Complex Morlet', 'cgau1': 'Complex Gaussian 1', 'cgau2': 'Complex Gaussian 2'},
            ),
            ui.output_ui('ui_input_wavelet_options'),

            ui.tags.strong('Smoothing parameters'),
            ui.input_action_button("button_action_compute_wct", label="Compute WCT"),
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
            freq1 = 0.02
            freq2 = 0.04
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
        return xwt_coherence_morl(
            y1,
            y2,
            wavelet_name=wavelet_name(),
            dt=(x[1] - x[0]),
            normalize=True,
            smoothing_params=dict(smooth_factor=-0.1, boxcar_size=1),
        )
        


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
        wavelet = pywt.ContinuousWavelet(name)
        psi, x = wavelet.wavefun(10)
        ax.plot(x, np.real(psi))
        ax.plot(x, np.imag(psi))
        ax.plot(x, np.abs(psi))
        ax.title.set_text(f"mother wavelet ({name})")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_wct():
        fig, ax = plt.subplots()

        wct, times, freqs, coif = compute_coherence()
        hypyp.plots.plot_wavelet_coherence(np.abs(wct), times, freqs, coif, ax=ax, colorbar=False)
        return fig

    @render.plot()
    def plot_wct_half_time():
        fig, ax = plt.subplots()

        x, y1, y2, chirp_frequencies = signals()
        wct, times, freqs, coif = compute_coherence()
        col_mid = len(times)//2
        ax.plot(freqs, wct[:,col_mid])
        ax.title.set_text(f'Frequencies at t={times[col_mid]:.1f} (max expected: {chirp_frequencies[col_mid]:.2f}Hz, found: {freqs[np.argmax(wct[:,col_mid])]:.2f}Hz)')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        return fig

    @render.ui
    def ui_input_wavelet_options():
        if input.wavelet_type() == 'cmor':
            return [
                ui.input_numeric("wavelet_bandwidth", "Wavelet bandwidth", value=2),
                ui.input_numeric("wavelet_center_frequency", "Wavelet center frequency", value=1),
            ]
        return []


app = App(app_ui, server)