import os
from pathlib import Path
import sys

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from hypyp.signal import SynteticSignal
from hypyp.wavelet.matlab_wavelet import MatlabWavelet
import pywt
import pycwt
from scipy import fft
import mne

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots
from hypyp.wavelet.pywt_wavelet import Wavelet
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
                        ui.output_ui('ui_input_signal_slider'),
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
            ui.output_ui('ui_card_tracer'),
        ),
        selected='Wavelet Coherence',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.tags.strong('Signal parameters'),
            ui.input_select(
                "signal_choice",
                "",
                choices={
                    'chirp_sinusoid': 'One sinusoid, one chirp',
                    'chirp': 'Crossing chirps',
                    'sinusoid': 'Sinusoid different',
                    'sinusoid_almost_similar': 'Sinusoid almost similar',
                    'sinusoid_dephased': 'Sinusoid dephased',
                    'fnirs_syn_ce05_001_nirs_disk': 'fNIRS data from Passport Disk',
                    'fnirs_sub_110_session_1_pre': 'fNIRS sub 110 session 1',
                },
            ),
            ui.output_ui('ui_input_signal_options'),

            ui.tags.strong('Wavelet parameters'),
            ui.input_select(
                "wavelet_library",
                "",
                choices={
                    'pywt': 'Use pywavelets',
                    'pycwt': 'Use pycwt (Matlab based)',
                },
            ),
            ui.output_ui('ui_input_wavelet_type'),
            ui.output_ui('ui_input_wavelet_options'),

            ui.tags.strong('Coherence parameters'),
            ui.output_ui('ui_input_coherence_options'),

            ui.input_action_button("button_action_compute_wct", label="Compute WCT"),

            ui.tags.strong('Display parameters'),
            ui_option_row("Daughter wavelet id", ui.input_numeric("display_daughter_wavelet_id", "", value=0)),
            ui_option_row("WCT frequencies at time", ui.input_numeric("display_wct_frequencies_at_time", "", value=-1)),

            ui_option_row("Downsample", ui.input_checkbox("display_downsample", "", value=True), sizes=(8,4)),
            ui_option_row("Show COI", ui.input_checkbox("display_show_coif", "", value=True), sizes=(8,4)),
            ui_option_row("Show Nyquist", ui.input_checkbox("display_show_nyquist", "", value=True), sizes=(8,4)),
            ui_option_row("Show tracer plots", ui.input_checkbox("display_show_tracer", "", value=False), sizes=(8,4)),
            ui_option_row("Show Log value for tracers", ui.input_checkbox("display_show_log_tracer", "", value=False), sizes=(8,4)),
            ui.input_select(
                "diplay_show_tracer_complex",
                "",
                choices={
                    'abs': 'Magnitude',
                    'real': 'Real part',
                    'imag': 'Imaginary part',
                },
            ),
            open="always",
        ),
        title="Wavelet Explorer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    def is_real_data(input):
        return input.signal_choice().startswith('fnirs_')

    @reactive.calc()
    def signals():
        fs = input.signal_sampling_frequency()
        N = input.signal_n()
        T = 1.0 / fs
        x = np.linspace(0, N/fs, N)

        if input.signal_choice() == 'sinusoid':
            freq1 = input.signal_sinusoid_freq1()
            freq2 = input.signal_sinusoid_freq2()
            y1 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(freq1).y
            y2 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(freq2).y
        elif input.signal_choice() == 'sinusoid_almost_similar':
            freq1 = input.signal_sinusoid_almost_similar_freq1()
            freq2 = freq1 - 0.001
            y1 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(freq1).y
            y2 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(freq2).y
        elif input.signal_choice() == 'sinusoid_dephased':
            freq1 = input.signal_sinusoid_dephased_freq1()
            y1 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(freq1).y
            y1_fft = np.fft.fft(y1)
            y1_magnitude = np.abs(y1_fft)
            y1_phase = np.angle(y1_fft)
            y1_random_phase = np.random.uniform(-np.pi, np.pi, len(y1_phase))
            y1_randomized_fft = y1_magnitude * np.exp(1j * y1_random_phase)
            y2 = np.fft.ifft(y1_randomized_fft).real
            y2 = y2 * 1 / (np.abs(np.max(y2)-np.min(y2)) / 2)
        elif input.signal_choice() == 'chirp':
            signal1 = SynteticSignal(tmax=N/fs, n_points=N)
            signal1.add_chirp(input.signal_chirp_freq1(), input.signal_chirp_freq2())
            y1 = signal1.y
            y2 = np.flip(y1)
        elif input.signal_choice() == 'chirp_sinusoid':
            y1 = SynteticSignal(tmax=N/fs, n_points=N).add_sin(input.signal_chirp_sinusoid_freq1()).y
            y2 = SynteticSignal(tmax=N/fs, n_points=N).add_chirp(input.signal_chirp_sinusoid_freq2(), input.signal_chirp_sinusoid_freq3()).y
        elif input.signal_choice() == 'fnirs_syn_ce05_001_nirs_disk':
            base_path = "/media/patrice/My Passport/DataNIRS/"
            filename = "syn_ce05_001.nirs"
            file_path = os.path.join(base_path, filename)
            mat = scipy.io.loadmat(file_path)
            x = mat['t']
            y1 = mat['d'][:,0]
            y2 = mat['d'][:,19]
    
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


        if not is_real_data(input):
            noise_level = input.signal_noise_level()
            #noise_level = 0

            y1 +=  noise_level * np.random.normal(0, 1, len(x))
            y2 +=  noise_level * np.random.normal(0, 1, len(x))

        signal_from = 0
        signal_to = len(x)
        try:
            if input.signal_range() is not None:
                signal_from = N * input.signal_range()[0] // 100
                signal_to = N * input.signal_range()[1] // 100
        except:
            pass

        return (
            x[signal_from:signal_to],
            y1[signal_from:signal_to],
            y2[signal_from:signal_to],
            len(x) # return length of signal to allow scrolling
        )


    @render.ui
    def ui_input_signal_options():
        options = []
        if input.signal_choice() == 'sinusoid':
            options.append(ui_option_row("Freq 1", ui.input_numeric("signal_sinusoid_freq1", "", value=1))),
            options.append(ui_option_row("Freq 2", ui.input_numeric("signal_sinusoid_freq2", "", value=0.8))),
        elif input.signal_choice() == 'sinusoid_almost_similar':
            options.append(ui_option_row("Freq", ui.input_numeric("signal_sinusoid_almost_similar_freq1", "", value=0.2))),
        elif input.signal_choice() == 'sinusoid_dephased':
            options.append(ui_option_row("Freq", ui.input_numeric("signal_sinusoid_dephased_freq1", "", value=0.02))),
        elif input.signal_choice() == 'chirp':
            options.append(ui_option_row("Freq Chirp from", ui.input_numeric("signal_chirp_freq1", "", value=0.2))),
            options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_freq2", "", value=2))),
        elif input.signal_choice() == 'chirp_sinusoid':
            options.append(ui_option_row("Freq Sinusoid", ui.input_numeric("signal_chirp_sinusoid_freq1", "", value=1))),
            options.append(ui_option_row("Freq Chirp from", ui.input_numeric("signal_chirp_sinusoid_freq2", "", value=0.2))),
            options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_sinusoid_freq3", "", value=2))),
        
        if not is_real_data(input):
            options.append(ui_option_row("Sampling freq. (Hz)", ui.input_numeric("signal_sampling_frequency", "", value=5))),
            options.append(ui_option_row("Nb. points", ui.input_numeric("signal_n", "", value=2000))),
            options.append(ui_option_row("Noise level", ui.input_numeric("signal_noise_level", "", value=0.01))),
        
        return options

    @render.ui
    def ui_input_signal_slider():
        options = []
        try:
            options.append(ui.input_slider("signal_range", "", min=0, max=100, value=[0, 100], width='100%'))
        except:
            pass
            
        return options

    @reactive.calc()
    def wavelet_name():
        if input.wavelet_type() == 'cmor':
            return f'cmor{input.wavelet_bandwidth()}, {input.wavelet_center_frequency()}'
        else:
            return input.wavelet_type()

    @reactive.calc()
    def wavelet():
        if input.wavelet_library() == 'pywt':
            wavelet = Wavelet(
                wavelet_name=wavelet_name(),
                precision=input.wavelet_precision(),
                upper_bound=input.wavelet_upper_bound(),
                lower_bound=-input.wavelet_upper_bound(),
                wct_smoothing_smooth_factor=input.smoothing_smooth_factor(),
                wct_smoothing_boxcar_size=input.smoothing_boxcar_size(),
            )
            if input.wavelet_hack_compute_each_scale():
                wavelet.cwt_params['hack_compute_each_scale'] = True

        elif input.wavelet_library() == 'pycwt':
            wavelet = MatlabWavelet(
                precision=input.wavelet_precision(),
                upper_bound=input.wavelet_upper_bound(),
                lower_bound=-input.wavelet_upper_bound(),
            )
        else:
            raise RuntimeError(f'Unknown wavelet library: {input.wavelet_library()}')
        return wavelet

    @reactive.event(input.button_action_compute_wct)
    def compute_coherence():
        x, y1, y2, _ = signals()
        return wavelet().wct(y1, y2, x[1] - x[0])

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
        ax.set_yscale('log')
        ax.grid()
        return fig

    @render.plot()
    def plot_mother_wavelet():
        fig, ax = plt.subplots()
        wav = wavelet()
        ax.plot(wav.psi_x, np.real(wav.psi))
        ax.plot(wav.psi_x, np.imag(wav.psi))
        ax.plot(wav.psi_x, np.abs(wav.psi))
        ax.title.set_text(f"mother wavelet ({wav.wavelet_name})")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_daughter_wavelet():
        fig, ax = plt.subplots()

        wct_res = compute_coherence()
        id = input.display_daughter_wavelet_id()
        psi = wct_res.tracer['psi_scales'][id]
        ax.plot(np.real(psi))
        ax.plot(np.imag(psi))
        ax.plot(np.abs(psi))
        ax.title.set_text(f"daughter wavelet {id}/{len(wct_res.tracer['psi_scales'])} for {wct_res.frequencies[id]:.3f}Hz")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_wct():
        fig, ax = plt.subplots()

        wct_res = compute_coherence()
        hypyp.plots.plot_wavelet_coherence(
            np.abs(wct_res.wct),
            wct_res.times,
            wct_res.frequencies,
            wct_res.coif,
            ax=ax,
            colorbar=False,
            downsample=input.display_downsample(),
            show_coif=input.display_show_coif(),
            show_nyquist=input.display_show_nyquist(),
        )
        return fig

    def ZZ_with_options(ZZ):
        if input.display_show_log_tracer():
            ZZ = np.log(ZZ)
        if input.diplay_show_tracer_complex() == 'abs' and ZZ.dtype.kind == 'c':
            ZZ = np.abs(ZZ)
        elif input.diplay_show_tracer_complex() == 'real':
            ZZ = np.real(ZZ)
        elif input.diplay_show_tracer_complex() == 'imag':
            ZZ = np.imag(ZZ)
        return ZZ
        

    @render.plot()
    def plot_tracer_W1():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['W1']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('W1')
        return fig

    @render.plot()
    def plot_tracer_W2():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['W2']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('W2')
        return fig

    @render.plot()
    def plot_tracer_W12():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['W12']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('W12')
        return fig

    @render.plot()
    def plot_tracer_S1():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['S1']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('S1')
        return fig

    @render.plot()
    def plot_tracer_S2():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['S2']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('S2')
        return fig

    @render.plot()
    def plot_tracer_S12():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        tracer = wct_res.tracer
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.tracer['S12']))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('S12')
        return fig

    @render.plot()
    def plot_tracer_frequencies():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        ax.plot(wct_res.frequencies)
        ax.title.set_text('frequencies')
        return fig

    @render.plot()
    def plot_tracer_scales():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        ax.plot(wct_res.scales)
        ax.title.set_text('scales')
        return fig

    @render.plot()
    def plot_tracer_wct():
        fig, ax = plt.subplots()
        wct_res = compute_coherence()
        im = ax.pcolormesh(wct_res.times, wct_res.frequencies, ZZ_with_options(wct_res.wct))
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('wct')
        return fig

    @render.plot()
    def plot_wct_half_time():
        fig, ax = plt.subplots()

        x, y1, y2, _ = signals()
        wct_res  = compute_coherence()

        if input.display_wct_frequencies_at_time() == -1:
            col = len(wct_res.times)//2
        else:
            dt = x[1] - x[0]
            col = int(input.display_wct_frequencies_at_time() / dt)
            
        ax.plot(wct_res.frequencies, wct_res.wct[:,col])
        ax.title.set_text(f'Frequencies at t={wct_res.times[col]:.1f} (max found: {wct_res.frequencies[np.argmax(wct_res.wct[:,col])]:.2f}Hz)')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        return fig

    @render.ui
    def ui_input_wavelet_type():
        options = []
        if input.wavelet_library() == 'pywt':
            options.append(ui.input_select(
                    "wavelet_type",
                    "",
                    choices={
                        'cmor': 'Complex Morlet',
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
                ))
        return options


    @render.ui
    def ui_input_wavelet_options():
        options = []
        if input.wavelet_library() == 'pywt':
            if input.wavelet_type() == 'cmor':
                options.append(ui_option_row("Bandwidth", ui.input_numeric("wavelet_bandwidth", "", value=2)))
                options.append(ui_option_row("Center frequency", ui.input_numeric("wavelet_center_frequency", "", value=1)))

            options.append(ui_option_row("Upper bound", ui.input_numeric("wavelet_upper_bound", "", value=8)))
            options.append(ui_option_row("Precision", ui.input_numeric("wavelet_precision", "", value=10)))
            options.append(ui_option_row("Hack compute each scale", ui.input_checkbox("wavelet_hack_compute_each_scale", "", value=False), sizes=(8,4)))
        return options
    
    @render.ui
    def ui_input_coherence_options():
        options = []
        if input.wavelet_library() == 'pywt':
            options.append(ui_option_row("Smooth factor", ui.input_numeric("smoothing_smooth_factor", "", value=-0.1)))
            options.append(ui_option_row("Boxcar size", ui.input_numeric("smoothing_boxcar_size", "", value=1)))
        return options
    
    @render.ui
    def ui_card_tracer():
        if input.display_show_tracer():
            return ui.card(
                ui.tags.strong('Tracing of intermediary results'),
                ui.row(
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_W1'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_W2'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_W12'),
                    ),
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_S1'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_S2'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_S12'),
                    ),
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_frequencies'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_scales'),
                    ),
                    ui.column(
                        4,
                        ui.output_plot('plot_tracer_wct'),
                    ),
                ),
            )


app = App(app_ui, server)