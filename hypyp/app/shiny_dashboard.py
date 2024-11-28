import os
from pathlib import Path
import sys
import tempfile

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import fft
import mne

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.fnirs.data_browser import DataBrowser
from hypyp.fnirs.subject import Subject
from hypyp.fnirs.preprocessors.mne_preprocessor import MnePreprocessor
from hypyp.fnirs.preprocessors.upstream_preprocessor import UpstreamPreprocessor
from hypyp.signal import SynteticSignal
from hypyp.wavelet.matlab_wavelet import MatlabWavelet
from hypyp.wavelet.pycwt_wavelet import PycwtWavelet
from hypyp.wavelet.scipy_wavelet import ScipyWavelet, DEFAULT_SCIPY_CENTER_FREQUENCY

# TODO: Cedalion is optional, this import should be in a try-catch
from hypyp.fnirs.preprocessors.cedalion_preprocessor import CedalionPreprocessor

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

DEFAULT_PLOT_SIGNAL_HEIGHT = 150 # px
DEFAULT_PLOT_MNE_HEIGHT = 1200 # px
DEFAULT_SIGNAL_DURATION_SELECTION = 60 # seconds

HARDCODED_PRELOADED_EXTERNAL_PATH = "/media/patrice/My Passport/"
STR_SAME_AS_SUBJECT_1 = 'Same as Subject 1'

# This is to avoid having external windows launched
matplotlib.use('Agg')

def ui_option_row(label, ui_element, sizes=(6, 6), center=False):
    label_style = ""
    if center:
        label_style += "text-align: right;"
    return ui.row(
        ui.column(sizes[0], label, style=label_style),
        ui.column(sizes[1], ui_element),
    )

# UI
app_ui = ui.page_fluid(
    ui.page_navbar(
        ui.nav_spacer(),
        ui.nav_panel(
            "Data Browser",
            ui.tags.h4(ui.output_text('text_s1_file_path')),
            ui.output_ui('ui_input_s1_mne_duration_slider'),
            ui.output_ui('ui_preprocess_steps'),
        ),
        ui.nav_panel(
            "Choice of Signals",
            ui.row(
                ui.column(12,
                    ui.card(
                        ui.tags.strong('Choice of signals'),
                        ui.output_ui('ui_input_signal_slider'),
                        ui.output_plot('plot_signals', height=DEFAULT_PLOT_SIGNAL_HEIGHT),
                        ui.output_plot('plot_signals_spectrum', height=DEFAULT_PLOT_SIGNAL_HEIGHT),
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.tags.strong('Signal 1'),
                        ui.output_table('text_info_s1_file_path'),
                        ui.output_table('table_info_s1'),
                    )
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.tags.strong('Signal 2'),
                        ui.output_table('text_info_s2_file_path'),
                        ui.output_table('table_info_s2'),
                    )
                ),
            ),
        ),
        ui.nav_panel(
            "Wavelet Coherence",
            ui.row(
                ui.column(6,
                    ui.card(
                        ui.tags.strong('Wavelet'),
                        ui.output_plot('plot_mother_wavelet'),
                        ui.output_ui('ui_plot_daughter_wavelet'),
                        ui.row(
                            ui.column(6, ui.output_plot('plot_scales')),
                            ui.column(6, ui.output_plot('plot_frequencies')),
                        ),
                    ),
                ),
                ui.column(6,
                    ui.card(
                        ui.tags.strong('Wavelet Coherence'),
                        ui.output_plot('plot_wtc'),
                        ui.output_ui('ui_plot_wtc_at_time'),
                    ),
                ),
            ),
            ui.output_ui('ui_card_tracer'),
        ),
        ui.nav_spacer(),
        selected='Data Browser',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.tags.strong('Signal parameters'),
            ui.input_select(
                "signal_type",
                "",
                choices={
                    'data_files': 'Local data files',
                    'testing': 'Test signals',
                },
            ),
            ui.output_ui('ui_input_signal_choice'),
            ui.output_ui('ui_input_signal_choice_property'),
            ui.output_ui('ui_input_signal_options'),

            ui.tags.strong('Wavelet parameters'),
            ui.input_select(
                "wavelet_library",
                "",
                choices={
                    'pywavelets': 'Use pywavelets',
                    'scipy': 'Use scipy.signal (deprecated)',
                    'pycwt': 'Use pycwt (based on matlab code)',
                    'matlab': 'Use Matlab engine',
                },
            ),
            ui.output_ui('ui_input_wavelet_type'),
            ui.output_ui('ui_input_wavelet_options'),

            ui.tags.strong('Coherence parameters'),
            ui.output_ui('ui_input_coherence_options'),

            ui.input_action_button("button_action_compute_wtc", label="Compute Wavelet Transform Coherence"),

            ui.tags.strong('Display parameters'),
            ui_option_row("Signal offset y", ui.input_checkbox("display_signal_offset_y", "", value=True), sizes=(8,4)),
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
            width=400,
        ),
        title="HyPyP fNIRS explorer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc()
    def get_signals() -> PairSignals:
        info_table1 = []
        info_table2 = []
        pair = None
        if input.signal_type() == 'testing':
            fs = input.signal_sampling_frequency()
            N = input.signal_n()
            x = np.linspace(0, N/fs, N)
            tmax = N / fs
            if input.signal_testing_choice() == 'sinusoid':
                y1 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_sinusoid_freq1()).y
                y2 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_sinusoid_freq2()).y
            elif input.signal_testing_choice() == 'sinusoid_almost_similar':
                y1 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_sinusoid_almost_similar_freq1()).y
                y2 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_sinusoid_almost_similar_freq1() - 0.001).y
            elif input.signal_testing_choice() == 'sinusoid_dephased':
                y1 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_sinusoid_dephased_freq1()).y
                y1_fft = np.fft.fft(y1)
                y1_magnitude = np.abs(y1_fft)
                y1_phase = np.angle(y1_fft)
                y1_random_phase = np.random.uniform(-np.pi, np.pi, len(y1_phase))
                y1_randomized_fft = y1_magnitude * np.exp(1j * y1_random_phase)
                y2 = np.fft.ifft(y1_randomized_fft).real
                y2 = y2 * 1 / (np.abs(np.max(y2)-np.min(y2)) / 2)
            elif input.signal_testing_choice() == 'chirp':
                signal1 = SynteticSignal(tmax=tmax, n_points=N)
                signal1.add_chirp(input.signal_chirp_freq1(), input.signal_chirp_freq2())
                y1 = signal1.y
                y2 = np.flip(y1)
            elif input.signal_testing_choice() == 'chirp_sinusoid':
                y1 = SynteticSignal(tmax=tmax, n_points=N).add_sin(input.signal_chirp_sinusoid_freq1()).y
                y2 = SynteticSignal(tmax=tmax, n_points=N).add_chirp(input.signal_chirp_sinusoid_freq2(), input.signal_chirp_sinusoid_freq3()).y

            noise_level = input.signal_noise_level()
            y1 +=  noise_level * np.random.normal(0, 1, len(x))
            y2 +=  noise_level * np.random.normal(0, 1, len(x))

        elif input.signal_type() == 'data_files':
            # TODO this only works with MNE
            s1_raw = get_subject1_step().obj.copy()
            s2_raw = get_subject2_step().obj.copy()

            s1_selected_ch = s1_raw.pick(mne.pick_channels(s1_raw.ch_names, include = [input.signal_data_files_s1_channel()]))
            s2_selected_ch = s2_raw.pick(mne.pick_channels(s2_raw.ch_names, include = [input.signal_data_files_s2_channel()]))

            y1 = np.ma.masked_invalid(s1_selected_ch.get_data())[0,:]
            y2 = np.ma.masked_invalid(s2_selected_ch.get_data())[0,:]

            # Crop signals
            stop = min(len(y1), len(y2))
            y1 = y1[:stop] 
            y2 = y2[:stop] 
            
            # Force N for this signal
            N = len(y1)
            fs = s1_selected_ch.info['sfreq']
            x = np.linspace(0, N/fs, N)
            info_table1 = [(k, s1_raw.info[k]) for k in s1_raw.info.keys()]
            info_table2 = [(k, s2_raw.info[k]) for k in s2_raw.info.keys()]

        if pair is None:
            pair = PairSignals(x, y1, y2, info_table1, info_table2)

        try:
            if input.signal_range() is not None:
                return pair.sub(input.signal_range())
        except:
            pass
        
        return pair

    def get_signal_duration():
        if input.signal_type() == 'testing':
            fs = input.signal_sampling_frequency()
            N = input.signal_n()
            return N/fs
        if input.signal_type() == 'data_files':
            return min([get_subject1_step().duration, get_subject2_step().duration])
            

    @reactive.calc()
    def get_wavelet():
        if input.wavelet_library() == 'pywavelets':
            wavelet_name = input.wavelet_name()
            if input.wavelet_name() == 'cmor':
                wavelet_name = f'cmor{input.wavelet_bandwidth()}, {input.wavelet_center_frequency()}'

            wavelet = PywaveletsWavelet(
                wavelet_name=wavelet_name,
                precision=input.wavelet_precision(),
                upper_bound=input.wavelet_upper_bound(),
                lower_bound=-input.wavelet_upper_bound(),
                wtc_smoothing_boxcar_size=input.smoothing_boxcar_size(),
                cache=None,
            )
            # TODO: is this still implemented?
            if input.wavelet_hack_compute_each_scale():
                wavelet.cwt_params['hack_compute_each_scale'] = True

        elif input.wavelet_library() == 'pycwt':
            wavelet = PycwtWavelet(
                precision=input.wavelet_precision(),
                upper_bound=input.wavelet_upper_bound(),
                lower_bound=-input.wavelet_upper_bound(),
                compute_significance=input.wavelet_pycwt_significance(),
            )

        elif input.wavelet_library() == 'scipy':
            wavelet = ScipyWavelet(
                center_frequency=input.wavelet_scipy_center_frequency(),
            )

        elif input.wavelet_library() == 'matlab':
            wavelet = MatlabWavelet()

        else:
            raise RuntimeError(f'Unknown wavelet library: {input.wavelet_library()}')

        return wavelet

    @reactive.event(input.button_action_compute_wtc)
    def compute_coherence():
        pair = get_signals()
        return get_wavelet().wtc(pair)

    def get_signal_data_files_s1_path():
        return input.signal_data_files_s1_path()
        
    def get_signal_data_files_s2_path():
        value = input.signal_data_files_s2_path()
        if value == STR_SAME_AS_SUBJECT_1:
            return input.signal_data_files_s1_path()
        return value

    @render.text
    def text_s1_file_path():
        return f"File: {get_signal_data_files_s1_path()}"

    @render.text
    def text_s2_file_path():
        value = input.signal_data_files_s2_path()
        return f"File: {get_signal_data_files_s2_path()}"

    @render.text
    def text_info_s1_file_path():
        return f"File: {get_signal_data_files_s1_path()}"

    @render.text
    def text_info_s2_file_path():
        return f"File: {get_signal_data_files_s2_path()}"

    def get_data_browser():
        return DataBrowser().add_source(HARDCODED_PRELOADED_EXTERNAL_PATH)

    def get_preprocessor():
        value = input.subject_preprocessor()
        if value == 'upstream':
            return UpstreamPreprocessor()
        if value == 'mne':
            return MnePreprocessor()
        if value == 'cedalion':
            return CedalionPreprocessor()
        raise RuntimeError(f'Unknown preprocessor "{value}"')
        

    @reactive.calc()
    def get_subject1():
        return Subject().load_file(get_signal_data_files_s1_path(), get_preprocessor())

    @reactive.calc()
    def get_subject2():
        return Subject().load_file(get_signal_data_files_s2_path(), get_preprocessor())

    @reactive.calc()
    def get_subject1_step():
        return get_subject1().get_preprocess_step(input.signal_data_files_analysis_property())

    @reactive.calc()
    def get_subject2_step():
        return get_subject2().get_preprocess_step(input.signal_data_files_analysis_property())

    def get_mne_raw_plot_kwargs(step, duration_range):
        try:
            range_start, range_end = duration_range
            start = range_start
            duration = (range_end - range_start)
            return dict(
                n_channels=len(step.ch_names),
                start=start,
                duration=duration,
                scalings='auto',
                show_scrollbars=False,
                block=True,
                show=False,
            )
        except:
            # TODO this is here because of CedalionPreprocessingStep. This code flow is ugly
            return dict()

    def mne_figure_as_image(fig):
        temp_img_path = tempfile.NamedTemporaryFile(suffix=".png").name
        fig.savefig(temp_img_path)
        return {"src": temp_img_path, "alt": "MNE Plot"}

    @render.ui
    def ui_preprocess_steps():
        # Need to wrap the plot function to have dynamic display in shiny.
        # Because of order of execution, this cannot be directly in the loop
        def bind_plot_mne_figure(step: int):
            def plot_mne_figure():
                return mne_figure_as_image(step.plot(**get_mne_raw_plot_kwargs(step, input.input_s1_mne_duration_slider())))
            # need to rename the function because every "output plot" must have a unique name
            plot_mne_figure.__name__ = f'{plot_mne_figure.__name__}_{step.key}'
            renderer = render.image(plot_mne_figure)
            # This is needed to avoid having the scrollbar on the right
            renderer._auto_output_ui_kwargs = dict(height=f'{DEFAULT_PLOT_MNE_HEIGHT}px')
            return renderer

        nav_panels = []
        subject = get_subject1()
        for step in subject.preprocess_steps:
            nav_panels.append(ui.nav_panel(step.desc, bind_plot_mne_figure(step)))

        return ui.navset_card_tab(*nav_panels, selected=subject.preprocess_steps[-1].desc)

    @render.plot
    def plot_s1_mne_sci():
        subject = get_subject1()
        fig, ax = plt.subplots()
        ax.hist(subject.quality_sci)
        ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])
        ax.title.set_text('Scalp Coupling Index')
        return fig

    @render.ui
    def ui_input_signal_choice():
        choices = []
        if input.signal_type() == 'testing':
            choices.append(ui.input_select(
                "signal_testing_choice",
                "",
                choices={
                    'chirp_sinusoid': 'One sinusoid, one chirp',
                    'chirp': 'Crossing chirps',
                    'sinusoid': 'Sinusoid different',
                    'sinusoid_almost_similar': 'Sinusoid almost similar',
                    'sinusoid_dephased': 'Sinusoid dephased',
                },
            ))
        
        elif input.signal_type() == 'data_files':
            browser = get_data_browser()
            browser.download_demo_dataset()

            choices.append(ui_option_row("Subject 1 file", ui.input_select(
                "signal_data_files_s1_path",
                "",
                choices=browser.list_all_files(),
                # for comparison with jupyter notebook
                #selected="/home/patrice/work/ppsp/HyPyP-synchro/data/fNIRS/downloads/fathers/FCS28/parent/NIRS-2019-11-10_003.hdr",
            )))
            choices.append(ui_option_row("Subject 2 file", ui.input_select(
                "signal_data_files_s2_path",
                "",
                choices=[STR_SAME_AS_SUBJECT_1] + browser.list_all_files(),
            )))
            choices.append(ui_option_row("Preprocessor", ui.input_select(
                "subject_preprocessor",
                "",
                choices={
                    'upstream': 'MNE (no preprocessing, load as-is)',
                    'mne': 'MNE (basic fNIRS preprocessing)',
                    'cedalion': 'Cedalion (proof of concept preprocessing)',
                }
            )))
        
        return choices

    @render.ui
    def ui_input_signal_choice_property():
        options = []
        if input.signal_type() == 'data_files':
            options.append(ui_option_row(
                "Which step to use for analysis",
                ui.input_select(
                    "signal_data_files_analysis_property",
                    "",
                    choices=get_subject1().preprocess_step_choices,
                )
            ))
        return options

    @render.ui
    def ui_input_signal_options():
        options = []
        if input.signal_type() == 'testing':
            if input.signal_testing_choice() == 'sinusoid':
                options.append(ui_option_row("Freq 1", ui.input_numeric("signal_sinusoid_freq1", "", value=1))),
                options.append(ui_option_row("Freq 2", ui.input_numeric("signal_sinusoid_freq2", "", value=0.8))),
            elif input.signal_testing_choice() == 'sinusoid_almost_similar':
                options.append(ui_option_row("Freq", ui.input_numeric("signal_sinusoid_almost_similar_freq1", "", value=0.2))),
            elif input.signal_testing_choice() == 'sinusoid_dephased':
                options.append(ui_option_row("Freq", ui.input_numeric("signal_sinusoid_dephased_freq1", "", value=0.02))),
            elif input.signal_testing_choice() == 'chirp':
                options.append(ui_option_row("Freq Chirp from", ui.input_numeric("signal_chirp_freq1", "", value=0.2))),
                options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_freq2", "", value=2))),
            elif input.signal_testing_choice() == 'chirp_sinusoid':
                options.append(ui_option_row("Freq Sinusoid", ui.input_numeric("signal_chirp_sinusoid_freq1", "", value=1))),
                options.append(ui_option_row("Freq Chirp from", ui.input_numeric("signal_chirp_sinusoid_freq2", "", value=0.2))),
                options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_sinusoid_freq3", "", value=2))),
        
            options.append(ui_option_row("Sampling freq. (Hz)", ui.input_numeric("signal_sampling_frequency", "", value=5))),
            options.append(ui_option_row("Nb. points", ui.input_numeric("signal_n", "", value=2000))),
            options.append(ui_option_row("Noise level", ui.input_numeric("signal_noise_level", "", value=0.01))),
        
        if input.signal_type() == 'data_files':
            # TODO this try-except is here to have a PoC of Cedalion integration
            try:
                ch_names1 = get_subject1().pre.ch_names
                ch_names2 = get_subject2().pre.ch_names
            except:
                ch_names1 = []
                ch_names2 = []

            options.append(ui_option_row(
                "Subject 1 channel",
                ui.input_select(
                    "signal_data_files_s1_channel",
                    "",
                    choices=ch_names1
                )
                
            ))
            options.append(ui_option_row(
                "Subject 2 channel",
                ui.input_select(
                    "signal_data_files_s2_channel",
                    "",
                    choices=ch_names2
                )
            ))

        return options

    @render.ui
    def ui_input_signal_slider():
        duration = int(get_signal_duration())
        default_duration = min([DEFAULT_SIGNAL_DURATION_SELECTION, duration])
        return ui.input_slider("signal_range", "", min=0, max=duration, value=[0, default_duration], width='100%')

    @render.ui
    def ui_input_s1_mne_duration_slider():
        duration = int(get_signal_duration())
        default_duration = min([DEFAULT_SIGNAL_DURATION_SELECTION, duration])
        return ui.input_slider("input_s1_mne_duration_slider", "", min=0, max=duration, value=[0, default_duration], width='100%')

    @render.plot(height=DEFAULT_PLOT_SIGNAL_HEIGHT)
    def plot_signals():
        fig, ax = plt.subplots()
        pair = get_signals()
        offset = 0
        if input.display_signal_offset_y():
            offset = np.mean(np.abs(pair.y1)) + np.mean(np.abs(pair.y2))
        ax.plot(pair.x, pair.y1 + offset)
        ax.plot(pair.x, pair.y2 - offset)
        return fig
    
    @render.table
    def table_info_s1():
        return pd.DataFrame(get_signals().info_table1, columns=['Key', 'Value'])
        
    @render.table
    def table_info_s2():
        return pd.DataFrame(get_signals().info_table2, columns=['Key', 'Value'])
        

    @render.plot(height=DEFAULT_PLOT_SIGNAL_HEIGHT)
    def plot_signals_spectrum():
        fig, ax = plt.subplots()
        pair = get_signals()
        N = pair.n

        yf = fft.fft(pair.y1)
        xf = fft.fftfreq(N, pair.dt)[:N//2]
        yf2 = fft.fft(pair.y2)

        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.plot(xf, 2.0/N * np.abs(yf2[0:N//2]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.title.set_text('Spectrum')
        ax.grid()
        return fig

    @render.plot()
    def plot_mother_wavelet():
        fig, ax = plt.subplots()
        wav = get_wavelet()
        ax.plot(wav.psi_x, np.real(wav.psi))
        ax.plot(wav.psi_x, np.imag(wav.psi))
        ax.plot(wav.psi_x, np.abs(wav.psi))
        ax.title.set_text(f"mother wavelet ({wav.wavelet_name})")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_daughter_wavelet():
        fig, ax = plt.subplots()

        wtc_res = compute_coherence()
        id = input.display_daughter_wavelet_id()
        psi = wtc_res.tracer['psi_scales'][id]
        ax.plot(np.real(psi))
        ax.plot(np.imag(psi))
        ax.plot(np.abs(psi))
        ax.title.set_text(f"daughter wavelet {id}/{len(wtc_res.tracer['psi_scales'])} for {wtc_res.frequencies[id]:.3f}Hz")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.ui
    def ui_plot_daughter_wavelet():
        # block on this, so that we render only when ready
        _ = compute_coherence()
        
        return [
            ui.output_plot('plot_daughter_wavelet'),
            ui_option_row("Plot daughter wavelet id", ui.input_numeric("display_daughter_wavelet_id", "", value=0), center=True),
        ]
        
    @render.plot()
    def plot_wtc():
        fig, ax = plt.subplots()
        compute_coherence().plot(
            ax=ax,
            colorbar=False,
            downsample=input.display_downsample(),
            show_coif=input.display_show_coif(),
            show_nyquist=input.display_show_nyquist(),
        )
        return fig

    @render.plot()
    def plot_wtc_at_time():
        fig, ax = plt.subplots()

        wtc_res  = compute_coherence()

        if input.display_wtc_frequencies_at_time() is None:
            # region of interest
            # TODO this computation is wrong. Use wtc_masked
            roi = wtc_res.wtc * (wtc_res.wtc > wtc_res.coif[np.newaxis, :]).astype(int)
            col = np.argmax(np.sum(roi, axis=0))
        else:
            col = int(input.display_wtc_frequencies_at_time() / wtc_res.dt)
            
        ax.plot(wtc_res.frequencies, wtc_res.wtc[:,col])
        ax.title.set_text(f'Coherence at t={wtc_res.times[col]:.1f} (max found: {wtc_res.frequencies[np.argmax(wtc_res.wtc[:,col])]:.2f}Hz)')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        return fig

    @render.ui
    def ui_plot_wtc_at_time():
        # block on this, so that we render only when ready
        _ = compute_coherence()
        
        return [
            ui.output_plot('plot_wtc_at_time'),
            ui_option_row("Plot coherence at time", ui.input_numeric("display_wtc_frequencies_at_time", "", value=None), center=True),
        ]
        
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

    def downsample_mat_for_plot(times, frequencies, ZZ):
        if not input.display_downsample():
            return (times, frequencies, ZZ, 1)
        times, ZZ, factor = hypyp.utils.downsample_in_time(times, ZZ, t=500)
        return times, frequencies, ZZ, factor

    def get_fig_plot_tracer_mat(key):
        fig, ax = plt.subplots()
        wtc_res = compute_coherence()
        times, frequencies, ZZ, _ = downsample_mat_for_plot(wtc_res.times, wtc_res.frequencies, ZZ_with_options(wtc_res.tracer[key]))
        im = ax.pcolormesh(times, frequencies, ZZ)
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text(key)
        return fig

    @render.plot()
    def plot_tracer_W1():
        return get_fig_plot_tracer_mat('W1')
    @render.plot()
    def plot_tracer_W2():
        return get_fig_plot_tracer_mat('W2')
    @render.plot()
    def plot_tracer_W12():
        return get_fig_plot_tracer_mat('W12')
    @render.plot()
    def plot_tracer_S1():
        return get_fig_plot_tracer_mat('S1')
    @render.plot()
    def plot_tracer_S2():
        return get_fig_plot_tracer_mat('S2')
    @render.plot()
    def plot_tracer_S12():
        return get_fig_plot_tracer_mat('S12')
    @render.plot()
    def plot_tracer_wtc():
        fig, ax = plt.subplots()
        wtc_res = compute_coherence()
        times, frequencies, ZZ, _ = downsample_mat_for_plot(wtc_res.times, wtc_res.frequencies, ZZ_with_options(wtc_res.wtc))
        im = ax.pcolormesh(times, frequencies, ZZ)
        ax.set_yscale('log')
        fig.colorbar(im, ax=ax)
        ax.title.set_text('wtc')
        return fig

    @render.plot()
    def plot_frequencies():
        fig, ax = plt.subplots()
        wtc_res = compute_coherence()
        ax.scatter(np.arange(len(wtc_res.frequencies)), wtc_res.frequencies, marker='.')
        ax.title.set_text('frequencies')
        return fig

    @render.plot()
    def plot_scales():
        fig, ax = plt.subplots()
        wtc_res = compute_coherence()
        ax.scatter(np.arange(len(wtc_res.scales)), wtc_res.scales, marker='.')
        ax.title.set_text('scales')
        return fig

    @render.ui
    def ui_input_wavelet_type():
        options = []
        if input.wavelet_library() == 'pywavelets':
            options.append(ui.input_select(
                "wavelet_name",
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
        if input.wavelet_library() == 'pywavelets':
            if input.wavelet_name() == 'cmor':
                options.append(ui_option_row("Bandwidth", ui.input_numeric("wavelet_bandwidth", "", value=2)))
                options.append(ui_option_row("Center frequency", ui.input_numeric("wavelet_center_frequency", "", value=1)))

            options.append(ui_option_row("Upper bound", ui.input_numeric("wavelet_upper_bound", "", value=8)))
            options.append(ui_option_row("Precision", ui.input_numeric("wavelet_precision", "", value=10)))
            options.append(ui_option_row("Hack compute each scale", ui.input_checkbox("wavelet_hack_compute_each_scale", "", value=False), sizes=(8,4)))

        if input.wavelet_library() == 'scipy':
            options.append(ui_option_row("Center frequency", ui.input_numeric("wavelet_scipy_center_frequency", "", value=DEFAULT_SCIPY_CENTER_FREQUENCY)))

        return options
    
    @render.ui
    def ui_input_coherence_options():
        options = []
        if input.wavelet_library() in ['pywavelets','scipy']:
            options.append(ui_option_row("Boxcar size", ui.input_numeric("smoothing_boxcar_size", "", value=1)))
        if input.wavelet_library() in ['pycwt']:
            options.append(ui_option_row("Compute significance (slow)", ui.input_checkbox("wavelet_pycwt_significance", "", value=False), sizes=(8,4)))
        return options
    
    @render.ui
    def ui_card_tracer():
        if input.display_show_tracer():
            return ui.card(
                ui.tags.strong('Tracing of intermediary results'),
                ui.row(
                    ui.column(4, ui.output_plot('plot_tracer_W1')),
                    ui.column(4, ui.output_plot('plot_tracer_W2')),
                    ui.column(4, ui.output_plot('plot_tracer_W12')),
                ),
                ui.row(
                    ui.column(4, ui.output_plot('plot_tracer_S1')),
                    ui.column(4, ui.output_plot('plot_tracer_S2')),
                    ui.column(4, ui.output_plot('plot_tracer_S12')),
                ),
                ui.row(
                    ui.column(4),
                    ui.column(4),
                    ui.column(4, ui.output_plot('plot_tracer_wtc')),
                ),
            )

app = App(app_ui, server)