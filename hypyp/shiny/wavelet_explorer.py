import os
import tempfile

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import fft
import mne

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.fnirs.data_browser import DataBrowser
from hypyp.fnirs.recording import Recording
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_raw_to_haemo import MnePreprocessorRawToHaemo
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_as_is import MnePreprocessorAsIs
from hypyp.fnirs.preprocessor.implementations.cedalion_preprocessor import CedalionPreprocessor
from hypyp.signal import SyntheticSignal
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.matlab_wavelet import MatlabWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexGaussianWavelet, ComplexMorletWavelet
from hypyp.wavelet.implementations.pycwt_wavelet import PycwtWavelet


DEFAULT_PLOT_SIGNAL_HEIGHT = 150 # px
DEFAULT_PLOT_MNE_HEIGHT = 1200 # px
DEFAULT_SIGNAL_DURATION_SELECTION = 60 # seconds

# Use this environment variable to add a lookup path for NIRS data
HYPYP_NIRS_DATA_PATH = os.getenv("HYPYP_NIRS_DATA_PATH", "")

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
            "Wavelet Coherence",
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
                ui.column(6,
                    ui.card(
                        ui.tags.strong('Wavelet'),
                        ui.output_plot('plot_mother_wavelet'),
                    ),
                ),
                ui.column(6,
                    ui.card(
                        ui.tags.strong('Wavelet Coherence'),
                        ui.output_plot('plot_wtc'),
                    ),
                ),
            ),
        ),
        ui.nav_spacer(),
        #selected='Data Browser',
        selected='Wavelet Coherence',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.tags.strong('Signal parameters'),
            ui.input_select(
                "signal_type",
                "",
                choices={
                    'data_files': 'Local data files',
                    'testing': 'Synthetic signals',
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
                    'pycwt': 'Use pycwt (based on matlab code)',
                    'matlab': 'Use Matlab engine',
                },
            ),
            ui.output_ui('ui_input_wavelet_type'),
            ui.output_ui('ui_input_wavelet_options'),
            ui.output_ui('ui_input_coherence_options'),

            ui.input_action_button("button_action_compute_wtc", label="Compute Wavelet Transform Coherence"),

            ui.tags.strong('Display parameters'),
            ui_option_row("Signal offset y", ui.input_checkbox("display_signal_offset_y", "", value=True), sizes=(8,4)),
            ui_option_row("Downsample", ui.input_checkbox("display_downsample", "", value=True), sizes=(8,4)),
            ui_option_row("Show COI", ui.input_checkbox("display_show_coi", "", value=True), sizes=(8,4)),
            ui_option_row("Show Nyquist", ui.input_checkbox("display_show_nyquist", "", value=True), sizes=(8,4)),
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
        pair = None
        if input.signal_type() == 'testing':
            fs = input.signal_sampling_frequency()
            N = input.signal_n()
            x = np.linspace(0, N/fs, N)
            tmax = N / fs
            if input.signal_testing_choice() == 'sinusoid':
                y1 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_sinusoid_freq1()).y
                y2 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_sinusoid_freq2()).y
            elif input.signal_testing_choice() == 'sinusoid_almost_similar':
                y1 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_sinusoid_almost_similar_freq1()).y
                y2 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_sinusoid_almost_similar_freq1() - 0.001).y
            elif input.signal_testing_choice() == 'sinusoid_dephased':
                y1 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_sinusoid_dephased_freq1()).y
                y1_fft = np.fft.fft(y1)
                y1_magnitude = np.abs(y1_fft)
                y1_phase = np.angle(y1_fft)
                y1_random_phase = np.random.uniform(-np.pi, np.pi, len(y1_phase))
                y1_randomized_fft = y1_magnitude * np.exp(1j * y1_random_phase)
                y2 = np.fft.ifft(y1_randomized_fft).real
                y2 = y2 * 1 / (np.abs(np.max(y2)-np.min(y2)) / 2)
            elif input.signal_testing_choice() == 'chirp':
                signal1 = SyntheticSignal(duration=tmax, n_points=N)
                signal1.add_chirp(input.signal_chirp_freq1(), input.signal_chirp_freq2())
                y1 = signal1.y
                y2 = np.flip(y1)
            elif input.signal_testing_choice() == 'chirp_sinusoid':
                y1 = SyntheticSignal(duration=tmax, n_points=N).add_sin(input.signal_chirp_sinusoid_freq1()).y
                y2 = SyntheticSignal(duration=tmax, n_points=N).add_chirp(input.signal_chirp_sinusoid_freq2(), input.signal_chirp_sinusoid_freq3()).y

            noise_level = input.signal_noise_level()
            y1 +=  noise_level * np.random.normal(0, 1, len(x))
            y2 +=  noise_level * np.random.normal(0, 1, len(x))

        elif input.signal_type() == 'data_files':
            # TODO this only works with MNE
            s1_raw = get_recording_s1_step().obj.copy()
            s2_raw = get_recording_s2_step().obj.copy()

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

        if pair is None:
            pair = PairSignals(
                x,
                y1,
                y2,
                label_s1='s1',
                label_s2='s2',
                label_task='foo',
            )

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
            return min([get_recording_s1_step().duration, get_recording_s2_step().duration])
            

    @reactive.calc()
    def get_wavelet():
        if input.wavelet_library() == 'pywavelets':
            wavelet_name = input.wavelet_name()
            if wavelet_name == 'cmor':
                return ComplexMorletWavelet(
                    bandwidth_frequency=input.wavelet_bandwidth(),
                    center_frequency=input.wavelet_center_frequency(),
                    period_range=(input.wavelet_period_range_low(), input.wavelet_period_range_high()),
                    disable_caching=True,
                )
            elif wavelet_name == 'cgau':
                return ComplexGaussianWavelet(
                    degree=int(input.wavelet_degree()),
                    period_range=(input.wavelet_period_range_low(), input.wavelet_period_range_high()),
                    wtc_smoothing_win_size=input.smoothing_win_size(),
                    disable_caching=True,
                )
            else:
                raise RuntimeError(f'Unknown wavelet_name: {wavelet_name}')

        if input.wavelet_library() == 'pycwt':
            return PycwtWavelet()

        if input.wavelet_library() == 'matlab':
            return MatlabWavelet()

        raise RuntimeError(f'Unknown wavelet library: {input.wavelet_library()}')

    @reactive.event(input.button_action_compute_wtc)
    def compute_coherence():
        wavelet = get_wavelet()
        pair = get_signals()
        wtc = wavelet.wtc(pair)
        if input.display_downsample():
            wtc.downsample_in_time(500)
        return wtc

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
        data_browser = DataBrowser()
        if HYPYP_NIRS_DATA_PATH != "":
            data_browser.add_source(HYPYP_NIRS_DATA_PATH)
        return data_browser

    def get_preprocessor():
        value = input.recording_preprocessor()
        if value == 'upstream':
            return MnePreprocessorAsIs()
        if value == 'mne':
            return MnePreprocessorRawToHaemo()
        if value == 'cedalion':
            return CedalionPreprocessor()
        raise RuntimeError(f'Unknown preprocessor type "{value}"')
        

    @reactive.calc()
    def get_recording_s1():
        return Recording().load_file(get_signal_data_files_s1_path(), get_preprocessor())

    @reactive.calc()
    def get_recording_s2():
        return Recording().load_file(get_signal_data_files_s2_path(), get_preprocessor())

    @reactive.calc()
    def get_recording_s1_step():
        return get_recording_s1().get_preprocess_step(input.signal_data_files_analysis_property())

    @reactive.calc()
    def get_recording_s2_step():
        return get_recording_s2().get_preprocess_step(input.signal_data_files_analysis_property())

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
            # TODO this is here because of CedalionStep. This code flow is ugly
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
        recording = get_recording_s1()
        for step in recording.preprocess_steps:
            nav_panels.append(ui.nav_panel(step.desc, bind_plot_mne_figure(step)))

        return ui.navset_card_tab(*nav_panels, selected=recording.preprocess_steps[-1].desc)

    @render.plot
    def plot_s1_mne_sci():
        recording = get_recording_s1()
        fig, ax = plt.subplots()
        ax.hist(recording.quality_sci)
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
                    'chirp': 'Crossing chirps',
                    'chirp_sinusoid': 'One sinusoid, one chirp',
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
                #selected="/home/patrice/work/ppsp/HyPyP-synchro/data/NIRS/downloads/fathers/FCS28/parent/NIRS-2019-11-10_003.hdr",
            )))
            choices.append(ui_option_row("Subject 2 file", ui.input_select(
                "signal_data_files_s2_path",
                "",
                choices=[STR_SAME_AS_SUBJECT_1] + browser.list_all_files(),
            )))
            choices.append(ui_option_row("Preprocessor Class", ui.input_select(
                "recording_preprocessor",
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
                    choices=get_recording_s1().preprocess_step_choices,
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
                options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_freq2", "", value=0.05))),
            elif input.signal_testing_choice() == 'chirp_sinusoid':
                options.append(ui_option_row("Freq Sinusoid", ui.input_numeric("signal_chirp_sinusoid_freq1", "", value=1))),
                options.append(ui_option_row("Freq Chirp from", ui.input_numeric("signal_chirp_sinusoid_freq2", "", value=0.2))),
                options.append(ui_option_row("Freq Chirp to", ui.input_numeric("signal_chirp_sinusoid_freq3", "", value=2))),
        
            options.append(ui_option_row("Sampling freq. (Hz)", ui.input_numeric("signal_sampling_frequency", "", value=5))),
            options.append(ui_option_row("Nb. points", ui.input_numeric("signal_n", "", value=2000))),
            options.append(ui_option_row("Noise level", ui.input_numeric("signal_noise_level", "", value=0.1))),
        
        if input.signal_type() == 'data_files':
            # TODO this try-except is here to have a PoC of Cedalion integration
            try:
                ch_names1 = get_recording_s1().preprocessed.ch_names
                ch_names2 = get_recording_s2().preprocessed.ch_names
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
        ax.title.set_text(f"mother wavelet ({wav.wavelet_name_with_args})")
        ax.legend(['real', 'imag', 'abs'])
        return fig

    @render.plot()
    def plot_wtc():
        fig, ax = plt.subplots()
        compute_coherence().plot(
            ax=ax,
            show_colorbar=False,
            show_cone_of_influence=input.display_show_coi(),
            show_nyquist=input.display_show_nyquist(),
        )
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
                    'cgau': 'Complex Gaussian',
                },
            ))
        return options


    @render.ui
    def ui_input_wavelet_options():
        options = []
        if input.wavelet_library() == 'pywavelets':
            if input.wavelet_name() == 'cmor':
                options.append(ui_option_row("Bandwidth/Center Freq.", ui.row(
                    ui.column(6, ui.input_numeric("wavelet_bandwidth", "", value=ComplexMorletWavelet.default_bandwidth_frequency)),
                    ui.column(6, ui.input_numeric("wavelet_center_frequency", "", value=ComplexMorletWavelet.default_center_frequency)),
                )))
            if input.wavelet_name() == 'cgau':
                options.append(ui_option_row("Gaussian degree", ui.row(
                   ui.input_select(
                        "wavelet_degree",
                        "",
                        choices=list(range(1,9)),
                        selected=ComplexGaussianWavelet.default_degree,
                    ))))

        options.append(ui_option_row("Period range", ui.row(
            ui.column(6, ui.input_numeric("wavelet_period_range_low", "", value=BaseWavelet.default_period_range[0])),
            ui.column(6, ui.input_numeric("wavelet_period_range_high", "", value=BaseWavelet.default_period_range[1])),
        )))

        return options
    
    @render.ui
    def ui_input_coherence_options():
        options = []
        if input.wavelet_library() in ['pycwt']:
            pass
        return options
    
app = App(app_ui, server)