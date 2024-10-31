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

from hypyp.fnirs.pair_signals import PairSignals
from hypyp.fnirs.data_loader_fnirs import DataLoaderFNIRS
from hypyp.fnirs.subject_fnirs import SubjectFNIRS
from hypyp.signal import SynteticSignal
from hypyp.wavelet.matlab_wavelet import MatlabWavelet
from hypyp.wavelet.pycwt_wavelet import PycwtWavelet
from hypyp.wavelet.scipy_wavelet import ScipyWavelet, DEFAULT_SCIPY_CENTER_FREQUENCY

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

default_plot_signal_height = 150
HARDCODED_PRELOADED_EXTERNAL_PATH = "/media/patrice/My Passport/DataNIRS/"
HARDCODED_PRELOADED_EXTERNAL_FILENAME = "syn_ce05_001.nirs"
HARDCODED_PRELOADED_EXTERNAL_CHANNEL_A = 0
HARDCODED_PRELOADED_EXTERNAL_CHANNEL_B = 19

HARDCODED_PRELOADED_COMMITTED_PATH = './data/'
HARDCODED_PRELOADED_COMMITTED_FILENAME = 'sub-110_session-1_pre_raw.fif'
HARDCODED_PRELOADED_COMMITTED_CHANNEL_A = "S4_D4 hbo"
HARDCODED_PRELOADED_COMMITTED_CHANNEL_B = "S7_D6 hbo"

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
            "Local data browser",
            ui.tags.h4(ui.output_text('text_s1_file_path')),
            ui.input_slider("input_s1_mne_duration_slider", "", min=0, max=100, value=[0, 20], width='100%'),
            ui.navset_card_tab(  
                ui.nav_panel(
                    "Raw",
                    ui.output_image('image_plot_s1_mne_raw'),
                ),
                ui.nav_panel(
                    "Optical density",
                    ui.output_image('image_plot_s1_mne_raw_od'),
                ),
                ui.nav_panel(
                    "Optical density clean",
                    ui.output_image('image_plot_s1_mne_raw_od_clean'),
                ),
                ui.nav_panel(
                    "Hemoglobin",
                    ui.output_image('image_plot_s1_mne_raw_haemo'),
                ),
                ui.nav_panel(
                    "Hemoglobin Filtered",
                    ui.output_image('image_plot_s1_mne_raw_haemo_filtered'),
                ),
                ui.nav_panel(
                    "Hemoglobin PSD",
                    ui.output_image('image_plot_s1_mne_raw_haemo_psd'),
                ),
                ui.nav_panel(
                    "Hemoglobin Filtered PSD",
                    ui.output_image('image_plot_s1_mne_raw_haemo_filtered_psd'),
                ),
                ui.nav_panel(
                    "Signal quality",
                    ui.row(
                        ui.column(3),
                        ui.column(6, ui.output_plot('plot_s1_mne_sci')),
                        ui.column(3),
                    ),
                ),
                #selected="Optical density",
            ),
        ),
        ui.nav_panel(
            "Choice of signals",
            ui.row(
                ui.column(12,
                    ui.card(
                        ui.tags.strong('Choice of signals'),
                        ui.output_ui('ui_input_signal_slider'),
                        ui.output_plot('plot_signals', height=default_plot_signal_height),
                        ui.output_plot('plot_signals_spectrum', height=default_plot_signal_height),
                    ),
                ),
            ),
            ui.row(
                ui.column(6, ui.card(ui.tags.strong('Signal 1'), ui.output_table('table_info_s1'))),
                ui.column(6, ui.card(ui.tags.strong('Signal 2'), ui.output_table('table_info_s2'))),
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
        selected='Local data browser',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.tags.strong('Signal parameters'),
            ui.input_select(
                "signal_type",
                "",
                choices={
                    'data_files': 'Local data files',
                    'preloaded': 'Preloaded signals',
                    'testing': 'Test signals',
                },
            ),
            ui.output_ui('ui_input_signal_choice'),
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
        title="Wavelet Explorer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc()
    def get_signals():
        info_table1 = []
        info_table2 = []
        dyad = None
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
            # select channels from csv of best channels
            s1 = get_subject1()
            s2 = get_subject2()

            s1_selected_ch = s1.__getattribute__(input.signal_data_files_analysis_property()).copy().pick(mne.pick_channels(s1.raw_haemo.ch_names, include = [input.signal_data_files_s1_channel()]))
            s2_selected_ch = s2.__getattribute__(input.signal_data_files_analysis_property()).copy().pick(mne.pick_channels(s2.raw_haemo.ch_names, include = [input.signal_data_files_s2_channel()]))

            y1 = s1_selected_ch.get_data()[0,:]
            y2 = s2_selected_ch.get_data()[0,:]

            # Crop signals
            stop = min(len(y1), len(y2))
            y1 = y1[:stop] 
            y2 = y2[:stop] 
            
            # Force N for this signal
            N = len(y1)
            fs = s1_selected_ch.info['sfreq']
            x = np.linspace(0, N/fs, N)
            

        elif input.signal_type() == 'preloaded':
            if input.signal_preloaded_choice() == 'preloaded_passport_disk':
                file_path = os.path.join(HARDCODED_PRELOADED_EXTERNAL_PATH, HARDCODED_PRELOADED_EXTERNAL_FILENAME)
                dyad = DataLoaderFNIRS.read_two_signals_from_mat(file_path, HARDCODED_PRELOADED_EXTERNAL_CHANNEL_A, HARDCODED_PRELOADED_EXTERNAL_CHANNEL_B)
        
            elif input.signal_preloaded_choice() == 'preloaded_committed':
                file_path = os.path.join(HARDCODED_PRELOADED_COMMITTED_PATH, HARDCODED_PRELOADED_COMMITTED_FILENAME)

                #set events
                tmin = 0 
                tmax = 300
                baseline = (0, 0)

                # use the same file for both
                s1 = mne.io.read_raw_fif(file_path, verbose=True, preload=True)
                s2 = mne.io.read_raw_fif(file_path, verbose=True, preload=True)

                # select channels from csv of best channels
                s1_selected_ch = s1.copy().pick(mne.pick_channels(s1.ch_names, include = [HARDCODED_PRELOADED_COMMITTED_CHANNEL_A]))
                s2_selected_ch = s2.copy().pick(mne.pick_channels(s2.ch_names, include = [HARDCODED_PRELOADED_COMMITTED_CHANNEL_B]))

                # get events
                events1, event_dict1 = mne.events_from_annotations(s1_selected_ch)
                events2, event_dict2 = mne.events_from_annotations(s2_selected_ch)
                epo1 = mne.Epochs(
                    s1_selected_ch,
                    events1,
                    event_id=event_dict1,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    reject_by_annotation=False
                )
                epo2 = mne.Epochs(
                    s2_selected_ch,
                    events2,
                    event_id=event_dict2,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    reject_by_annotation=False
                )
                y1 = epo1.get_data()[0,0,:]
                y2 = epo2.get_data()[0,0,:]
                
                # Force N for this signal
                N = len(y1)
                fs = epo1.info['sfreq']
                x = np.linspace(0, N/fs, N)
                info_table1 = [(k,epo1.info[k]) for k in epo1.info.keys()]
                info_table2 = [(k,epo2.info[k]) for k in epo2.info.keys()]

        if dyad is None:
            dyad = PairSignals(x, y1, y2, info_table1, info_table2)

        try:
            if input.signal_range() is not None:
                return dyad.sub_hundred(input.signal_range())
        except:
            pass
        
        return dyad

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
                wtc_smoothing_smooth_factor=input.smoothing_smooth_factor(),
                wtc_smoothing_boxcar_size=input.smoothing_boxcar_size(),
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
        dyad = get_signals()
        return get_wavelet().wtc(dyad.y1, dyad.y2, dyad.dt)

    @render.text
    def text_s1_file_path():
        return f"File: {input.signal_data_files_s1_path()}"

    @reactive.calc()
    def get_subject1():
        loader = DataLoaderFNIRS()
        return SubjectFNIRS().load_file(loader, input.signal_data_files_s1_path())

    @reactive.calc()
    def get_subject2():
        loader = DataLoaderFNIRS()
        return SubjectFNIRS().load_file(loader, input.signal_data_files_s2_path())

    def get_mne_raw_plot_kwargs(raw, duration_percent_range):
        range_start, range_end = duration_percent_range
        range_start /= 100
        range_end /= 100
        start = range_start * raw.n_times / raw.info['sfreq']
        duration = (range_end - range_start) * raw.n_times / raw.info['sfreq']
        return dict(
            n_channels=len(raw.ch_names),
            start=start,
            duration=duration,
            scalings='auto',
            show_scrollbars=False,
            block=True,
            show=False,
        )

    def mne_figure_as_image(fig):
        temp_img_path = tempfile.NamedTemporaryFile(suffix=".png").name
        fig.savefig(temp_img_path)
        return {"src": temp_img_path, "alt": "MNE Plot"}

    @render.image
    def image_plot_s1_mne_raw():
        subject = get_subject1()
        fig = subject.raw.plot(**get_mne_raw_plot_kwargs(subject.raw, input.input_s1_mne_duration_slider()))
        return mne_figure_as_image(fig)


    @render.image
    def image_plot_s1_mne_raw_od():
        subject = get_subject1()
        fig = subject.raw_od.plot(**get_mne_raw_plot_kwargs(subject.raw_od, input.input_s1_mne_duration_slider()))
        return mne_figure_as_image(fig)

    @render.image
    def image_plot_s1_mne_raw_od_clean():
        subject = get_subject1()
        fig = subject.raw_od_clean.plot(**get_mne_raw_plot_kwargs(subject.raw_od_clean, input.input_s1_mne_duration_slider()))
        return mne_figure_as_image(fig)
    
    @render.image
    def image_plot_s1_mne_raw_haemo():
        subject = get_subject1()
        fig = subject.raw_haemo.plot(**get_mne_raw_plot_kwargs(subject.raw_haemo, input.input_s1_mne_duration_slider()), theme="light")
        return mne_figure_as_image(fig)
    
    @render.image
    def image_plot_s1_mne_raw_haemo_psd():
        subject = get_subject1()
        fig = subject.raw_haemo.compute_psd().plot()
        return mne_figure_as_image(fig)
    
    @render.image
    def image_plot_s1_mne_raw_haemo_filtered():
        subject = get_subject1()
        fig = subject.raw_haemo_filtered.plot(**get_mne_raw_plot_kwargs(subject.raw_haemo_filtered, input.input_s1_mne_duration_slider()), theme="light")
        return mne_figure_as_image(fig)
    
    @render.image
    def image_plot_s1_mne_raw_haemo_filtered_psd():
        subject = get_subject1()
        fig = subject.raw_haemo_filtered.compute_psd().plot()
        return mne_figure_as_image(fig)
    
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
            loader = DataLoaderFNIRS()
            loader.download_demo_dataset()

            choices.append(ui_option_row("Signal 1 file", ui.input_select(
                "signal_data_files_s1_path",
                "",
                choices=loader.list_all_files(),
                # for comparison with jupyter notebook
                #selected="/home/patrice/work/ppsp/HyPyP-synchro/data/fNIRS/downloads/fathers/FCS28/parent/NIRS-2019-11-10_003.hdr",
            )))
            choices.append(ui_option_row("Signal 2 file", ui.input_select(
                "signal_data_files_s2_path",
                "",
                choices=loader.list_all_files(),
            )))
        
        elif input.signal_type() == 'preloaded':
            choices.append(ui.input_select(
                "signal_preloaded_choice",
                "",
                choices={
                    'preloaded_committed': 'fNIRS sub 110 session 1',
                    'preloaded_passport_disk': 'fNIRS data from Passport Disk',
                },
            ))
        
        return choices

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
            options.append(ui_option_row(
                "Signal 1 channel",
                ui.input_select(
                    "signal_data_files_s1_channel",
                    "",
                    choices=get_subject1().raw_haemo.ch_names,
                )
            ))
            options.append(ui_option_row(
                "Signal 2 channel",
                ui.input_select(
                    "signal_data_files_s2_channel",
                    "",
                    choices=get_subject2().raw_haemo.ch_names,
                )
            ))
            options.append(ui_option_row(
                "What signal to use",
                ui.input_select(
                    "signal_data_files_analysis_property",
                    "",
                    choices=get_subject1().get_analysis_properties(),
                )
            ))

        return options

    @render.ui
    def ui_input_signal_slider():
        options = []
        try:
            options.append(ui.input_slider("signal_range", "", min=0, max=100, value=[0, 100], width='100%'))
        except:
            pass
            
        return options

    @render.plot(height=default_plot_signal_height)
    def plot_signals():
        fig, ax = plt.subplots()
        dyad = get_signals()
        offset = 0
        if input.display_signal_offset_y():
            offset = np.mean(np.abs(dyad.y1)) + np.mean(np.abs(dyad.y2))
        ax.plot(dyad.x, dyad.y1 + offset)
        ax.plot(dyad.x, dyad.y2 - offset)
        return fig
    
    @render.table
    def table_info_s1():
        return pd.DataFrame(get_signals().info_table1, columns=['Key', 'Value'])
        
    @render.table
    def table_info_s2():
        return pd.DataFrame(get_signals().info_table2, columns=['Key', 'Value'])
        

    @render.plot(height=default_plot_signal_height)
    def plot_signals_spectrum():
        fig, ax = plt.subplots()
        dyad = get_signals()
        N = dyad.n

        yf = fft.fft(dyad.y1)
        xf = fft.fftfreq(N, dyad.dt)[:N//2]
        yf2 = fft.fft(dyad.y2)

        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.plot(xf, 2.0/N * np.abs(yf2[0:N//2]))
        ax.set_xscale('log')
        ax.set_yscale('log')
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

        dyad = get_signals()
        wtc_res  = compute_coherence()

        if input.display_wtc_frequencies_at_time() is None:
            # region of interest
            roi = wtc_res.wtc * (wtc_res.wtc > wtc_res.coif[np.newaxis, :]).astype(int)
            col = np.argmax(np.sum(roi, axis=0))
        else:
            col = int(input.display_wtc_frequencies_at_time() / dyad.dt)
            
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
        times, ZZ, factor = hypyp.plots.downsample_in_time(times, ZZ, t=500)
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
            options.append(ui_option_row("Smooth factor", ui.input_numeric("smoothing_smooth_factor", "", value=-0.1)))
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