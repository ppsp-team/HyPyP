import os
from pathlib import Path
import sys
import tempfile

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots

from hypyp.fnirs import Study, Dyad
from hypyp.wavelet.base_wavelet import WTC

HARDCODED_RESULTS_PATH = "./data/results"

SIDEBAR_WIDTH = 400 # px
DEFAULT_PLOT_COHERENCE_MATRIX_HEIGHT = 1000
DEFAULT_PLOT_COHERENCE_PER_TASK_HEIGHT = 600
DEFAULT_PLOT_WTC_HEIGHT = 600
DEFAULT_PLOT_CONNECTOGRAM_HEIGHT = 1000

STR_ALL_DYADS = 'All dyads'
STR_ALL_TASKS = 'All tasks'

# This is to avoid having external windows launched
matplotlib.use('Agg')

def mne_figure_as_image(fig):
    temp_img_path = tempfile.NamedTemporaryFile(suffix=".png").name
    fig.savefig(temp_img_path)
    return {"src": temp_img_path, "alt": "MNE Plot"}
        
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
            "Study Info",
            ui.row(
                ui.column(
                    12,
                    ui.output_table('table_info_study')
                )
            ),
        ),
        ui.nav_panel(
            "Coherence Matrix",
            ui.row(
                ui.column(
                    10,
                    ui.output_plot('plot_coherence_matrix', height=DEFAULT_PLOT_COHERENCE_MATRIX_HEIGHT)
                ),
                ui.column(
                    2,
                    ui.input_select(
                        'coherence_select_grouping',
                        'Coherence grouping',
                        choices={
                            'roi': 'Region of Interest',
                            'channel': 'Individual Channels',
                            'channel_roi': 'Channel-ROI',
                            'roi_channel': 'ROI-Channel',
                        })
                ),
            ),
        ),
        ui.nav_panel(
            "Connectograms",
            ui.row(
                ui.column(
                    4,
                    ui.output_plot('plot_connectogram_s1', height=DEFAULT_PLOT_CONNECTOGRAM_HEIGHT)
                ),
                ui.column(
                    4,
                    ui.output_plot('plot_connectogram_s2', height=DEFAULT_PLOT_CONNECTOGRAM_HEIGHT)
                ),
                ui.column(
                    4,
                    ui.output_plot('plot_connectogram', height=DEFAULT_PLOT_CONNECTOGRAM_HEIGHT)
                ),
            ),
        ),
        ui.nav_panel(
            "Coherence Per Task",
            ui.row(
                ui.column(
                    12,
                    ui.output_plot('plot_coherence_per_task', height=DEFAULT_PLOT_COHERENCE_PER_TASK_HEIGHT)
                ),
            ),
        ),
        ui.nav_panel(
            "Data Frame",
            ui.row(
                ui.column(
                    12,
                    ui.output_data_frame('data_frame')
                ),
            ),
        ),
        ui.nav_panel(
            "Wavelet Transform Coherence",
            ui.row(
                ui.column(
                    10,
                    ui.output_plot('plot_wtc', height=DEFAULT_PLOT_WTC_HEIGHT)
                ),
                ui.column(
                    2,
                    ui.output_ui('ui_input_select_wtc'),
                ),
            ),
        ),
        ui.nav_spacer(),
        selected='Study Info',
        #selected='Coherence Matrix',
        #selected='Connectograms',
        #selected='Coherence Per Task',
        #selected='Wavelet Transform Coherence',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.output_ui('ui_input_study_file'),
            ui.output_ui('ui_input_select_dyad'),
            ui.output_ui('ui_input_select_task'),
            width=SIDEBAR_WIDTH,
        ),
        title="HyPyP fNIRS results viewer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def get_study() -> Study:
        study_file_name = input.select_study_file()
        if study_file_name == '':
            return None
        study_file_path = os.path.join(HARDCODED_RESULTS_PATH, study_file_name)
        return Study.from_pickle(study_file_path)

    @reactive.calc
    def get_dyad() -> Dyad:
        study = get_study()
        if study is None:
            return None

        if input.select_dyad() is None or input.select_dyad() == STR_ALL_DYADS:
            return None
            
        dyads = [dyad for dyad in study.dyads if dyad.label == input.select_dyad()]
        if len(dyads) == 0:
            return None
        return dyads[0]

    @reactive.calc
    def get_wtc() -> WTC:
        dyad = get_dyad()
        if dyad is None:
            return None
        wtcs = [wtc for wtc in dyad.wtcs if wtc.label_pair == input.select_wtc()]
        if len(wtcs) == 0:
            return None
        return wtcs[0]

    @render.ui
    def ui_input_study_file():
        my_list = [d for d in os.listdir(HARDCODED_RESULTS_PATH) if d.endswith('.pickle')]
        my_list.sort()
        my_list = [''] + my_list
        #print(my_list)
        return ui.input_select(
            "select_study_file",
            f"Study file ({HARDCODED_RESULTS_PATH})",
            choices=my_list,
            selected="fnirs_study_lionlab.pickle",
        )
    
    @render.ui
    def ui_input_select_dyad():
        study = get_study()
        if study is None:
            return None
        return ui.input_select(
            "select_dyad",
            f"Select Dyad",
            choices=[STR_ALL_DYADS] + [dyad.label for dyad in study.dyads],
        )
    
    @render.ui
    def ui_input_select_task():
        study = get_study()
        dyad = get_dyad()
        if dyad is not None:
            tasks = dyad.tasks
        elif study is not None:
            tasks = study.dyads[0].tasks
        else:
            return None

        return ui.input_select(
            "select_task",
            f"Select Task",
            choices=[STR_ALL_TASKS] + [task.name for task in tasks],
        )
    
    @render.ui
    def ui_input_select_wtc():
        dyad = get_dyad()
        if dyad is None:
            return None
        return ui.input_select(
            "select_wtc",
            f"Select Channel Pair",
            # TODO the filtering here is sketchy
            choices=[wtc.label_pair for wtc in dyad.wtcs if wtc.label_pair.startswith(input.select_task())]
        )
    
    @render.table
    def table_info_study():
        study = get_study()
        if study is None:
            return None
        return pd.DataFrame({
            'Dyad Label': [dyad.label for dyad in study.dyads],
            'Tasks': [', '.join([task.name for task in dyad.tasks]) for dyad in study.dyads],
            'Subject 1': [dyad.s1.subject_label for dyad in study.dyads],
            'Subject 2': [dyad.s2.subject_label for dyad in study.dyads],
        })
    
    def get_query():
        task = input.select_task() 
        q = f'task == "{task}"' if task != STR_ALL_TASKS else None
        return q

    @render.plot
    def plot_coherence_matrix():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None

        grouping = input.coherence_select_grouping()
        if grouping == 'roi':
            return obj.plot_coherence_matrix_per_roi(query=get_query())
        elif grouping == 'channel':
            return obj.plot_coherence_matrix_per_channel(query=get_query())
        elif grouping == 'roi_channel':
            return obj.plot_coherence_matrix('roi1', 'channel2', query=get_query())
        elif grouping == 'channel_roi':
            return obj.plot_coherence_matrix('channel1', 'roi2', query=get_query())
        else:
            raise RuntimeError(f'Unknown grouping {grouping}')

    @render.plot
    def plot_connectogram_s1():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None
        return obj.plot_coherence_connectogram_s1(query=get_query(), title='Subject1')
        
    @render.plot
    def plot_connectogram_s2():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None
        return obj.plot_coherence_connectogram_s2(query=get_query(), title='Subject2')
        
    @render.plot
    def plot_connectogram():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None
        return obj.plot_coherence_connectogram(query=get_query())
        
    @render.data_frame
    def data_frame():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None
        # have a maximum so that it loads
        return render.DataGrid(obj.df[:1000], selection_mode="rows")
        
    @render.plot
    def plot_coherence_per_task():
        obj = get_dyad()
        if obj is None:
            obj = get_study()
        if obj is None:
            return None
        return obj.plot_coherence_bars_per_task()
        
    @render.plot
    def plot_wtc():
        fig, ax = plt.subplots()
        wtc = get_wtc()
        if wtc is None:
            return None
        
        wtc.plot(
            ax=ax,
            show_colorbar=True,
            show_cone_of_influence=True,
            show_nyquist=True,
        )
        return fig
    

app = App(app_ui, server)