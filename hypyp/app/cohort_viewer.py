import os
from pathlib import Path
import sys

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

root = os.path.join(Path(__file__).parent, '..', '..')
sys.path.append(root)
import hypyp.plots

from hypyp.fnirs import Cohort, Dyad
from hypyp.wavelet.base_wavelet import WTC

HARDCODED_RESULTS_PATH = "./data/results"

SIDEBAR_WIDTH = 400 # px
DEFAULT_PLOT_CONNECTIVITY_HEIGHT = 1000 # px

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
            "Cohort Info",
            ui.row(
                ui.column(
                    12,
                    ui.output_table('table_info_cohort')
                )
            ),
        ),
        ui.nav_panel(
            "Dyad connectivity",
            ui.row(
                ui.column(
                    12,
                    ui.output_plot('plot_connectivity', height=DEFAULT_PLOT_CONNECTIVITY_HEIGHT)
                )
            ),
        ),
        ui.nav_panel(
            "Wavelet Transform Coherence",
            ui.row(
                ui.column(
                    12,
                    ui.output_plot('plot_wtc', height=DEFAULT_PLOT_CONNECTIVITY_HEIGHT)
                )
            ),
            #ui.output_ui('ui_wtc_tracer'),
        ),
        ui.nav_spacer(),
        #selected='Cohort Info',
        selected='Dyad connectivity',
        #selected='Wavelet Transform Coherence',
        id='main_nav',
        sidebar=ui.sidebar(
            ui.output_ui('ui_input_cohort_file'),
            ui.output_ui('ui_input_select_dyad'),
            ui.output_ui('ui_input_select_task'),
            ui.output_ui('ui_input_select_wtc'),
            width=SIDEBAR_WIDTH,
        ),
        title="HyPyP fNIRS results viewer",
        fillable=True,
        bg="transparent"
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def get_cohort() -> Cohort:
        cohort_file_name = input.select_cohort_file()
        if cohort_file_name == '':
            return None
        cohort_file_path = os.path.join(HARDCODED_RESULTS_PATH, cohort_file_name)
        return Cohort.from_pickle(cohort_file_path)

    @reactive.calc
    def get_dyad() -> Dyad:
        cohort = get_cohort()
        if cohort is None:
            return None
        dyads = [dyad for dyad in cohort.dyads if dyad.label == input.select_dyad()]
        if len(dyads) == 0:
            return None
        return dyads[0]

    @reactive.calc
    def get_wtc() -> WTC:
        dyad = get_dyad()
        if dyad is None:
            return None
        wtcs = [wtc for wtc in dyad.wtcs if wtc.label == input.select_wtc()]
        if len(wtcs) == 0:
            return None
        return wtcs[0]

    @render.ui
    def ui_input_cohort_file():
        my_list = [d for d in os.listdir(HARDCODED_RESULTS_PATH) if d.endswith('.pickle')]
        my_list.sort()
        my_list = [''] + my_list
        #print(my_list)
        return ui.input_select(
            "select_cohort_file",
            f"Cohort file ({HARDCODED_RESULTS_PATH})",
            choices=my_list,
            selected="fnirs_cohort_lionlab_small.pickle",
        )
    
    @render.ui
    def ui_input_select_dyad():
        cohort = get_cohort()
        if cohort is None:
            return None
        return ui.input_select(
            "select_dyad",
            f"Select Dyad",
            choices=[dyad.label for dyad in cohort.dyads],
        )
    
    @render.ui
    def ui_input_select_task():
        dyad = get_dyad()
        if dyad is None:
            return None
        return ui.input_select(
            "select_task",
            f"Select Task",
            choices=[task[0] for task in dyad.tasks],
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
            choices=[wtc.label for wtc in dyad.wtcs if wtc.label.startswith(input.select_task())]
        )
    
    @render.table
    def table_info_cohort():
        cohort = get_cohort()
        if cohort is None:
            return None
        return pd.DataFrame({
            'Dyad Label': [dyad.label for dyad in cohort.dyads],
            'Tasks': [', '.join([task[0] for task in dyad.tasks]) for dyad in cohort.dyads],
            'Subject 1': [dyad.s1.label for dyad in cohort.dyads],
            'Subject 2': [dyad.s2.label for dyad in cohort.dyads],
        })
    
    @render.plot
    def plot_connectivity():
        dyad = get_dyad()
        if dyad is None:
            return None
        return dyad.plot_connectivity_matrix_for_task(input.select_task())
        
    @render.plot
    def plot_wtc():
        fig, ax = plt.subplots()
        wtc = get_wtc()
        if wtc is None:
            return None
        
        wtc.plot(
            ax=ax,
            colorbar=True,
            downsample=False, # downsampling is already done
            show_coif=True,
            show_nyquist=True,
        )
        return fig
    
    #@render.ui
    #def ui_wtc_tracer():
    #    wtc = get_wtc()
    #    if wtc is None or wtc.tracer is None:
    #        return None

    #    fig, axes = plt.subplots(nrows=2, ncols=2)

    #    times, ZZ, _ = hypyp.plots.downsample_in_time(wtc.times, wtc.tracer['W1'], t=500)
    #    axes[0,0].pcolormesh(times, wtc.frequencies, ZZ)
    #    axes[0,0].set_title('Subject 1 CWT')
    #    axes[0,0].set_yscale('log')

    #    times, ZZ, _ = hypyp.plots.downsample_in_time(wtc.times, wtc.tracer['W2'], t=500)
    #    axes[0,1].pcolormesh(times, wtc.frequencies, ZZ)
    #    axes[0,1].set_title('Subject 2 CWT')
    #    axes[0,1].set_yscale('log')

    #    times, ZZ, _ = hypyp.plots.downsample_in_time(wtc.times, wtc.tracer['S1'], t=500)
    #    axes[1,0].pcolormesh(times, wtc.frequencies, ZZ)
    #    axes[1,0].set_title('Subject 1 CWT Smoothed')
    #    axes[1,0].set_yscale('log')

    #    times, ZZ, _ = hypyp.plots.downsample_in_time(wtc.times, wtc.tracer['S2'], t=500)
    #    axes[1,1].pcolormesh(times, wtc.frequencies, ZZ)
    #    axes[1,1].set_yscale('log')

    #    return fig

app = App(app_ui, server)