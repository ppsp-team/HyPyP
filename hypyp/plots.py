import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as itertools
from matplotlib.ticker import FuncFormatter
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

# Define a custom locator and formatter for periods
def custom_locator_periods(ymin, ymax):
    ticks = []
    ticks.extend(range(math.ceil(ymin), 11))
    ticks.extend(range(12, 21, 2))
    ticks.extend(range(25, int(ymax) + 1, 5))
    return ticks
    
def plot_cwt(W, times, periods, coi, ax=None, title=None, show_colorbar=True, show_coi=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    xx, yy = np.meshgrid(times, periods)
    
    im = ax.pcolor(xx, yy, np.abs(W))
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Period (seconds)')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('CWT Weights')

    if show_colorbar:
        fig.colorbar(im, ax=ax)

    # cone of influence
    if show_coi:
        ax.plot(times, coi)
        ax.fill_between(times, coi, y2=np.max(periods), step="mid", alpha=0.4)

    ymin, ymax = ax.get_ylim()  # Get the y-axis limits
    ax.set_yticks(custom_locator_periods(ymin, ymax))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(y)}" if y >= 1 else ""))
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(periods.min(), periods.max())
    ax.invert_yaxis()

    return fig

def plot_wtc(
    wtc,
    times,
    periods,
    coi,
    sfreq,
    bin_seconds=None,
    period_cuts=None,
    title=None,
    ax=None,
    show_colorbar=True,
    show_coi=True,
    show_nyquist=True,
    show_bins=True,
):
    color_shaded = '0.2'
    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    xx, yy = np.meshgrid(times, periods)
    
    im = ax.pcolor(xx, yy, wtc, vmin=0, vmax=1)
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Period (seconds)')

    # Cone of influence
    if show_coi:
        ax.plot(times, coi, color=color_shaded)
        ax.fill_between(times, coi, y2=np.max(periods), step="mid", color=color_shaded, alpha=0.4)

    if show_nyquist:
        nyquist = np.ones((len(times),)) * sfreq
        nyquist_period = 1 / nyquist
        ax.plot(times, nyquist_period, color=color_shaded)
        ax.fill_between(times, nyquist_period, y2=np.min(periods), step="mid", color=color_shaded, alpha=0.4)
    
    if show_bins:
        if bin_seconds is not None:
            for time_cut in np.arange(0, max(times), bin_seconds):
                plt.axvline(x=time_cut, color='red', lw=0.5)

        if period_cuts is not None:
            for period_cut in period_cuts:
                plt.axhline(y=period_cut, color='red', lw=0.5)
    
    # Dynamically set ticks based on the current range
    ymin, ymax = ax.get_ylim()  # Get the y-axis limits
    ax.set_yticks(custom_locator_periods(ymin, ymax))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(y)}" if y >= 1 else ""))
    #ax.yaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(periods.min(), periods.max())

    ax.invert_yaxis()

    if show_colorbar:
        fig.colorbar(im)

    if title is not None:
        ax.set_title(title)

    return fig

def subplot_heatmap_from_pivot(pivot, ordered_fields, ax):
    index_order = [i for i in ordered_fields if i in pivot.index]
    # Append indices not in ordered_fields
    for i in pivot.index:
        if i not in index_order:
            index_order.append(i)
    column_order = [c for c in ordered_fields if c in pivot.columns]
    for c in pivot.columns:
        if c not in column_order:
            column_order.append(c)

    pivot_reordered = pivot.reindex(index=index_order, columns=column_order)
    heatmap = sns.heatmap(pivot_reordered, cmap='viridis', vmin=0, vmax=1, cbar=False, ax=ax)

    ax.set_xticks(ticks=range(len(pivot_reordered.columns)))
    ax.set_xticklabels(pivot_reordered.columns, rotation=90, ha='left', fontsize=6 if len(column_order)>15 else 10)
    ax.set_yticks(ticks=range(len(pivot_reordered.index)))
    ax.set_yticklabels(pivot_reordered.index, rotation=0, va='top', fontsize=6 if len(index_order)>15 else 10)
    ax.tick_params(axis='both', which='both', length=0)

    return heatmap

def plot_coherence_matrix(
    df,
    s1_label,
    s2_label,
    field1, # roi1 or channel1
    field2, # roi2 or channel2
    ordered_fields,
):
    # We don't sharex and sharey because the list of channels might be different in the 2 subjects

    dyad_selector = (df['subject1']==s1_label) & (df['subject2']==s2_label)
    s1_selector = (df['subject1']==s1_label) & (df['subject2']==s1_label) & (df['channel1']!=df['channel2'])
    s2_selector = (df['subject1']==s2_label) & (df['subject2']==s2_label) & (df['channel1']!=df['channel2'])
    df_dyad = df[dyad_selector]
    df_s1 = df[s1_selector]
    df_s2 = df[s2_selector]

    pivot_s1 = df_s1.pivot_table(index=field1, columns=field2, values='coherence', aggfunc='mean', observed=False)
    pivot_s2 = df_s2.pivot_table(index=field2, columns=field1, values='coherence', aggfunc='mean', observed=False)
    pivot_dyad = df_dyad.pivot_table(index=field1, columns=field2, values='coherence', aggfunc='mean', observed=False)
    
    if np.all(df['is_intra']):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=False, sharey=False)
        subplot_heatmap_from_pivot(pivot_s1.rename_axis(index=s1_label, columns=s1_label), ordered_fields=ordered_fields, ax=ax)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=False, sharey=False)
        subplot_heatmap_from_pivot(pivot_s1.rename_axis(index=s1_label, columns=s1_label), ordered_fields=ordered_fields, ax=axes[0,0]) # top left
        subplot_heatmap_from_pivot(pivot_dyad.rename_axis(index=s1_label, columns=s2_label), ordered_fields=ordered_fields, ax=axes[0,1]) # top right
        subplot_heatmap_from_pivot(pivot_dyad.T.rename_axis(index=s2_label, columns=s1_label), ordered_fields=ordered_fields, ax=axes[1,0]) # bottom left
        subplot_heatmap_from_pivot(pivot_s2.rename_axis(index=s2_label, columns=s2_label), ordered_fields=ordered_fields, ax=axes[1,1]) # bottom right

    #fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.tight_layout()
    return fig
    

def plot_coherence_connectogram(df_pivot, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    else:
        fig = ax.get_figure()
    
    #node_angles = circular_layout(
    #    df.columns, list(df.columns), start_pos=90, group_boundaries=[0, len(df.columns) // 2]
    #)
    plot_connectivity_circle(df_pivot.values,
        df_pivot.columns,
        #node_angles=node_angles,
        title=title,
        vmin=0,
        vmax=1,
        colormap='Greys',
        ax=ax,
        facecolor='white',
        textcolor='black',
        node_edgecolor='black',
    )
    return fig
    
def plot_coherence_connectogram_split(df_pivot, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    else:
        fig = ax.get_figure()
    
    values = df_pivot.values.flatten()
    s1_roi_list = df_pivot.index.to_list()
    s2_roi_list = df_pivot.columns.to_list()
    node_names = s1_roi_list + s2_roi_list

    s1_roi_idx = []
    s2_roi_idx = []

    for i in range(len(s1_roi_list)):
        for j in range(len(s2_roi_list)):
            s1_roi_idx.append(i)
            s2_roi_idx.append(j+len(s1_roi_list))

    indices = (
        np.array(s1_roi_idx),
        np.array(s2_roi_idx),
    )

    node_names_ordered = []
    for i in range(len(s1_roi_list)):
        node_names_ordered.append(s1_roi_list[i])
    for i in range(len(s2_roi_list)):
        node_names_ordered.append(s2_roi_list[len(s2_roi_list)-i-1])

    node_angles = circular_layout(
        node_names, node_names_ordered, start_pos=90, group_boundaries=[0, len(s1_roi_list)]
    )

    plot_connectivity_circle(
        values,
        node_names,
        indices = indices,
        node_angles=node_angles,
        title=title,
        vmin=0,
        vmax=1,
        colormap='Greys',
        ax=ax,
        facecolor='white',
        textcolor='black',
        node_edgecolor='black',
    )
    return fig
    

def plot_coherence_bars_per_task(df, is_intra=False):
    selector = (df['roi1'] == df['roi2']) & (df['is_intra'] == is_intra)
    if is_intra:
        # exclude same channel, since they always have a coherence of 1
        selector = selector & (df['channel1']!=df['channel2'])

    filtered_df = df[selector]

    p = sns.catplot(
        data=filtered_df, kind="bar",
        x="roi1", y="coherence", hue="task",
        #palette="dark", alpha=.6, height=6
    )
    p.despine(left=True)
    p.set_axis_labels("", "Coherence")
    p.set_xticklabels(rotation=45)
    p.set(ylim=(0, 1))

    p.legend.set_title("Task")
    plt.subplots_adjust(bottom=0.5)
    return p


