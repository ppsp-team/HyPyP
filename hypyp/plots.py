import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as itertools
from matplotlib.ticker import FuncFormatter
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

# Define a custom locator and formatter for periods
def custom_locator_freqs(ymin, ymax):
    ticks = []
    ticks.extend(range(math.ceil(ymin), 11))
    ticks.extend(range(12, 21, 2))
    ticks.extend(range(25, 40 + 1, 5))
    ticks.extend(range(50, int(ymax) + 1, 10))
    return ticks
    
def plot_wavelet_transform_weights(
    W,
    times,
    freqs,
    coif,
    sfreq,
    bin_seconds=None,
    frequency_cuts=None,
    title=None,
    ax=None,
    show_colorbar=True,
    show_cone_of_influence=True,
    show_nyquist=True,
    show_bins=True,
):
    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    xx, yy = np.meshgrid(times, freqs)
    
    #im = ax.pcolor(xx, yy, W, vmin=0, vmax=1)
    im = ax.pcolor(xx, yy, np.abs(W))
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    color_invalid = 'C0'
    # Cone of influence
    if show_cone_of_influence:
        ax.plot(times, coif, color=color_invalid)
        ax.fill_between(times, coif, y2=np.min(freqs), step="mid", color=color_invalid, alpha=0.4)

    if show_nyquist:
        nyquist = np.ones((len(times),)) * (sfreq / 2)
        ax.plot(times, nyquist, color=color_invalid)
        ax.fill_between(times, nyquist, y2=np.max(freqs), step="mid", color=color_invalid, alpha=0.4)
    
    if show_bins:
        if bin_seconds is not None:
            for time_cut in np.arange(0, max(times), bin_seconds):
                plt.axvline(x=time_cut, color='red', lw=0.5)

        if frequency_cuts is not None:
            for frequency_cut in frequency_cuts:
                plt.axhline(y=frequency_cut, color='red', lw=0.5)
    
    # Dynamically set ticks based on the current range
    ymin, ymax = ax.get_ylim()  # Get the y-axis limits
    ax.set_yticks(custom_locator_freqs(ymin, ymax))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(y)}" if y >= 1 else f"{y:.1f}"))
    #ax.yaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(freqs.min(), freqs.max())

    if show_colorbar:
        fig.colorbar(im)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('CWT Weights')


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

    dyad_selector = (df['is_intra']==False)
    s1_selector = (df['is_intra']==True) & (df['is_intra_of']==1) & (df['channel1']!=df['channel2'])
    s2_selector = (df['is_intra']==True) & (df['is_intra_of']==2) & (df['channel1']!=df['channel2'])
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
        #vmin=0,
        #vmax=1,
        colormap='Greys',
        ax=ax,
        facecolor='white',
        textcolor='black',
        node_edgecolor='black',
        node_colors=['white'],
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
        #vmin=0,
        #vmax=1,
        colormap='Greys',
        ax=ax,
        facecolor='white',
        textcolor='black',
        node_edgecolor='black',
        node_colors=['white'],
    )
    return fig
    

def plot_coherence_bars_per_task(df):
    selector = (df['roi1'] == df['roi2'])

    filtered_df = df[selector]

    # remove intra-subject "same channel" coherence to avoid counting them in means
    intra_same_ch_selector = (filtered_df['is_intra']==True) & (filtered_df['channel1']==filtered_df['channel2'])
    filtered_df[intra_same_ch_selector] = np.nan

    p = sns.catplot(
        data=filtered_df, kind="bar",
        x="roi1", y="coherence", hue="task",
        col="is_intra",
        #palette="dark", alpha=.6, height=6
    )
    p.despine(left=True)
    p.set_axis_labels("", "Coherence")
    p.set_xticklabels(rotation=45)
    p.set(ylim=(0, 1))

    p.legend.set_title("Task")

    p.set_titles('Is intra: {col_name}')

    plt.subplots_adjust(bottom=0.5)
    return p


