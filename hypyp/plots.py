import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as itertools
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.measure import block_reduce
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


def spectrogram_plot(z, times, frequencies, coif, cmap="viridis", norm=Normalize(), ax=None, colorbar=True):
    ###########################################################################
    # plot
    
    # set default colormap, if none specified
    if cmap is None:
        cmap = cm.get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    xx,yy = np.meshgrid(times,frequencies)
    ZZ = z
    
    if norm is None:
        im = ax.pcolor(xx,yy,ZZ, cmap=cmap)
        ax.plot(times,coif)
        ax.fill_between(times,coif, step="mid", alpha=0.4)
    else:
        im = ax.pcolor(xx,yy,ZZ, norm=norm, cmap=cmap)
        ax.plot(times,coif)
        ax.fill_between(times,coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())
    ax.title.set_text('Wavelet coherence')

    return ax

def plot_wtc(
    wtc,
    times,
    frequencies,
    coi,
    sfreq,
    ax=None,
    colorbar=True,
    show_coi=True,
    show_nyquist=True,
    title=None,
):
    color_shaded = '0.2'
    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    times_orig = times
    
    periods = 1 / frequencies
    xx, yy = np.meshgrid(times, periods)
    
    #im = ax.pcolor(xx, yy, wtc, norm=Normalize())
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
    
    # Define a custom locator and formatter
    def custom_locator(ymin, ymax):
        ticks = []
        ticks.extend(range(math.ceil(ymin), 11))
        ticks.extend(range(12, 21, 2))
        ticks.extend(range(25, int(ymax) + 1, 5))
        return ticks
    
    # Dynamically set ticks based on the current range
    ymin, ymax = ax.get_ylim()  # Get the y-axis limits
    ax.set_yticks(custom_locator(ymin, ymax))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(y)}" if y >= 1 else ""))
    #ax.yaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(periods.min(), periods.max())

    ax.invert_yaxis()


    if colorbar:
        #cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        #fig.colorbar(im, cax=cbaxes, orientation='vertical')
        fig.colorbar(im)

    if title is not None:
        fig.suptitle(title)

    return ax

def plot_coherence_matrix(
        z,
        ch_names1,
        ch_names2,
        label1,
        label2,
        title='Coherence matrix',
        with_intra=False,
        ax=None):
    # create the figure if needed
    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(z, cmap='viridis', cbar=True, ax=ax, vmin=0, vmax=1)

    if title != '':
        ax.set_title(title)

    # Set x and y ticks
    ax.set_yticks(ticks=np.arange(len(ch_names1)), labels=ch_names1, fontsize=6, rotation=0)
    ax.set_xticks(ticks=np.arange(len(ch_names2)), labels=ch_names2, fontsize=6, rotation=90)

    if with_intra:
        x_quadrant_labels = [label2, label1]
        y_quadrant_labels = [label2, label1]

        x_quadrant_boundaries = [0, len(ch_names1)//2, len(ch_names1)]
        y_quadrant_boundaries = [0, len(ch_names1)//2, len(ch_names1)]

        x_quadrant_positions = [(x_quadrant_boundaries[i] + x_quadrant_boundaries[i+1]) / 2 for i in range(len(x_quadrant_boundaries) - 1)]
        for pos, label in zip(x_quadrant_positions, x_quadrant_labels):
            ax.text(pos, -0.2, label, ha='center', va='center', fontsize=12, transform=ax.get_xaxis_transform())

        y_quadrant_positions = [(y_quadrant_boundaries[i] + y_quadrant_boundaries[i+1]) / 2 for i in range(len(y_quadrant_boundaries) - 1)]
        for pos, label in zip(y_quadrant_positions, y_quadrant_labels):
            ax.text(-0.2, pos, label, ha='center', va='center', fontsize=12, rotation=90, transform=ax.get_yaxis_transform())
    
    else:
        ax.set_xlabel(label2)
        ax.set_ylabel(label1)

    return ax

def plot_coherence_matrix_df(
    df,
    s1_label,
    s2_label,
    field1, # roi1 or channel1
    field2, # roi2 or channel2
    ordered_fields,
):
    # We don't sharex and sharey because the list of channels might be different in the 2 subjects
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=False, sharey=False)

    def heatmap_from_pivot(pivot, ax):
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

    dyad_selector = (df['subject1']==s1_label) & (df['subject2']==s2_label)
    s1_selector = (df['subject1']==s1_label) & (df['subject2']==s1_label) & (df['channel1']!=df['channel2'])
    s2_selector = (df['subject1']==s2_label) & (df['subject2']==s2_label) & (df['channel1']!=df['channel2'])
    df_dyad = df[dyad_selector]
    df_s1 = df[s1_selector]
    df_s2 = df[s2_selector]

    pivot_s1 = df_s1.pivot_table(index=field1, columns=field2, values='coherence', aggfunc='mean')
    pivot_s2 = df_s2.pivot_table(index=field2, columns=field1, values='coherence', aggfunc='mean')
    pivot_dyad = df_dyad.pivot_table(index=field1, columns=field2, values='coherence', aggfunc='mean')
    
    heatmap_from_pivot(pivot_s1.rename_axis(index=s1_label, columns=s1_label), ax=axes[0,0])
    heatmap_from_pivot(pivot_dyad.rename_axis(index=s1_label, columns=s2_label), ax=axes[0,1])
    heatmap_from_pivot(pivot_dyad.T.rename_axis(index=s2_label, columns=s1_label), ax=axes[1,0])
    heatmap_from_pivot(pivot_s2.rename_axis(index=s2_label, columns=s2_label), ax=axes[1,1])

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig
    

def plot_connectogram(df, title=''):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    #node_angles = circular_layout(
    #    df.columns, list(df.columns), start_pos=90, group_boundaries=[0, len(df.columns) // 2]
    #)
    plot_connectivity_circle(df.values,
        df.columns,
        #node_angles=node_angles,
        title=title,
        colormap='Greys',
        ax=ax,
        facecolor='white',
        textcolor='black',
        node_edgecolor='black',
    )
    return fig
    

def plot_coherence_per_task_bars(df, is_intra=False):
    selector = (df['roi1'] == df['roi2']) & (df['is_intra'] == is_intra)
    if is_intra:
        # exclude same channel, since they always have a coherence of 1
        selector = selector & (df['channel1']!=df['channel2'])

    filtered_df = df[selector]

    # Draw a nested barplot by species and sex
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


def spectrogram_plot_period(
    z,
    times,
    frequencies,
    coif,
    cmap="viridis",
    norm=Normalize(),
    ax=None,
    colorbar=True
):
    ###########################################################################
    # plot
    
    # set default colormap, if none specified
    if cmap is None:
        cmap = cm.get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    #periods = np.log2(5/frequencies)
    periods = np.log2(1/frequencies)

    #ntimes = 5*times
    ntimes = times
    
    xx,yy = np.meshgrid(times, periods)
    ZZ = z
    
    im = ax.pcolor(xx, yy, ZZ, cmap=cmap)
    #im = ax.pcolormesh(ZZ)
    #ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('Period')
    ax.plot(ntimes,coif)
    ax.fill_between(times,coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    #ax.set_xlim(times.min(), times.max())
    #ax.set_ylim(periods.min(), periods.max())

    steps = np.arange(0, len(periods), 10)
    ax.set_yticks(np.round(periods[steps], 2), np.round(2**(periods[steps]), 2))
    
    #ax.invert_yaxis()
    
    return ax

def plot_cwt_weights(W, times, frequencies, coif):
    fig, ax = plt.subplots()
    im = ax.pcolormesh(times, frequencies, np.abs(W))
    ax.set_yscale('log')
    fig.colorbar(im, ax=ax)
    ax.title.set_text('Weights')

    # cone of influence
    ax.plot(times, coif)
    ax.fill_between(times, coif, step="mid", alpha=0.4)
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())



def plot_line(items, key, title, use_log_scale=False):
    fig, axes = plt.subplots(1, len(items), figsize=(18,4))
    fig.suptitle(title)
    for i in range(len(items)):
        item = items[i]
        try:
            print(f"{title}: {item[key].shape}")
            axes[i].plot(item[key])
            if use_log_scale:
                axes[i].set_yscale('log')
        except Exception as e:
            print(e)
            pass
    plt.show()
    
def plot_coifs(items):
    fig, axes = plt.subplots(1, len(items), figsize=(18,4))
    fig.suptitle('coif')
    for i in range(len(items)):
        item = items[i]
        axes[i].plot(item['coif'])
    plt.show()
    