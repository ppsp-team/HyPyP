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

def plot_wavelet_coherence(
    wtc,
    times,
    frequencies,
    coif,
    sig=None,
    ax=None,
    colorbar=True,
    downsample=False,
    show_coif=True,
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
    if downsample:
        factor = len(times) // 500 + 1
        print(f"Downscaling for display by a factor of {factor}")
        times = block_reduce(times, block_size=factor, func=np.mean, cval=np.max(times))
        wtc = block_reduce(wtc, block_size=(1,factor), func=np.mean, cval=np.mean(wtc))
        coif = block_reduce(coif, block_size=factor, func=np.mean, cval=np.mean(coif))
    
    periods = 1 / frequencies
    xx, yy = np.meshgrid(times, periods)
    
    #im = ax.pcolor(xx, yy, wtc, norm=Normalize())
    im = ax.pcolor(xx, yy, wtc, vmin=0, vmax=1)
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Period (seconds)')

    # Cone of influence
    if show_coif:
        # TODO use "coi" instead of doing back and forth between frequency and period values
        ax.plot(times, 1/coif, color=color_shaded)
        ax.fill_between(times, 1/coif, y2=1000, step="mid", color=color_shaded, alpha=0.4)

    ## Nyquist frequency
    #if show_nyquist:
    #    y_nyquist = np.ones((len(times),)) * 1 / np.diff(times_orig).mean() / 2
    #    y_top = np.ones((len(times),)) * periods[0]
    #    ax.plot(times, y_nyquist, color=color_shaded)
    #    ax.fill_between(times, y_nyquist, y_top, step="mid", color=color_shaded, alpha=0.4)
    
    # Define a custom locator and formatter
    def custom_locator(ymin, ymax):
        ticks = []
        ticks.extend(range(math.ceil(ymin), 11))
        ticks.extend(range(12, 21, 2))
        ticks.extend(range(25, int(ymax) + 1, 5))
        return ticks
    
    def custom_formatter(y, _):
        return f"{int(y)}" if y >= 1 else ""

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

    if sig is not None:
        ax.contour(xx, yy, wtc, levels=sig, color='k')
        
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

def plot_coherence_df(
    df,
    s1_label,
    s2_label,
    field1,
    field2,
    ordered_fields,
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    def heatmap_from_pivot(pivot, ax):
        index_order = [i for i in ordered_fields if i in pivot.index]
        column_order = [c for c in ordered_fields if c in pivot.columns]
        pivot_reordered = pivot.reindex(index=index_order, columns=column_order)
        heatmap = sns.heatmap(pivot_reordered, cmap='viridis', vmin=0, vmax=1, cbar=False, ax=ax)
        ax.set_xticks(ticks=range(len(pivot_reordered.columns)))
        ax.set_xticklabels(pivot_reordered.columns, rotation=90, ha='left', fontsize=6 if len(column_order)>15 else 10)
        ax.set_yticks(ticks=range(len(pivot_reordered.index)))
        ax.set_yticklabels(pivot_reordered.index, rotation=0, va='top', fontsize=6 if len(index_order)>15 else 10)
        ax.tick_params(axis='both', which='both', length=0)
        return heatmap

    dyad_selector = (df['subject1']==s1_label) & (df['subject2']==s2_label)
    s1_selector = (df['subject1']==s1_label) & (df['subject2']==s1_label)
    s2_selector = (df['subject1']==s2_label) & (df['subject2']==s2_label)
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

def plot_spectrogram_periods(items, tracers):
    fig, axes = plt.subplots(1, len(items), figsize=(18,6))
    fig.suptitle('spectrogram periods')
    for i in range(len(items)):
        item = items[i]
        print(f"shape of wtc: {item['wtc'].shape}")
        print(item['wtc'][55:77,:].mean())
        spectrogram_plot_period(
            np.abs(item['wtc']),
            item['times'],
            item['freq'],
            item['coif'],
            ax=axes[i],
            colorbar=False,
            norm=None
        )
        axes[i].title.set_text(tracers[i]['name'])
    
    
    
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
    
def plot_times(items):
    fig, axes = plt.subplots(1, len(items), figsize=(18,4))
    fig.suptitle('times')
    for i in range(len(items)):
        item = items[i]
        axes[i].plot(item['times'])
    plt.show()

def plot_coefs(tracers, key, title):
    fig, axes = plt.subplots(1, len(tracers), figsize=(18,10))
    axes = np.atleast_1d(axes)
    fig.suptitle(title)
    for i in range(len(tracers)):
        tracer = tracers[i]
        im = axes[i].pcolormesh(tracer['x1'], tracer['freq'], np.abs(tracer[key]))
        axes[i].set_yscale('log')
        fig.colorbar(im, ax=axes[i])
        axes[i].title.set_text(tracer['name'])

    
def plot_S12(tracers):
    fig, axes = plt.subplots(1, len(tracers), figsize=(18,4))
    fig.suptitle('S12 real')
    for i in range(len(tracers)):
        tracer = tracers[i]
        xx, yy = np.meshgrid(np.arange(0, tracer['S12'].shape[1]), np.arange(0, tracer['S12'].shape[0]))
        ZZ = np.real(tracer['S12'])
        axes[i].pcolor(xx, yy, ZZ)
        axes[i].title.set_text(tracer['name'])
    plt.show()
    
    fig, axes = plt.subplots(1, len(tracers), figsize=(18,4))
    fig.suptitle('S12 imag')
    for i in range(len(tracers)):
        tracer = tracers[i]
        xx, yy = np.meshgrid(np.arange(0, tracer['S12'].shape[1]), np.arange(0, tracer['S12'].shape[0]))
        ZZ = np.imag(tracer['S12'])
        axes[i].pcolor(xx, yy, ZZ)
    plt.show()
    
def plot_im_diff(left, right, title):
    fig, axes = plt.subplots(1, 3, figsize=(18,4))
    fig.suptitle(title)
    xx, yy = np.meshgrid(np.arange(0, left.shape[1]), np.arange(0, left.shape[0]))

    im = axes[0].pcolor(xx, yy, left)
    fig.colorbar(im, ax=axes[0])

    im = axes[1].pcolor(xx, yy, right - left, cmap=cm.get_cmap('Greys'))
    fig.colorbar(im, ax=axes[1])

    im = axes[2].pcolor(xx, yy, right)
    fig.colorbar(im, ax=axes[2])
    
