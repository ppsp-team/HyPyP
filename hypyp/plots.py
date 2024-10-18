import matplotlib.pyplot as plt
import numpy as np
import itertools as itertools
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.measure import block_reduce


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
    wct,
    times,
    frequencies,
    coif,
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
        fig = plt.gcf()
    
    times_orig = times
    if downsample:
        factor = len(times) // 1000 + 1
        print(f"Downscaling for display by a factor of {factor}")
        times = block_reduce(times, block_size=factor, func=np.mean, cval=np.max(times))
        wct = block_reduce(wct, block_size=(1,factor), func=np.mean, cval=np.mean(wct))
        coif = block_reduce(coif, block_size=factor, func=np.mean, cval=np.mean(coif))
    
    xx, yy = np.meshgrid(times, frequencies)
    
    im = ax.pcolor(xx, yy, wct, norm=Normalize())
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # Cone of influence
    if show_coif:
        ax.plot(times, coif, color=color_shaded)
        ax.fill_between(times,coif, step="mid", color=color_shaded, alpha=0.4)

    # Nyquist frequency
    if show_nyquist:
        y_nyquist = np.ones((len(times),)) * 1 / np.diff(times_orig).mean() / 2
        y_top = np.ones((len(times),)) * frequencies[0]
        ax.plot(times, y_nyquist, color=color_shaded)
        ax.fill_between(times, y_nyquist, y_top, step="mid", color=color_shaded, alpha=0.4)
    
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    if title is not None:
        fig.suptitle(title)
        
    return ax

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
        print(f"shape of wct: {item['wct'].shape}")
        print(item['wct'][55:77,:].mean())
        spectrogram_plot_period(
            np.abs(item['wct']),
            item['times'],
            item['freq'],
            item['coif'],
            ax=axes[i],
            colorbar=False,
            norm=None
        )
        axes[i].title.set_text(tracers[i]['name'])
    
    
    


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
    
