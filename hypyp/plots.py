import matplotlib.pyplot as plt
import numpy as np
import itertools as itertools
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

    return ax

def spectrogram_plot_period(z, times, frequencies, coif, cmap="viridis", norm=Normalize(), ax=None, colorbar=True):
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

    periods = np.log2(5/frequencies)

    ntimes = 5*times
    
    xx,yy = np.meshgrid(ntimes,periods)
    ZZ = z
    
    if norm is None:
        im = ax.pcolor(xx,yy,ZZ, cmap=cmap)
        ax.plot(ntimes,coif)
        ax.fill_between(times,coif, step="mid", alpha=0.4)
    else:
        im = ax.pcolor(xx,yy,ZZ, norm=norm, cmap=cmap)
        ax.plot(ntimes,coif)
        ax.fill_between(times,coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    ax.set_xlim(ntimes.min(), ntimes.max())
    ax.set_ylim(periods.min(), periods.max())

    steps = np.arange(0, len(periods), 10)
    ax.set_yticks(np.round(periods[steps], 2), np.round(2**(periods[steps]), 2))
    
    ax.invert_yaxis()
    
    return ax

    
    


def plot_line(items, key, title):
    fig, axes = plt.subplots(1, len(items), figsize=(18,4))
    fig.suptitle(title)
    for i in range(len(items)):
        item = items[i]
        try:
            print(f"{title}: {item[key].shape}")
            axes[i].plot(item[key])
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

def plot_im(tracers, key, title):
    fig, axes = plt.subplots(2, len(tracers), figsize=(18,4))
    fig.suptitle(title)
    # real
    for i in range(len(tracers)):
        tracer = tracers[i]
        xx, yy = np.meshgrid(np.arange(0, tracer[key].shape[1]), np.arange(0, tracer[key].shape[0]))
        ZZ = np.real(tracer[key])
        im = axes[0, i].pcolor(xx, yy, ZZ)
        fig.colorbar(im, ax=axes[0, i])
    # imag
    for i in range(len(tracers)):
        tracer = tracers[i]
        xx, yy = np.meshgrid(np.arange(0, tracer[key].shape[1]), np.arange(0, tracer[key].shape[0]))
        ZZ = np.imag(tracer[key])
        im = axes[1, i].pcolor(xx, yy, ZZ)
        fig.colorbar(im, ax=axes[1, i])

    
def plot_S12(tracers):
    fig, axes = plt.subplots(1, len(tracers), figsize=(18,4))
    fig.suptitle('S12 real')
    for i in range(len(tracers)):
        tracer = tracers[i]
        xx, yy = np.meshgrid(np.arange(0, tracer['S12'].shape[1]), np.arange(0, tracer['S12'].shape[0]))
        ZZ = np.real(tracer['S12'])
        axes[i].pcolor(xx, yy, ZZ)
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
    

def plot_spectrogram_periods(items):
    fig, axes = plt.subplots(1, len(items), figsize=(18,4))
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
    