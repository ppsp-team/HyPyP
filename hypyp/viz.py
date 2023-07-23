#!/usr/bin/env python
# coding=utf-8

"""
Basic visualization functions

| Option | Description |
| ------ | ----------- |
| title           | viz.py |
| authors    | Guillaume Dumas, Amir Djalovski, Anaël Ayrolles, Florence Brun |
| date            | 2020-03-18 |
"""

from pathlib import Path
from copy import copy
from typing import Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import meshio
import importlib_resources
from contextlib import ExitStack
import atexit
import math

from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.mesh import Mesh
from hypyp.ext.mpl3d.camera import Camera
from hypyp.analyses import xwt


def transform(locs: np.ndarray,traX: float=0.15, traY: float=0, traZ: float=0.5, rotY: float=(np.pi)/2, rotZ: float=(np.pi)/2) -> np.ndarray:
    """
    Calculates new locations for the EEG locations.

    Arguments:
        locs: array of shape (n_sensors, 3)
            3d coordinates of the sensors
        traX: float
            X translation to apply to the sensors
        traY: float
            Y translation to apply to the sensors
        traZ: float
            Z translation to apply to the sensors
        rotY: float
            Y rotation to apply to the sensors
        rotZ: float
            Z rotation to apply to the sensors

    Returns:
        result: array (n_sensors, 3)
            new 3d coordinates of the sensors
    """
    # Z rotation
    newX = locs[:, 0] * np.cos(rotZ) - locs[:, 1] * np.sin(rotZ)
    newY = locs[:, 0] * np.sin(rotZ) + locs[:, 1] * np.cos(rotZ)
    locs[:, 0] = newX
    locs[:, 0] = locs[:, 0] + traX
    locs[:, 1] = newY
    locs[:, 1] = locs[:, 1] + traY

    # Reduce the size of the eeg headsets
    newZ = locs[:, 0] * np.cos(rotZ) + locs[:, 1] * np.cos(rotZ) + locs[:, 2] * np.cos(rotZ/2)
    locs[:, 2] = newZ
    locs[:, 2] = locs[:, 2]

    # Y rotation
    newX = locs[:, 0] * np.cos(rotY) + locs[:, 2] * np.sin(rotY)
    newZ = - locs[:, 0] * np.sin(rotY) + locs[:, 2] * np.cos(rotY)
    locs[:, 0] = newX
    locs[:, 2] = newZ
    locs[:, 2] = locs[:, 2] + traZ
    locs[:, 0] = locs[:, 0]

    return locs


def plot_sensors_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False):
    """
    Plots sensors in 2D with x representation for bad sensors.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        lab: option to plot channel names
            True by default.

    Returns:
        None: plot the sensors in 2D within the current axis.
    """

    # extract sensor info and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
    lab1 = [ch for ch in epo1.ch_names]

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = []
    bads_epo1 = epo1.info['bads']
    bads_epo2 = []
    bads_epo2 = epo2.info['bads']
    
    # plot sensors ('x' for bads)
    for ch in epo1.ch_names:
      if ch in bads_epo1:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        plt.plot(x1, y1, marker='x', color='dimgrey')
        if lab:
          plt.text(x1+0.012, y1+0.012, lab1[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')
      else:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        plt.plot(x1, y1, marker='o', color='dimgrey')
        if lab:
          plt.text(x1+0.012, y1+0.012, lab1[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')

    
    for ch in epo2.ch_names:
      if ch in bads_epo2:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        plt.plot(x2, y2, marker='x', color='dimgrey')
        if lab:
          plt.text(x2+0.012, y2+0.012, lab2[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')
      else:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        plt.plot(x2, y2, marker='o', color='dimgrey')
        if lab:
          plt.text(x2+0.012, y2+0.012, lab2[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')

def plot_links_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: str='auto', steps: int=10):
    """
    Plots hyper-connectivity in 2D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        C: array, (len(loc1), len(loc2))
            matrix with the values of hyper-connectivity
        threshold: float | str
            threshold for the inter-brain links;
            only those above the set value will be plotted
            Can also be "auto" to use a threshold automatically
            calculated from your matrix as the maximum median 
            by column + the maximum standard error by column.
            Note that the automatic threshold is specific to a 
            dyad and does not allow to compare different dyads.
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines

    Returns:
        None: plot the links in 2D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)

    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)

    # Calculate automatic threshold
    if threshold == 'auto':
      threshold = np.max(np.median(C, 0))+np.max(np.std(C, 0))
    else:
      threshold = threshold

    # define colormap
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

    # plot links
    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            if C[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    plt.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             '-', color=color_p, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        plt.plot([xn, xnn], [yn, ynn],
                                 '-', color=color_p, linewidth=weight)
            if C[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((-C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    plt.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             '-', color=color_n, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((-C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        plt.plot([xn, xnn], [yn, ynn],
                                 '-', color=color_n, linewidth=weight)


def plot_sensors_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False):
    """
    Plots sensors in 3D with x representation for bad sensors.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        lab: option to plot channel names
            False by default.

    Returns:
        None: plot the sensors in 3D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
    lab1 = [ch for ch in epo1.ch_names]

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 =[]
    bads_epo1 = epo1.info['bads']
    bads_epo2 =[]
    bads_epo2 = epo2.info['bads']

    # plot sensors ('x' for bads)
    for ch in epo1.ch_names:
      if ch in bads_epo1:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        ax.scatter(x1, y1, z1, marker='x', color='dimgrey')
        if lab:
                ax.text(x1+0.012, y1+0.012 ,z1, lab1[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')
      else:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        ax.scatter(x1, y1, z1, marker='o', color='dimgrey')
        if lab:
                ax.text(x1+0.012, y1+0.012 ,z1, lab1[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')

    for ch in epo2.ch_names:
      if ch in bads_epo2:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        ax.scatter(x2, y2, z2, marker='x', color='dimgrey')
        if lab:
                ax.text(x2+0.012, y2+0.012 ,z2, lab2[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')
      else:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        ax.scatter(x2, y2, z2, marker='o', color='dimgrey')
        if lab:
                ax.text(x2+0.012, y2+0.012 ,z2, lab2[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')

def plot_links_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: str='auto', steps: int=10):
    """
    Plots hyper-connectivity in 3D.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C: array, (len(loc1), len(loc2))
            matrix with the values of hyper-connectivity
        threshold: float | str
            threshold for the inter-brain links;
            only those above the set value will be plotted
            Can also be "auto" to use a threshold automatically
            calculated from your matrix as the maximum median 
            by column + the maximum standard error by column.
            Note that the automatic threshold is specific to a 
            dyad and does not allow to compare different dyads.
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines

    Returns:
        None: plot the links in 3D within the current axis.
            Plot hyper-connectivity in 3D.
    """
    
    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
  

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)
   

    ctr1 = np.nanmean(loc1, 0)
    ctr1[2] -= 0.2
    ctr2 = np.nanmean(loc2, 0)
    ctr2[2] -= 0.2

    # Calculate automatic threshold
    if threshold == 'auto':
      threshold = np.max(np.median(C, 0))+np.max(np.std(C, 0))
    else:
      threshold = threshold

    # define colormap
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

    # plot links
    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        z1 = loc1[e1, 2]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            z2 = loc2[e2, 2]
            if C[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    ax.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             [loc1[e1, 2], loc2[e2, 2]],
                             '-', color=color_p, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr1[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr2[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr1[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr2[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_p, linewidth=weight)
            if C[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((-C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    ax.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             [loc1[e1, 2], loc2[e2, 2]],
                             '-', color=color_n, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((-C[e1, e2]-threshold)/(np.nanmax(C[:]-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr1[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr2[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr1[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr2[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_n, linewidth=weight)


def plot_significant_sensors(T_obs_plot: np.ndarray, epochs: mne.Epochs):
    """
    Plots the significant sensors from a statistical test (simple t test or
    clusters corrected t test), computed between groups or conditions on power
    or connectivity values, across simple participants. For statistics with
    inter-brain connectivity values on participant pairs (merge data), use the
    plot_links_3d function.

    Arguments:
        T_obs_plot: statistical values to plot, from sensors above alpha threshold,
            array of shape (n_tests,).
        epochs: one participant Epochs object to sample channel information in info.

    Returns:
        None: plot topomap with the T or F statistics for significant sensors.
    """

    # getting sensors position
    pos = np.array([[0, 0]])
    for i in range(0, len(epochs.info['ch_names'])):
        cor = np.array([epochs.info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis=0)
    pos = pos[1:]
    # topoplot of significant sensors
    if np.max(np.abs(T_obs_plot)) != 0:
        vmax = np.max(np.abs(T_obs_plot))
        vmin = -vmax
    else:
        vmax = None
        vmin = None
        mne.viz.plot_topomap(T_obs_plot, pos, vlim=(vmin, vmax), sensors=True)

    return None


def plot_2d_topomap_inter(ax):
    """
    Plot 2D head topomap for hyper-connectivity
    
    Arguments:
        ax : Matplotlib axis

    Returns:
        None : plot the 2D topomap within the current axis.
    """

    # plot first Head 
    N = 300             # number of points for interpolation
    xy_center = [-0.178,0]   # center of the plot
    radius = 0.1         # radius

    # draw a circle
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "w")
    ax.add_patch(circle)
    
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [-0.19,0.095], width = 0.05, height = 0.025, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [-0.19,-0.095], width = 0.05, height = 0.025, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    ## add a nose
    xy = [[-0.087,-0.027],[-0.087,0.027], [-0.068,0]]
    polygon = matplotlib.patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
    

    # Plot second Head 
    x2y2_center = [0.178,0]   # center of the plot
    radius2 = 0.1         # radius
    
    # draw a circle
    circle = matplotlib.patches.Circle(xy = x2y2_center, radius = radius2, edgecolor = "k", facecolor = "w")
    ax.add_patch(circle)
    
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    ## add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [0.19,0.095], width = 0.05, height = 0.025, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [0.19,-0.095], width = 0.05, height = 0.025, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    ## add a nose
    x2y2 = [[0.087,-0.027],[0.087,0.027], [0.068,0]]
    polygon = matplotlib.patches.Polygon(xy = x2y2, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
    

   
  
def get_3d_heads_inter():
    """
    Returns Vertices and Faces of a 3D OBJ representing two facing heads.
    """
   
    # Extract vertices and faces for the first head
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('hypyp') / 'data/Basehead.obj'
    filename = file_manager.enter_context(importlib_resources.as_file(ref))

    mesh = meshio.read(Path(filename).resolve())
    zoom = 0.064
    interval = 0.32

    head1_v = mesh.points*zoom
    head1_f = mesh.cells[0].data

    # Copy the first head to create a second head
    head2_v = copy(mesh.points*zoom)
    # Move the vertices by Y rotation and Z translation
    rotY = np.pi
    newX = head2_v[:, 0] * np.cos(rotY) - head2_v[:, 2] * np.sin(rotY)
    newZ = head2_v[:, 0] * np.sin(rotY) + head2_v[:, 2] * np.cos(rotY)
    head2_v[:, 0] = newX
    head2_v[:, 2] = newZ

    head1_v[:, 2] = head1_v[:, 2] - interval/2
    head2_v[:, 2] = head2_v[:, 2] + interval/2

    # Use the same faces
    head2_f = copy(mesh.cells[0].data)

    # Concatenate the vertices
    vertices = np.concatenate((head1_v, head2_v))
    # Concatenate the faces, shift vertices indexes for second head
    faces = np.concatenate((head1_f, head2_f + len(head1_v)))
    return vertices, faces

def plot_3d_heads(ax, vertices, faces):
    """
    Plot heads models in 3D.

    Arguments:
        ax : Matplotlib axis created with projection='3d'
        vertices : arrays of shape (V, 3)
            3d coordinates of the vertices
        faces : arrays of shape (F, 4)
            vertices number of face

    Returns:
        None : plot the head faces in 3D within the current axis.
    """
    # extract vertices coordinates
    x_V = vertices[:, 2]
    y_V = vertices[:, 0]
    z_V = vertices[:, 1]

    # plot link between vertices
    for F in range(len(faces)):
        V0 = faces[F, 0]
        V1 = faces[F, 1]
        V2 = faces[F, 2]
        V3 = faces[F, 3]
        ax.plot([x_V[V0], x_V[V1]],
                [y_V[V0], y_V[V1]],
                [z_V[V0], z_V[V1]],
                '-', color= 'grey', linewidth=0.3)
        ax.plot([x_V[V1], x_V[V2]],
                [y_V[V1], y_V[V2]],
                [z_V[V1], z_V[V2]],
                '-', color= 'grey', linewidth=0.3)
        ax.plot([x_V[V2], x_V[V3]],
                [y_V[V2], y_V[V3]],
                [z_V[V2], z_V[V3]],
                '-', color= 'grey', linewidth=0.3)
        ax.plot([x_V[V3], x_V[V1]],
                [y_V[V3], y_V[V1]],
                [z_V[V3], z_V[V1]],
                '-', color= 'grey', linewidth=0.3)


def viz_2D_topomap_inter (epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = False):
    """
    Visualization of inter-brain connectivity in 2D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C: array, (len(loc1), len(loc2))
            matrix with the values of hyper-connectivity
        threshold: float
            threshold for the inter-brain links;
            only those above the set value will be plotted
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines
        lab: option to plot channel names
            False by default.

    Returns:
        Plot head topomap with sensors and 
            connectivity links in 2D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 1)
    ax.axis("off")
    plot_2d_topomap_inter(ax)
    plot_sensors_2d_inter(epo1, epo2, lab = lab) # bads are represented as squares
    # plotting links according to sign (red for positive values,
    # blue for negative) and value (line thickness increases
    # with the strength of connectivity)
    plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps)
    plt.tight_layout()
    plt.show()

    return (ax)


def viz_2D_headmodel_inter (epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = True):
    """
    Visualization of inter-brain connectivity in 2D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C: array, (len(loc1), len(loc2))
            matrix with the values of hyper-connectivity
        threshold: float
            threshold for the inter-brain links;
            only those above the set value will be plotted
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines
        lab: option to plot channel names
            True by default.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 2D.
        ax: The new Axes object.
    """

    # Visualization of inter-brain connectivity in 2D
    # defining head model and adding sensors
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    vertices, faces = get_3d_heads_inter()
    camera = Camera("ortho", theta=90, phi=180, scale=1)
    mesh = Mesh(ax, camera.transform @ glm.yrotate(90), vertices, faces,
                facecolors='white',  edgecolors='black', linewidths=.25)
    camera.connect(ax, mesh.update)
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    plot_sensors_2d_inter(epo1, epo2, lab=lab)  # bads are represented as squares
    # plotting links according to sign (red for positive values,
    # blue for negative) and value (line thickness increases
    # with the strength of connectivity)
    plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps)
    plt.tight_layout()
    plt.show()
    
    return (ax)
    

def viz_3D_inter (epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = False):
    """
    Visualization of inter-brain connectivity in 3D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C: array, (len(loc1), len(loc2))
            matrix with the values of hyper-connectivity
        threshold: float
            threshold for the inter-brain links;
            only those above the set value will be plotted
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines
        lab: option to plot channel names
            False by default.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 3D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    vertices, faces = get_3d_heads_inter()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis("off")
    plot_3d_heads(ax, vertices, faces)
    plot_sensors_3d_inter(ax, epo1, epo2, lab=lab) # bads are represented as squares
    # plotting links according to sign (red for positive values,
    # blue for negative) and value (line thickness increases
    # with the strength of connectivity)
    plot_links_3d_inter(ax, epo1, epo2, C=C, threshold=threshold, steps=steps)
    plt.tight_layout()
    plt.show()

    return (ax)


def transform_2d_intra(locs: np.ndarray,traX: float=0.15, traY: float=0, traZ:float=0, rotZ: float=(np.pi)/2) -> np.ndarray:
    """
    Calculates new locations for the EEG locations.

    Arguments:
        locs: array of shape (n_sensors, 3)
            3d coordinates of the sensors
        traX: float
            X translation to apply to the sensors
        traY: float
            Y translation to apply to the sensors
        traZ: float
            Z translation to apply to the sensors
        rotZ: float
            Z rotation to apply to the sensors

    Returns:
        result: array (n_sensors, 3)
            new coordinates of the sensors
    """
    # translation
    locs[:, 0] = locs[:, 0] + traX
    locs[:, 1] = locs[:, 1] + traY
    locs[:, 2] = locs[:, 2] + traZ

    # Reduce the size of the eeg headsets
    newZ = locs[:, 0] * np.cos(rotZ) + locs[:, 1] * np.cos(rotZ) + locs[:, 2] * np.cos(rotZ/2)
    locs[:, 2] = newZ
    locs[:, 2] = locs[:, 2]

    return locs


def plot_2d_topomap_intra(ax):
    """
    Plot 2D head topomap for intra-brain visualisation
    
    Arguments:
        ax : Matplotlib axis

    Returns:
        None : plot the 2D topomap within the current axis.
    """

    # plot first Head 
    N = 300             # number of points for interpolation
    xy_center = [-0.178,0]   # center of the plot
    radius = 0.1         # radius

    # draw a circle
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "w")
    ax.add_patch(circle)
    
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [-0.083,-0.012], width = 0.025, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [-0.273,-0.012], width = 0.025, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    ## add a nose
    xy = [[-0.151,0.091],[-0.205,0.091], [-0.178,0.11]]
    polygon = matplotlib.patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
    

    # Plot second Head 
    x2y2_center = [0.178,0]   # center of the plot
    radius2 = 0.1         # radius
    
    # draw a circle
    circle = matplotlib.patches.Circle(xy = x2y2_center, radius = radius2, edgecolor = "k", facecolor = "w")
    ax.add_patch(circle)
    
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    ## add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [0.083,-0.012], width = 0.025, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [0.273,-0.012], width = 0.025, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    ## add a nose
    x2y2 = [[0.151,0.091],[0.205,0.091], [0.178,0.11]]
    polygon = matplotlib.patches.Polygon(xy = x2y2, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
    

def plot_sensors_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False):
    """
    Plots sensors in 2D with x representation for bad sensors.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        lab: option to plot channel names
            True by default.

    Returns:
        None: plot the sensors in 2D within the current axis.
    """

    # extract sensor info and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))
    lab1 = [ch for ch in epo1.ch_names]

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = []
    bads_epo1 = epo1.info['bads']
    bads_epo2 = []
    bads_epo2 = epo2.info['bads']

    # plot sensors
    for ch in epo1.ch_names:
      if ch in bads_epo1:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        plt.plot(x1, y1, marker='x', color='dimgrey')
        if lab:
          plt.text(x1+0.012, y1+0.012, lab1[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')
      else:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        plt.plot(x1, y1, marker='o', color='dimgrey')
        if lab:
          plt.text(x1+0.012, y1+0.012, lab1[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')

    
    for ch in epo2.ch_names:
      if ch in bads_epo2:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        plt.plot(x2, y2, marker='x', color='dimgrey')
        if lab:
          plt.text(x2+0.012, y2+0.012, lab2[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')
      else:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        plt.plot(x2, y2, marker='o', color='dimgrey')
        if lab:
          plt.text(x2+0.012, y2+0.012, lab2[index_ch],
                   horizontalalignment='center',
                   verticalalignment='center')
  
    
def plot_links_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs,
                        C1: np.ndarray, C2: np.ndarray,
                        threshold: str='auto', steps: int=2):
    """
    Plots hyper-connectivity in 2D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        C1: array, (len(loc1), len(loc1))
            matrix with the values of intra-brain connectivity
        C2: array, (len(loc2), len(loc2))
            matrix with the values of intra-brain connectivity
        threshold: float | str
            threshold for the inter-brain links;
            only those above the set value will be plotted
            Can also be "auto" to use a threshold automatically
            calculated from your matrix as the maximum median 
            by column + the maximum standard error by column.
            Note that the automatic threshold is specific to a 
            dyad and does not allow to compare different dyads.
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines

    Returns:
        None: plot the links in 2D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))

    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)
    
    # Calculate vmin and vmax for colormap as min and max [C1, C2]
    Cmax1=np.nanmax(C1[:])
    Cmax2=np.nanmax(C2[:])
    Cmax=[]
    Cmax=[Cmax1, Cmax2]
    vmax=np.nanmax(Cmax)
    Cmin1=np.nanmin(C1[:])
    Cmin2=np.nanmin(C2[:])
    Cmin=[]
    Cmin=[Cmin1, Cmin2]
    vmin=np.min(Cmin)

    # Calculate automatic threshold
    if threshold == 'auto':
      threshold = np.max([np.median(C1, 0),np.median(C2,0)])+np.max([np.std(C1, 0),np.std(C2, 0)])
    else:
      threshold = threshold
  

    # Define colormap for both participant
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=vmin, vmax=-threshold)

    # plot links
    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        for e2 in range(len(loc1)):
            x2 = loc1[e2, 0]
            y2 = loc1[e2, 1]
            if C1[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C1[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    plt.plot([loc1[e1, 0], loc1[e2, 0]],
                             [loc1[e1, 1], loc1[e2, 1]],
                             '-', color=color_p, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr1[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr1[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr1[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr1[1]) +
                               b**3 * y2)
                        plt.plot([xn, xnn], [yn, ynn],
                                 '-', color=color_p, linewidth=weight)
            if C1[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C1[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((-C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    plt.plot([loc1[e1, 0], loc1[e2, 0]],
                             [loc1[e1, 1], loc1[e2, 1]],
                             '-', color=color_n, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((-C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr1[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr1[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr1[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr1[1]) +
                               b**3 * y2)
                        plt.plot([xn, xnn], [yn, ynn],
                                 '-', color=color_n, linewidth=weight)

    for e1 in range(len(loc2)):
      x1 = loc2[e1, 0]
      y1 = loc2[e1, 1]
      for e2 in range(len(loc2)):
          x2 = loc2[e2, 0]
          y2 = loc2[e2, 1]
          if C2[e1, e2] >= threshold:
              color_p = cmap_p(norm_p(C2[e1, e2]))
              if steps <= 2:
                  weight = 0.2 +1.6*((C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                  plt.plot([loc2[e1, 0], loc2[e2, 0]],
                           [loc2[e1, 1], loc2[e2, 1]],
                           '-', color=color_p, linewidth=weight)
              else:
                  alphas = np.linspace(0, 1, steps)
                  weight = 0.2 +1.6*((C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                  for idx in range(len(alphas)-1):
                      a = alphas[idx]
                      b = alphas[idx+1]
                      xn = ((1-a)**3 * x1 +
                            3 * (1-a)**2 * a * (2 * x1 - ctr2[0]) +
                            3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                            a**3 * x2)
                      xnn = ((1-b)**3 * x1 +
                             3 * (1-b)**2 * b * (2 * x1 - ctr2[0]) +
                             3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                             b**3 * x2)
                      yn = ((1-a)**3 * y1 +
                            3 * (1-a)**2 * a * (2 * y1 - ctr2[1]) +
                            3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                            a**3 * y2)
                      ynn = ((1-b)**3 * y1 +
                             3 * (1-b)**2 * b * (2 * y1 - ctr2[1]) +
                             3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                             b**3 * y2)
                      plt.plot([xn, xnn], [yn, ynn],
                               '-', color=color_p, linewidth=weight)
          if C2[e1, e2] <= -threshold:
              color_n = cmap_n(norm_n(C2[e1, e2]))
              if steps <= 2:
                  weight = 0.2 +1.6*((-C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                  plt.plot([loc2[e1, 0], loc2[e2, 0]],
                           [loc2[e1, 1], loc2[e2, 1]],
                           '-', color=color_n, linewidth=weight)
              else:
                  alphas = np.linspace(0, 1, steps)
                  weight = 0.2 +1.6*((-C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                  for idx in range(len(alphas)-1):
                      a = alphas[idx]
                      b = alphas[idx+1]
                      xn = ((1-a)**3 * x1 +
                            3 * (1-a)**2 * a * (2 * x1 - ctr2[0]) +
                            3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                            a**3 * x2)
                      xnn = ((1-b)**3 * x1 +
                             3 * (1-b)**2 * b * (2 * x1 - ctr2[0]) +
                             3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                             b**3 * x2)
                      yn = ((1-a)**3 * y1 +
                            3 * (1-a)**2 * a * (2 * y1 - ctr2[1]) +
                            3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                            a**3 * y2)
                      ynn = ((1-b)**3 * y1 +
                             3 * (1-b)**2 * b * (2 * y1 - ctr2[1]) +
                             3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                             b**3 * y2)
                      plt.plot([xn, xnn], [yn, ynn],
                               '-', color=color_n, linewidth=weight)


def viz_2D_topomap_intra (epo1: mne.Epochs, epo2: mne.Epochs,
                          C1: np.ndarray, C2: np.ndarray,
                          threshold: float=0.95, steps: int=2,
                          lab: bool = False):
    """
    Visualization of inter-brain connectivity in 3D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C1: array, (len(loc1), len(loc1))
            matrix with the values of intra-brain connectivity
        C2: array, (len(loc2), len(loc2))
            matrix with the values of intra-brain connectivity
        threshold: float
            threshold for the inter-brain links;
            only those above the set value will be plotted
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines
        lab: option to plot channel names
            False by default.

    Returns:
        Plot head topomap with sensors and 
            intra-brain connectivity links in 2D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 1)
    ax.axis("off")
    plot_2d_topomap_intra(ax)
    # bads are represented as squares
    plot_sensors_2d_intra(epo1, epo2, lab = lab)
    # plotting links according to sign (red for positive values,
    # blue for negative) and value (line thickness increases
    # with the strength of connectivity)
    plot_links_2d_intra(epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps)
    plt.tight_layout()
    plt.show()

    return (ax)

   
  
def get_3d_heads_intra():
    """
    Returns Vertices and Faces of a 3D OBJ representing two facing heads.
    """
   
    # Extract vertices and faces for the first head
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('hypyp') / 'data/Basehead.obj'
    filename = file_manager.enter_context(importlib_resources.as_file(ref))

    mesh = meshio.read(Path(filename).resolve())
    zoom = 0.064
    interval = 0.5

    head1_v = mesh.points*zoom
    head1_f = mesh.cells[0].data

    # Copy the first head to create a second head
    head2_v = copy(mesh.points*zoom)
    head2_v[:, 0] = head2_v[:, 0] + interval

    # Use the same faces
    head2_f = copy(mesh.cells[0].data)

    # Concatenate the vertices
    vertices = np.concatenate((head1_v, head2_v))
    # Concatenate the faces, shift vertices indexes for second head
    faces = np.concatenate((head1_f, head2_f + len(head1_v)))
    return vertices, faces



def plot_sensors_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False):
    """
    Plots sensors in 3D with x representation for bad sensors.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        lab: option to plot channel names
            False by default.

    Returns:
        None: plot the sensors in 3D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
    lab1 = [ch for ch in epo1.ch_names]

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 =[]
    bads_epo1 = epo1.info['bads']
    bads_epo2 =[]
    bads_epo2 = epo2.info['bads']

    # plot sensors
    for ch in epo1.ch_names:
      if ch in bads_epo1:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        ax.scatter(x1, y1, z1, marker='x', color='dimgrey')
        if lab:
                ax.text(x1+0.012, y1+0.012 ,z1, lab1[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')
      else:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        ax.scatter(x1, y1, z1, marker='o', color='dimgrey')
        if lab:
                ax.text(x1+0.012, y1+0.012 ,z1, lab1[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')

    for ch in epo2.ch_names:
      if ch in bads_epo2:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        ax.scatter(x2, y2, z2, marker='x', color='dimgrey')
        if lab:
                ax.text(x2+0.012, y2+0.012 ,z2, lab2[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')
      else:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        ax.scatter(x2, y2, z2, marker='o', color='dimgrey')
        if lab:
                ax.text(x2+0.012, y2+0.012 ,z2, lab2[index_ch],
                        horizontalalignment='center',
                        verticalalignment='center')


def plot_links_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs,
                        C1: np.ndarray, C2: np.ndarray, threshold: str='auto',
                        steps: int=10):
    """
    Plots hyper-connectivity in 3D.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        epo1: arrays of shape (n_sensors, 3)
            Epochs object to get channel information
        epo2: arrays of shape (n_sensors, 3)
            Epochs object to get channel information
        C1: array, (len(loc1), len(loc1))
            matrix with the values of intra-brain connectivity
        C2: array, (len(loc1), len(loc2))
            matrix with the values of intra-brain connectivity
        threshold: float | str
            threshold for the inter-brain links;
            only those above the set value will be plotted
            Can also be "auto" to use a threshold automatically
            calculated from your matrix as the maximum median 
            by column + the maximum standard error by column.
            Note that the automatic threshold is specific to a 
            dyad and does not allow to compare different dyads.
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines

    Returns:
        None: plot the links in 3D within the current axis.
          Plot hyper-connectivity in 3D.
    """
        
    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
  

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
   

    ctr1 = np.nanmean(loc1, 0)
    ctr1[2] -= 0.2
    ctr2 = np.nanmean(loc2, 0)
    ctr2[2] -= 0.2

    # Calculate vmin and vmax for colormap as min and max [C1, C2]
    Cmax1=np.nanmax(C1[:])
    Cmax2=np.nanmax(C2[:])
    Cmax=[]
    Cmax=[Cmax1, Cmax2]
    vmax=np.nanmax(Cmax)
    Cmin1=np.nanmin(C1[:])
    Cmin2=np.nanmin(C2[:])
    Cmin=[]
    Cmin=[Cmin1, Cmin2]
    vmin=np.min(Cmin)

    # Calculate automatic threshold
    if threshold == 'auto':
      threshold = np.max([np.median(C1, 0),np.median(C2,0)])+np.max([np.std(C1, 0),np.std(C2, 0)])
    else:
      threshold = threshold

    # Define colormap for both participant
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=vmin, vmax=-threshold)

    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        z1 = loc1[e1, 2]
        for e2 in range(len(loc1)):
            x2 = loc1[e2, 0]
            y2 = loc1[e2, 1]
            z2 = loc1[e2, 2]
            if C1[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C1[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    ax.plot([loc1[e1, 0], loc1[e2, 0]],
                             [loc1[e1, 1], loc1[e2, 1]],
                             [loc1[e1, 2], loc1[e2, 2]],
                             '-', color=color_p, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr1[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr1[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr1[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr1[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr1[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr1[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr1[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr1[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_p, linewidth=weight)
            if C1[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C1[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((-C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    ax.plot([loc1[e1, 0], loc1[e2, 0]],
                             [loc1[e1, 1], loc1[e2, 1]],
                             [loc1[e1, 2], loc1[e2, 2]],
                             '-', color=color_n, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((-C1[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr1[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr1[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr1[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr1[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr1[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr1[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr1[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr1[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr1[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr1[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_n, linewidth=weight)
  
    for e1 in range(len(loc2)):
        x1 = loc2[e1, 0]
        y1 = loc2[e1, 1]
        z1 = loc2[e1, 2]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            z2 = loc2[e2, 2]
            if C2[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C2[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    ax.plot([loc2[e1, 0], loc2[e2, 0]],
                             [loc2[e1, 1], loc2[e2, 1]],
                             [loc2[e1, 2], loc2[e2, 2]],
                             '-', color=color_p, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr2[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr2[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr2[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr2[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr2[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr2[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr2[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr2[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_p, linewidth=weight)
            if C2[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C2[e1, e2]))
                if steps <= 2:
                    weight = 0.2 +1.6*((-C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    ax.plot([loc2[e1, 0], loc2[e2, 0]],
                             [loc2[e1, 1], loc2[e2, 1]],
                             [loc2[e1, 2], loc2[e2, 2]],
                             '-', color=color_n, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((-C2[e1, e2]-threshold)/(np.nanmax(vmax-threshold)))
                    for idx in range(len(alphas)-1):
                        a = alphas[idx]
                        b = alphas[idx+1]
                        xn = ((1-a)**3 * x1 +
                              3 * (1-a)**2 * a * (2 * x1 - ctr2[0]) +
                              3 * (1-a) * a**2 * (2 * x2 - ctr2[0]) +
                              a**3 * x2)
                        xnn = ((1-b)**3 * x1 +
                               3 * (1-b)**2 * b * (2 * x1 - ctr2[0]) +
                               3 * (1-b) * b**2 * (2 * x2 - ctr2[0]) +
                               b**3 * x2)
                        yn = ((1-a)**3 * y1 +
                              3 * (1-a)**2 * a * (2 * y1 - ctr2[1]) +
                              3 * (1-a) * a**2 * (2 * y2 - ctr2[1]) +
                              a**3 * y2)
                        ynn = ((1-b)**3 * y1 +
                               3 * (1-b)**2 * b * (2 * y1 - ctr2[1]) +
                               3 * (1-b) * b**2 * (2 * y2 - ctr2[1]) +
                               b**3 * y2)
                        zn = ((1-a)**3 * z1 +
                              3 * (1-a)**2 * a * (2 * z1 - ctr2[2]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr2[2]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr2[2]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr2[2]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color=color_n, linewidth=weight)

                  

def viz_3D_intra (epo1: mne.Epochs, epo2: mne.Epochs,
                  C1: np.ndarray, C2: np.ndarray,
                  threshold: float=0.95, steps: int=10,
                  lab: bool = False):
    """
    Visualization of intra-brain connectivity in 3D.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channel information
        epo2: mne.Epochs
            Epochs object to get channel information
        C1: array, (len(loc1), len(loc1))
            matrix with the values of intra-brain connectivity
        C2: array, (len(loc2), len(loc2))
        threshold: float
            threshold for the inter-brain links;
            only those above the set value will be plotted
        steps: int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines
        lab: option to plot channel names
            False by default.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 3D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    vertices, faces = get_3d_heads_intra()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis("off")
    plot_3d_heads(ax, vertices, faces)
    # bads are represented as squares
    plot_sensors_3d_intra(ax, epo1, epo2, lab=lab)
    # plotting links according to sign (red for positive values,
    # blue for negative) and value (line thickness increases
    # with the strength of connectivity)
    plot_links_3d_intra(ax, epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps)
    plt.tight_layout()
    plt.show()

    return (ax)



def plot_xwt(sig1: mne.Epochs, sig2: mne.Epochs,
             sfreq: Union[int, float],
             freqs: Union[int, float, np.ndarray],
             time: int, analysis: str,
             figsize: tuple = (30, 8), tmin: int = 0,
             x_units: Union[int, float] = 100):
    """
    Plots the results of the Cross wavelet analysis.

    Arguments:
        sig1 : mne.Epochs
            Signal (eg. EEG data) of first participant.

        sig2 : mne.Epochs
            Signal (eg. EEG data) of second participant.

        freqs: int | float | np.ndarray
            Frequency range of interest in Hz.

        time: int
            Time of sample duration in seconds.

        analysis: str
            Sets type of analysis.

        figsize: tuple
            Figure size (default is (30, 8)).

        x_units: int | float
            distance between xticks on x-axis (time) (default is 100)

    Note:
        This function is not meant to be called indepedently,
        but is meant to be called when using plot_xwt_crosspower
        or plot_xwt_phase_angle.

    Returns:
        Figure: The figure with the xwt results.
    """

    dt = 1/sfreq
    xmax = time/dt
    xmin = tmin * dt
    tick_units = xmax/x_units
    unit_conv = time/x_units
    xticks = np.arange(xmin, xmax, tick_units)
    x_labels = np.arange(tmin, time, unit_conv)
    xmark = []
    for xlabel in x_labels:
        if xlabel != '':
            mark = str(round(xlabel, 3))
            xmark.append(mark)
        else:
            xmark = xlabel

    coi = []
    for f in freqs:
        dt_f = 1/f
        f_coi_init = math.sqrt(2*2*dt_f)
        f_coi = f_coi_init/dt
        coi.append(f_coi)

    coi_check = []
    for item in coi:
        if item >= (time/dt):
            coi_check.append(False)
        else:
            coi_check.append(True)

    if False in coi_check:
        print('Warning: your epoch length seems too short for the wavelet transform!')
    else:
        print('Epoch length is appropriate for wavelet transform')

    coi_index = np.arange(0, len(freqs))

    rev_coi = []
    for f in freqs:
        dt_f = 1/f
        f_coi = math.sqrt(2*2*dt_f)
        sub_max = (time-f_coi)/dt
        rev_coi.append(sub_max)

    fig = plt.figure(figsize=figsize)
    plt.subplot(122)

    if analysis == 'phase':
        data = xwt(sig1, sig2, sfreq, freqs, analysis='phase')
        analysis_title = 'Cross Wavelet Transform (Phase Angle)'
        cbar_title = 'Phase Difference'
        my_cm = matplotlib.cm.get_cmap('hsv')
        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='nearest')

    elif analysis == 'power':
        data = xwt(sig1, sig2, sfreq, freqs, analysis='power')
        normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        analysis_title = 'Cross Wavelet Transform (Power)'
        cbar_title = 'Cross Power'
        my_cm = matplotlib.cm.get_cmap('viridis')
        plt.imshow(normed_data, aspect='auto', cmap=my_cm,
                   interpolation='lanczos')

    elif analysis == 'wtc':
        data = xwt(sig1, sig2, sfreq, freqs, analysis='wtc')
        analysis_title = 'Wavelet Coherence'
        cbar_title = 'Coherence'
        my_cm = matplotlib.cm.get_cmap('plasma')
        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='lanczos')

    else:
        ValueError('Analysis must be set as phase, power, or wtc.')

    plt.gca().invert_yaxis()
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (s)')
    ylabels = np.linspace((freqs[0]), (freqs[-1]), len(freqs[0:]))
    ymark = []
    for ylabel in ylabels:
        if ylabel != '':
            mark = str(round(ylabel, 3))
            ymark.append(mark)
        else:
            ymark = ylabel

    plt.gca().set_yticks(ticks=np.arange(0, len(ylabels)), labels=ymark)
    plt.gca().set_xticks(ticks=xticks, labels=xmark)
    plt.xlim(tmin, xmax)
    plt.ylim(0, int(len(ylabels[0:-1])))
    plt.plot(coi, coi_index, 'w')
    plt.plot(rev_coi, coi_index, 'w')

    plt.fill_between(coi, coi_index, hatch='X', fc='w', alpha=0.5)
    plt.fill_between(rev_coi, coi_index, hatch='X', fc='w', alpha=0.5)

    plt.axvspan(xmin, min(coi), hatch='X', fc='w', alpha=0.5)
    plt.axvspan(xmax, max(rev_coi), hatch='X', fc='w', alpha=0.5)

    plt.title(analysis_title)
    plt.colorbar(label=cbar_title)

    return fig
