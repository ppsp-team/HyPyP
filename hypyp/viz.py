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


def transform(locs: np.ndarray, traX: float=0.15, traY: float=0, traZ: float=0.5, rotY: float=(np.pi)/2, rotZ: float=(np.pi)/2, children: bool=False, child_head: bool=False) -> np.ndarray:
    """
    Apply a series of transformations to a set of 3D coordinates.
    
    Parameters:
        locs (np.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates to be transformed.
        traX (float, optional): Translation along the X-axis. Default is 0.15.
        traY (float, optional): Translation along the Y-axis. Default is 0.
        traZ (float, optional): Translation along the Z-axis. Default is 0.5.
        rotY (float, optional): Rotation around the Y-axis in radians. Default is π/2.
        rotZ (float, optional): Rotation around the Z-axis in radians. Default is π/2.
        children (bool, optional): If True, apply transformations specific to children's models. Default is False.
        child_head (bool, optional): If True, apply transformations specific to a child's head model. 
                                    Only relevant if `children` is True. Default is False.
    Returns:
        np.ndarray: The transformed 3D coordinates.
    """
    if children:
        if child_head:
            scale = 0.68
            trans = 0.07
            yfactor = 1.15
        else:
            scale = 1.03
            trans = -0.01
            yfactor = 1.2
    else: 
        scale = 1.03
        trans = -0.01
        yfactor = 1.2
        
    # Z rotation
    newX = locs[:, 0] * np.cos(rotZ) - locs[:, 1] * np.sin(rotZ)
    newY = locs[:, 0] * np.sin(rotZ) + locs[:, 1] * np.cos(rotZ)
    locs[:, 0] = newX
    locs[:, 0] = locs[:, 0] + traX
    locs[:, 1] = newY * scale * yfactor
    locs[:, 1] = locs[:, 1] + traY

    # Reduce the size of the eeg headsets
    newZ = locs[:, 0] * np.cos(rotZ) + locs[:, 1] * np.cos(rotZ) + locs[:, 2] * np.cos(rotZ/2)
    locs[:, 2] = newZ * scale

    # Y rotation
    newX = locs[:, 0] * np.cos(rotY) + locs[:, 2] * np.sin(rotY)
    newZ = - locs[:, 0] * np.sin(rotY) + locs[:, 2] * np.cos(rotY)
    locs[:, 0] = newX * scale - trans
    locs[:, 2] = newZ * scale * yfactor
    locs[:, 2] = locs[:, 2] + traZ

    return locs


def transform_2d_intra(locs: np.ndarray, traX: float=0.15, traY: float=0, traZ: float=0, rotZ: float=(np.pi)/2, children: bool=False, child_head: bool=False) -> np.ndarray:
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        result: array (n_sensors, 3)
            new coordinates of the sensors
    """
    if children:
        if child_head:
            # Transformations for a child's head model
            scale = 0.68
        else:
            # Transformations for an adult's head model
            scale = 1.03
    else:
        scale = 1.0

    # translation
    locs[:, 0] = locs[:, 0] + traX
    locs[:, 1] = locs[:, 1] + traY
    locs[:, 2] = locs[:, 2] + traZ

    # Reduce the size of the eeg headsets
    newZ = locs[:, 0] * np.cos(rotZ) + locs[:, 1] * np.cos(rotZ) + locs[:, 2] * np.cos(rotZ/2)
    locs[:, 2] = newZ * scale

    return locs


def bezier_interpolation(t, p0, p1, c0, c1=None):
    """
    Calculate points on a Bezier curve.
    
    Parameters:
        t: Interpolation parameter (0 to 1)
        p0: Starting point
        p1: Ending point
        c0: Control point for p0
        c1: Control point for p1. If None, uses c0 for both (for intra-brain curves)
    
    Returns:
        Point on the Bezier curve
    """
    if c1 is None:
        c1 = c0
        
    return ((1 - t) ** 3 * p0 +
            3 * (1 - t) ** 2 * t * (2 * p0 - c0) +
            3 * (1 - t) * t ** 2 * (2 * p1 - c1) +
            t ** 3 * p1)


def plot_sensors_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = True, children: bool=False, child_head: bool=False):
    """
    Plots sensors in 2D with x representation for bad sensors.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        lab: option to plot channel names
            True by default.
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.
    Returns:
        None: plot the sensors in 2D within the current axis.
    """
    # extract sensor info and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))

    if children:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2, children=children, child_head=child_head)
    else:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)

    lab1 = [ch for ch in epo1.ch_names]
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = epo1.info['bads']
    bads_epo2 = epo2.info['bads']

    # plot sensors ('x' for bads)
    for ch in epo1.ch_names:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        if ch in bads_epo1:
            plt.plot(x1, y1, marker='x', color='dimgrey')
        else:
            plt.plot(x1, y1, marker='o', color='dimgrey')
        if lab:
            plt.text(x1+0.012, y1+0.012, lab1[index_ch], horizontalalignment='center', verticalalignment='center')

    for ch in epo2.ch_names:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        if ch in bads_epo2:
            plt.plot(x2, y2, marker='x', color='dimgrey')
        else:
            plt.plot(x2, y2, marker='o', color='dimgrey')
        if lab:
            plt.text(x2+0.012, y2+0.012, lab2[index_ch], horizontalalignment='center', verticalalignment='center')


def plot_sensors_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, children: bool=False, child_head: bool=False):
    """
    Plots sensors in 2D with x representation for bad sensors.

    Arguments:
        epo1: mne.Epochs
            Epochs object to get channels information
        epo2: mne.Epochs
            Epochs object to get channels information
        lab: option to plot channel names
            True by default.
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.
    Returns:
        None: plot the sensors in 2D within the current axis.
    """

    # extract sensor info and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))

    if children:
        loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2), children=children, child_head=child_head)
    else:
        loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))
        loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))

    lab1 = [ch for ch in epo1.ch_names]
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = epo1.info['bads']
    bads_epo2 = epo2.info['bads']

    # plot sensors ('x' for bads)
    for ch in epo1.ch_names:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        if ch in bads_epo1:
            plt.plot(x1, y1, marker='x', color='dimgrey')
        else:
            plt.plot(x1, y1, marker='o', color='dimgrey')
        if lab:
            plt.text(x1+0.012, y1+0.012, lab1[index_ch], horizontalalignment='center', verticalalignment='center')

    for ch in epo2.ch_names:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        if ch in bads_epo2:
            plt.plot(x2, y2, marker='x', color='dimgrey')
        else:
            plt.plot(x2, y2, marker='o', color='dimgrey')
        if lab:
            plt.text(x2+0.012, y2+0.012, lab2[index_ch], horizontalalignment='center', verticalalignment='center')


def plot_sensors_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the sensors in 3D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    if children:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2, children=children, child_head=child_head)
    else:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)

    lab1 = [ch for ch in epo1.ch_names]
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = epo1.info['bads']
    bads_epo2 = epo2.info['bads']

    # plot sensors ('x' for bads)
    for ch in epo1.ch_names:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        if ch in bads_epo1:
            ax.scatter(x1, y1, z1, marker='x', color='dimgrey')
        else:
            ax.scatter(x1, y1, z1, marker='o', color='dimgrey')
        if lab:
            ax.text(x1+0.012, y1+0.012, z1, lab1[index_ch], horizontalalignment='center', verticalalignment='center')

    for ch in epo2.ch_names:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        if ch in bads_epo2:
            ax.scatter(x2, y2, z2, marker='x', color='dimgrey')
        else:
            ax.scatter(x2, y2, z2, marker='o', color='dimgrey')
        if lab:
            ax.text(x2+0.012, y2+0.012, z2, lab2[index_ch], horizontalalignment='center', verticalalignment='center')



def plot_sensors_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, children: bool = False, child_head: bool = False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the sensors in 3D within the current axis.
    """

    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))

    if children:
        loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2), children=children, child_head=child_head)
    else:
        loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2))

    lab1 = [ch for ch in epo1.ch_names]
    lab2 = [ch for ch in epo2.ch_names]

    bads_epo1 = epo1.info['bads']
    bads_epo2 = epo2.info['bads']

    # plot sensors('x' for bads)
    for ch in epo1.ch_names:
        index_ch = epo1.ch_names.index(ch)
        x1, y1, z1 = loc1[index_ch, :]
        if ch in bads_epo1:
            ax.scatter(x1, y1, z1, marker='x', color='dimgrey')
        else:
            ax.scatter(x1, y1, z1, marker='o', color='dimgrey')
        if lab:
            ax.text(x1+0.012, y1+0.012, z1, lab1[index_ch], horizontalalignment='center', verticalalignment='center')

    for ch in epo2.ch_names:
        index_ch = epo2.ch_names.index(ch)
        x2, y2, z2 = loc2[index_ch, :]
        if ch in bads_epo2:
            ax.scatter(x2, y2, z2, marker='x', color='dimgrey')
        else:
            ax.scatter(x2, y2, z2, marker='o', color='dimgrey')
        if lab:
            ax.text(x2+0.012, y2+0.012, z2, lab2[index_ch], horizontalalignment='center', verticalalignment='center')


def plot_links_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: str='auto', steps: int=10, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the links in 2D within the current axis.
    """
    
    def plot_bezier_curve(x1, y1, x2, y2, ctr1, ctr2, color, weight, steps):
        if steps <= 2:
            plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=weight)
        else:
            alphas = np.linspace(0, 1, steps)
            for idx in range(len(alphas) - 1):
                a, b = alphas[idx], alphas[idx + 1]
                xn = bezier_interpolation(a, x1, x2, ctr1[0], ctr2[0])
                xnn = bezier_interpolation(b, x1, x2, ctr1[0], ctr2[0])
                yn = bezier_interpolation(a, y1, y2, ctr1[1], ctr2[1])
                ynn = bezier_interpolation(b, y1, y2, ctr1[1], ctr2[1])
                plt.plot([xn, xnn], [yn, ynn], '-', color=color, linewidth=weight)
    
    # extract sensor infos and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    
    if children:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2), children=True, child_head=not child_head)
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2, children=True, child_head=child_head)
    else:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)

    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)

    # Calculate automatic threshold
    if threshold == 'auto':
        threshold = np.max(np.median(C, 0)) + np.max(np.std(C, 0))
    else:
        threshold = threshold

    # define colormap
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

    # plot links
    for e1 in range(len(loc1)):
        x1, y1 = loc1[e1, 0], loc1[e1, 1]
        for e2 in range(len(loc2)):
            x2, y2 = loc2[e2, 0], loc2[e2, 1]
            if C[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C[e1, e2]))
                weight = 0.2 + 1.6 * ((C[e1, e2] - threshold) / (np.nanmax(C[:]) - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr1, ctr2, color_p, weight, steps)
            elif C[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C[e1, e2]))
                weight = 0.2 + 1.6 * ((-C[e1, e2] - threshold) / (np.nanmax(C[:]) - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr1, ctr2, color_n, weight, steps)


def plot_links_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, threshold: str='auto', steps: int=10, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the links in 2D within the current axis.
    """
    def plot_bezier_curve(x1, y1, x2, y2, ctr, color, weight, steps):
        if steps <= 2:
            plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=weight)
        else:
            alphas = np.linspace(0, 1, steps)
            for idx in range(len(alphas) - 1):
                a, b = alphas[idx], alphas[idx + 1]
                xn = bezier_interpolation(a, x1, x2, ctr[0])
                xnn = bezier_interpolation(b, x1, x2, ctr[0])
                yn = bezier_interpolation(a, y1, y2, ctr[1])
                ynn = bezier_interpolation(b, y1, y2, ctr[1])  # Fixed: ctr1 -> ctr
                plt.plot([xn, xnn], [yn, ynn], '-', color=color, linewidth=weight)

    # extract sensor infos and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))

    if children: 
        loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2), children=children, child_head=child_head)
    else: 
        loc1 = transform_2d_intra(loc1, traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))
        loc2 = transform_2d_intra(loc2, traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi/2))

    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)
    
    # Calculate vmin and vmax for colormap as min and max [C1, C2]
    vmax = np.nanmax([np.nanmax(C1), np.nanmax(C2)])
    vmin = np.nanmin([np.nanmin(C1), np.nanmin(C2)])

    # Calculate automatic threshold
    if threshold == 'auto':
        threshold = np.max([np.median(C1, 0), np.median(C2, 0)]) + np.max([np.std(C1, 0), np.std(C2, 0)])
    else:
        threshold = threshold

    # Define colormap for both participants
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=vmin, vmax=-threshold)

    # plot links for participant 1
    for e1 in range(len(loc1)):
        x1, y1 = loc1[e1, 0], loc1[e1, 1]
        for e2 in range(len(loc1)):
            x2, y2 = loc1[e2, 0], loc1[e2, 1]
            if C1[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C1[e1, e2]))
                weight = 0.2 + 1.6 * ((C1[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr1, color_p, weight, steps)
            elif C1[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C1[e1, e2]))
                weight = 0.2 + 1.6 * ((-C1[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr1, color_n, weight, steps)

    # plot links for participant 2
    for e1 in range(len(loc2)):
        x1, y1 = loc2[e1, 0], loc2[e1, 1]
        for e2 in range(len(loc2)):
            x2, y2 = loc2[e2, 0], loc2[e2, 1]
            if C2[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C2[e1, e2]))
                weight = 0.2 + 1.6 * ((C2[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr2, color_p, weight, steps)
            elif C2[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C2[e1, e2]))
                weight = 0.2 + 1.6 * ((-C2[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve(x1, y1, x2, y2, ctr2, color_n, weight, steps)


def plot_links_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: str='auto', steps: int=10, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the links in 3D within the current axis.
    """
    
    def plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr2, color, weight, steps):
        if steps <= 2:
            ax.plot([x1, x2], [y1, y2], [z1, z2], '-', color=color, linewidth=weight)
        else:
            alphas = np.linspace(0, 1, steps)
            for idx in range(len(alphas) - 1):
                a, b = alphas[idx], alphas[idx + 1]
                xn = bezier_interpolation(a, x1, x2, ctr1[0], ctr2[0])
                xnn = bezier_interpolation(b, x1, x2, ctr1[0], ctr2[0])
                yn = bezier_interpolation(a, y1, y2, ctr1[1], ctr2[1])
                ynn = bezier_interpolation(b, y1, y2, ctr1[1], ctr2[1])
                zn = bezier_interpolation(a, z1, z2, ctr1[2], ctr2[2])
                znn = bezier_interpolation(b, z1, z2, ctr1[2], ctr2[2])
                ax.plot([xn, xnn], [yn, ynn], [zn, znn], '-', color=color, linewidth=weight)

    # extract sensor infos and transform loc to fit with headmodel 
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    
    if children:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2), children=True, child_head=not child_head)
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2, children=True, child_head=child_head)
    else:
        loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)

    ctr1 = np.nanmean(loc1, 0)
    ctr1[2] -= 0.2
    ctr2 = np.nanmean(loc2, 0)
    ctr2[2] -= 0.2

    # Calculate automatic threshold
    if threshold == 'auto':
        threshold = np.max(np.median(C, 0)) + np.max(np.std(C, 0))
    else:
        threshold = threshold

    # define colormap
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

    # plot links
    for e1 in range(len(loc1)):
        x1, y1, z1 = loc1[e1, :]
        for e2 in range(len(loc2)):
            x2, y2, z2 = loc2[e2, :]
            if C[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C[e1, e2]))
                weight = 0.2 + 1.6 * ((C[e1, e2] - threshold) / (np.nanmax(C[:]) - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr2, color_p, weight, steps)
            elif C[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C[e1, e2]))
                weight = 0.2 + 1.6 * ((-C[e1, e2] - threshold) / (np.nanmax(C[:]) - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr2, color_n, weight, steps)


def plot_links_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, threshold: str='auto', steps: int=10, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        None: plot the links in 3D within the current axis.
          Plot hyper-connectivity in 3D.
    """
    
    def plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr2, color, weight, steps):
        if steps <= 2:
            ax.plot([x1, x2], [y1, y2], [z1, z2], '-', color=color, linewidth=weight)
        else:
            alphas = np.linspace(0, 1, steps)
            for idx in range(len(alphas) - 1):
                a, b = alphas[idx], alphas[idx + 1]
                xn = bezier_interpolation(a, x1, x2, ctr1[0], ctr2[0])
                xnn = bezier_interpolation(b, x1, x2, ctr1[0], ctr2[0])
                yn = bezier_interpolation(a, y1, y2, ctr1[1], ctr2[1])
                ynn = bezier_interpolation(b, y1, y2, ctr1[1], ctr2[1])
                zn = bezier_interpolation(a, z1, z2, ctr1[2], ctr2[2])
                znn = bezier_interpolation(b, z1, z2, ctr1[2], ctr2[2])
                ax.plot([xn, xnn], [yn, ynn], [zn, znn], '-', color=color, linewidth=weight)
        
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))

    if children: 
        loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2), children=children, child_head=not child_head)
        loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2), children=children, child_head=child_head)
    else:  
        loc1 = transform(loc1, traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi/2))
        loc2 = transform(loc2, traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi/2))

    ctr1 = np.nanmean(loc1, 0)
    ctr1[2] -= 0.2
    ctr2 = np.nanmean(loc2, 0)
    ctr2[2] -= 0.2

    # Calculate vmin and vmax for colormap as min and max [C1, C2]
    Cmax = [np.nanmax(C1[:]), np.nanmax(C2[:])]
    vmax = np.nanmax(Cmax)
    Cmin = [np.nanmin(C1[:]), np.nanmin(C2[:])]
    vmin = np.min(Cmin)

    # Calculate automatic threshold
    if threshold == 'auto':
        threshold = np.max([np.median(C1, 0), np.median(C2, 0)]) + np.max([np.std(C1, 0), np.std(C2, 0)])
    else:
        threshold = threshold

    # Define colormap for both participant
    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=vmin, vmax=-threshold)

    for e1 in range(len(loc1)):
        x1, y1, z1 = loc1[e1, :]
        for e2 in range(len(loc1)):
            x2, y2, z2 = loc1[e2, :]
            if C1[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C1[e1, e2]))
                weight = 0.2 + 1.6 * ((C1[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr1, color_p, weight, steps)
            elif C1[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C1[e1, e2]))
                weight = 0.2 + 1.6 * ((-C1[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr1, ctr1, color_n, weight, steps)

    for e1 in range(len(loc2)):
        x1, y1, z1 = loc2[e1, :]
        for e2 in range(len(loc2)):
            x2, y2, z2 = loc2[e2, :]
            if C2[e1, e2] >= threshold:
                color_p = cmap_p(norm_p(C2[e1, e2]))
                weight = 0.2 + 1.6 * ((C2[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr2, ctr2, color_p, weight, steps)
            elif C2[e1, e2] <= -threshold:
                color_n = cmap_n(norm_n(C2[e1, e2]))
                weight = 0.2 + 1.6 * ((-C2[e1, e2] - threshold) / (vmax - threshold))
                plot_bezier_curve_3d(x1, y1, z1, x2, y2, z2, ctr2, ctr2, color_n, weight, steps)


def plot_2d_topomap_inter(ax, children: bool=False):
    """
    Plot 2D head topomap for hyper-connectivity

    Arguments:
        ax : Matplotlib axis
        children : bool
            If True, apply transformations for child head model.

    Returns:
        None : plot the 2D topomap within the current axis.
    """
    def add_head(ax, center, radius, ear_width, ear_height, nose_coords, ear_offset, zorder=0):
        circle = matplotlib.patches.Circle(xy=center, radius=radius, edgecolor="k", facecolor="w")
        ax.add_patch(circle)
        ax.add_patch(matplotlib.patches.Ellipse(xy=[center[0], center[1] + radius + ear_offset], width=ear_height, height=ear_width, edgecolor="k", facecolor="w", zorder=zorder))
        ax.add_patch(matplotlib.patches.Ellipse(xy=[center[0], center[1] - radius - ear_offset], width=ear_height, height=ear_width, edgecolor="k", facecolor="w", zorder=zorder))
        ax.add_patch(matplotlib.patches.Polygon(xy=nose_coords, edgecolor="k", facecolor="w", zorder=zorder))

    if children:
        scaling_factor = 0.95
        add_head(ax, center=[-0.178, 0], radius=0.065, ear_width=0.01625, ear_height=0.0325, nose_coords=[[-0.130 * scaling_factor, -0.02 * scaling_factor], [-0.130 * scaling_factor, 0.02 * scaling_factor], [-0.107 * scaling_factor, 0]], ear_offset=0.005)
        add_head(ax, center=[0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[0.087, -0.027], [0.087, 0.027], [0.068, 0]], ear_offset=0.005)
    else: 
        add_head(ax, center=[-0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[-0.087, -0.027], [-0.087, 0.027], [-0.068, 0]], ear_offset=0.005)
        add_head(ax, center=[0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[0.087, -0.027], [0.087, 0.027], [0.068, 0]], ear_offset=0.005)

    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_2d_topomap_intra(ax, children: bool=False):
    """
    Plot 2D head topomap for intra-brain visualisation
    
    Arguments:
        ax : Matplotlib axis
        children : bool
            If True, apply transformations for child head model.

    Returns:
        None : plot the 2D topomap within the current axis.
    """
    def add_head(ax, center, radius, ear_width, ear_height, nose_coords, ear_offset, zorder=0):
        circle = matplotlib.patches.Circle(xy=center, radius=radius, edgecolor="k", facecolor="w")
        ax.add_patch(circle)
        ax.add_patch(matplotlib.patches.Ellipse(xy=[center[0] + radius + ear_offset, center[1]], width=ear_height, height=ear_width, edgecolor="k", facecolor="w", zorder=zorder))
        ax.add_patch(matplotlib.patches.Ellipse(xy=[center[0] - radius - ear_offset, center[1]], width=ear_height, height=ear_width, edgecolor="k", facecolor="w", zorder=zorder))
        ax.add_patch(matplotlib.patches.Polygon(xy=nose_coords, edgecolor="k", facecolor="w", zorder=zorder))

    if children:
        scaling_factor = 0.95
        add_head(ax, center=[-0.178, 0], radius=0.065, ear_width=0.01625, ear_height=0.0325, nose_coords=[[-0.130 * scaling_factor, -0.02 * scaling_factor], [-0.130 * scaling_factor, 0.02 * scaling_factor], [-0.107 * scaling_factor, 0]], ear_offset=0.005)
        add_head(ax, center=[0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[0.151, 0.091], [0.205, 0.091], [0.178, 0.11]], ear_offset=0.005)
    else:
        add_head(ax, center=[-0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[-0.151, 0.091], [-0.205, 0.091], [-0.178, 0.11]], ear_offset=0.005)
        add_head(ax, center=[0.178, 0], radius=0.1, ear_width=0.025, ear_height=0.05, nose_coords=[[0.151, 0.091], [0.205, 0.091], [0.178, 0.11]], ear_offset=0.005)

    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    ax.set_xticks([])
    ax.set_yticks([])


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


def get_3d_heads_inter(children: bool=False, child_head: bool=False):
    """
    Returns Vertices and Faces of a 3D OBJ representing two facing heads.
    If children and child_head are True, one head will be smaller.
    """
   
    # Extract vertices and faces for the first head
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('hypyp') / 'data/Basehead.obj'
    filename = file_manager.enter_context(importlib_resources.as_file(ref))

    mesh = meshio.read(Path(filename).resolve())
    zoom = 0.064
    interval = 0.32

    if children and child_head:
        scale_child = 0.68
        scale_adult = 1.03
    else:
        scale_child = 1.0
        scale_adult = 1.0

    head1_v = mesh.points * zoom * scale_adult
    head1_f = mesh.cells[0].data

    # Copy the first head to create a second head
    head2_v = copy(mesh.points * zoom * scale_child)
    # Move the vertices by Y rotation and Z translation
    rotY = np.pi
    newX = head2_v[:, 0] * np.cos(rotY) - head2_v[:, 2] * np.sin(rotY)
    newZ = head2_v[:, 0] * np.sin(rotY) + head2_v[:, 2] * np.cos(rotY)
    head2_v[:, 0] = newX
    head2_v[:, 2] = newZ

    head1_v[:, 2] = head1_v[:, 2] - interval / 2
    head2_v[:, 2] = head2_v[:, 2] + interval / 2

    # Use the same faces
    head2_f = copy(mesh.cells[0].data)

    # Concatenate the vertices
    vertices = np.concatenate((head1_v, head2_v))
    # Concatenate the faces, shift vertices indexes for second head
    faces = np.concatenate((head1_f, head2_f + len(head1_v)))
    return vertices, faces


def get_3d_heads_intra(children: bool=False, child_head: bool=False):
    """
    Returns Vertices and Faces of a 3D OBJ representing two facing heads.
    If children and child_head are True, one head will be smaller.
    """
   
    # Extract vertices and faces for the first head
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('hypyp') / 'data/Basehead.obj'
    filename = file_manager.enter_context(importlib_resources.as_file(ref))

    mesh = meshio.read(Path(filename).resolve())
    zoom = 0.064
    interval = 0.5

    if children and child_head:
        scale_child = 0.68
        scale_adult = 1.03
    else:
        scale_child = 1.0
        scale_adult = 1.0

    head1_v = mesh.points * zoom * scale_adult
    head1_f = mesh.cells[0].data

    # Copy the first head to create a second head
    head2_v = copy(mesh.points * zoom * scale_child)
    head2_v[:, 0] = head2_v[:, 0] + interval

    # Use the same faces
    head2_f = copy(mesh.cells[0].data)

    # Concatenate the vertices
    vertices = np.concatenate((head1_v, head2_v))
    # Concatenate the faces, shift vertices indexes for second head
    faces = np.concatenate((head1_f, head2_f + len(head1_v)))
    return vertices, faces


def plot_significant_sensors(T_obs_plot: np.ndarray, epochs: mne.Epochs, significant: np.ndarray = None):
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
    
    fig = plt.figure()
    ax = fig.add_subplot()
    if significant is not None:
        mne.viz.plot_topomap(T_obs_plot, pos, vlim=(vmin, vmax), sensors=False, axes=ax, show=False)
        loc1 = copy(np.array([ch['loc'][:3] for ch in epochs.info['chs']]))
        for ich, ch in enumerate(epochs.ch_names):
            index_ch = epochs.ch_names.index(ch)
            x1, y1, z1 = loc1[index_ch, :]
            if significant[ich] != 0:
                ax.plot(x1, y1, marker='o', color='white')
    else:        
        mne.viz.plot_topomap(T_obs_plot, pos, vlim=(vmin, vmax), sensors=True, axes=ax, show=True)

    return None


def viz_2D_headmodel_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = True, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 2D.
        ax: The new Axes object.
    """
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")  

    if children:
        vertices, faces = get_3d_heads_inter(children=True, child_head=child_head)
    else:
        vertices, faces = get_3d_heads_inter()
      
    camera = Camera("ortho", theta=90, phi=180, scale=1)
    mesh = Mesh(ax, camera.transform @ glm.yrotate(90), vertices, faces,
                facecolors='white', edgecolors='black', linewidths=.25)
    camera.connect(ax, mesh.update)  
    plt.gca().set_aspect('equal', 'box')  
    plt.axis('off')  

    if children:
        plot_sensors_2d_inter(epo1, epo2, lab=lab, children=True, child_head=child_head)
        plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps, children=True, child_head=child_head)
    else:
        plot_sensors_2d_inter(epo1, epo2, lab=lab)
        plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps)

    plt.tight_layout()  
    plt.show()  
    
    return ax


def viz_2D_topomap_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = False, children: bool=False, child_head: bool=False):
    """
    Visualize 2D topographic map interaction between two sets of epochs.
    
    Parameters:
        epo1 (mne.Epochs): The first set of epochs.
        epo2 (mne.Epochs): The second set of epochs.
        C (np.ndarray): Connectivity matrix.
        threshold (float, optional): Threshold for connectivity visualization. Default is 0.95.
        steps (int, optional): Number of steps for interpolation. Default is 10.
        lab (bool, optional): Whether to label the sensors. Default is False.
        children (bool, optional): Whether the data is from children. Default is False.
        child_head (bool, optional): Whether the child is the second participant. Default is False.
        
    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axes object with the plot.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)
    ax.axis("off")

    if children:
        plot_2d_topomap_inter(ax, children=True)
        plot_sensors_2d_inter(epo1, epo2, lab=lab, children=True, child_head=child_head)
        plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps, children=True, child_head=child_head)
    else:
        plot_2d_topomap_inter(ax)
        plot_sensors_2d_inter(epo1, epo2, lab=lab)
        plot_links_2d_inter(epo1, epo2, C=C, threshold=threshold, steps=steps)

    plt.tight_layout()
    plt.show()
    return ax


def viz_2D_topomap_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, threshold: float=0.95, steps: int=2, lab: bool = False, children: bool=False, child_head: bool=False):
    """
    Visualization of intra-brain connectivity in 2D.

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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        Plot head topomap with sensors and 
            intra-brain connectivity links in 2D.
        ax: The new Axes object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)
    ax.axis("off")
    
    if children:
        plot_2d_topomap_intra(ax, children=True)
        # bads are represented as squares
        plot_sensors_2d_intra(epo1, epo2, lab=lab, children=True, child_head=child_head)
        # plotting links according to sign (red for positive values,
        # blue for negative) and value (line thickness increases
        # with the strength of connectivity)
        plot_links_2d_intra(epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps, children=True, child_head=child_head)
    else:
        plot_2d_topomap_intra(ax)
        plot_sensors_2d_intra(epo1, epo2, lab=lab)
        plot_links_2d_intra(epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps)
        
    plt.tight_layout()
    plt.show()

    return ax


def viz_3D_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = False, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 3D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    if children:
        vertices, faces = get_3d_heads_inter(children=True, child_head=child_head)
    else:
        vertices, faces = get_3d_heads_inter()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis("off")
    plot_3d_heads(ax, vertices, faces)

    if children:
        plot_sensors_3d_inter(ax, epo1, epo2, lab=lab, children=True, child_head=child_head)
        plot_links_3d_inter(ax, epo1, epo2, C=C, threshold=threshold, steps=steps, children=True, child_head=child_head)
    else:
        plot_sensors_3d_inter(ax, epo1, epo2, lab=lab)
        plot_links_3d_inter(ax, epo1, epo2, C=C, threshold=threshold, steps=steps)

    plt.tight_layout()
    plt.show()

    return ax


def viz_3D_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, threshold: float=0.95, steps: int=10, lab: bool = False, children: bool=False, child_head: bool=False):
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
        children: bool
            If True, apply transformations for child head model.
        child_head: bool
            If True, apply transformations for child's head model.

    Returns:
        Plot headmodel with sensors and 
            connectivity links in 3D.
        ax: The new Axes object.
    """

    # defining head model and adding sensors
    if children:
        vertices, faces = get_3d_heads_intra(children=True, child_head=child_head)
    else:
        vertices, faces = get_3d_heads_intra()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis("off")
    plot_3d_heads(ax, vertices, faces)

    if children:
        # bads are represented as squares
        plot_sensors_3d_intra(ax, epo1, epo2, lab=lab, children=True, child_head=child_head)
        # plotting links according to sign (red for positive values,
        # blue for negative) and value (line thickness increases
        # with the strength of connectivity)
        plot_links_3d_intra(ax, epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps, children=True, child_head=child_head)
    else:
        plot_sensors_3d_intra(ax, epo1, epo2, lab=lab)
        plot_links_3d_intra(ax, epo1, epo2, C1=C1, C2=C2, threshold=threshold, steps=steps)

    plt.tight_layout()
    plt.show()

    return ax


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
        raise ValueError('Analysis must be set as phase, power, or wtc.')

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