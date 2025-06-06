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


def transform(locs: np.ndarray, traX: float=0.15, traY: float=0, traZ: float=0.5, 
             rotY: float=(np.pi)/2, rotZ: float=(np.pi)/2, children: bool=False, 
             child_head: bool=False) -> np.ndarray:
    """
    Apply a series of transformations to 3D coordinates for visualization.
    
    This function transforms electrode coordinates to fit properly into the head 
    models for hyperscanning visualizations. It can adjust for adult or child head 
    sizes and positions.
    
    Parameters
    ----------
    locs : np.ndarray
        Array of shape (n_sensors, 3) containing the 3D coordinates to transform
        
    traX : float, optional
        Translation along X-axis in meters (default=0.15)
        
    traY : float, optional
        Translation along Y-axis in meters (default=0)
        
    traZ : float, optional
        Translation along Z-axis in meters (default=0.5)
        
    rotY : float, optional
        Rotation around Y-axis in radians (default=π/2)
        
    rotZ : float, optional
        Rotation around Z-axis in radians (default=π/2)
        
    children : bool, optional
        Whether to apply transformations specific to children's head models (default=False)
        
    child_head : bool, optional
        If True and children=True, apply transformations for a child's head model
        If False and children=True, apply transformations for an adult's head model
        with a child (default=False)
    
    Returns
    -------
    np.ndarray
        The transformed 3D coordinates with the same shape as input
    
    Notes
    -----
    The transformation sequence is:
    1. Z-axis rotation
    2. Scaling (with different factors for children)
    3. X/Y/Z translations
    4. Y-axis rotation
    
    When using with hyperscanning visualization functions, this transformation
    is typically applied with different parameters to position each participant's 
    sensors appropriately in the visualization.
    
    Examples
    --------
    >>> # Transform adult electrode positions
    >>> transformed_locs = transform(electrode_positions, traX=-0.17, rotZ=-np.pi/2)
    >>> 
    >>> # Transform child electrode positions
    >>> child_locs = transform(
    ...     electrode_positions, 
    ...     traX=0.17, 
    ...     rotZ=np.pi/2, 
    ...     children=True, 
    ...     child_head=True
    ... )
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


def transform_2d_intra(locs: np.ndarray, traX: float=0.15, traY: float=0, traZ: float=0, 
                       rotZ: float=(np.pi)/2, children: bool=False, 
                       child_head: bool=False) -> np.ndarray:
    """
    Transform electrode coordinates for 2D intra-brain visualizations.
    
    This function applies transformations to electrode coordinates for
    2D visualizations of intra-brain connectivity. It's a simplified version
    of the transform() function with fewer rotations.
    
    Parameters
    ----------
    locs : np.ndarray
        Array of shape (n_sensors, 3) containing the 3D coordinates to transform
        
    traX : float, optional
        Translation along X-axis in meters (default=0.15)
        
    traY : float, optional
        Translation along Y-axis in meters (default=0)
        
    traZ : float, optional
        Translation along Z-axis in meters (default=0)
        
    rotZ : float, optional
        Rotation around Z-axis in radians (default=π/2)
        
    children : bool, optional
        Whether to apply transformations specific to children's head models (default=False)
        
    child_head : bool, optional
        If True and children=True, apply transformations for a child's head model
        If False and children=True, apply transformations for an adult's head model
        with a child (default=False)
    
    Returns
    -------
    np.ndarray
        The transformed 3D coordinates with the same shape as input
    
    Notes
    -----
    This function is primarily used for 2D intra-brain connectivity visualizations
    where the full 3D rotation from transform() is not needed.
    
    The main differences from transform() are:
    1. No Y-axis rotation
    2. Simpler scaling for the Z dimension
    
    Examples
    --------
    >>> # Transform adult electrode positions for 2D intra-brain visualization
    >>> transformed_locs = transform_2d_intra(
    ...     electrode_positions, 
    ...     traX=-0.178, 
    ...     traY=0.012
    ... )
    >>> 
    >>> # Transform child electrode positions
    >>> child_locs = transform_2d_intra(
    ...     electrode_positions, 
    ...     traX=0.178, 
    ...     traY=0.012, 
    ...     children=True, 
    ...     child_head=True
    ... )
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

     # Apply scaling to all coordinates
    locs = locs * scale
    
    # translation
    locs[:, 0] = locs[:, 0] + traX
    locs[:, 1] = locs[:, 1] + traY
    locs[:, 2] = locs[:, 2] + traZ

    # Reduce the size of the eeg headsets
    newZ = locs[:, 0] * np.cos(rotZ) + locs[:, 1] * np.cos(rotZ) + locs[:, 2] * np.cos(rotZ/2)
    locs[:, 2] = newZ

    return locs

def bezier_interpolation(t, p0, p1, c0, c1=None):
    """
    Calculate points on a cubic Bezier curve for smooth connection visualization.
    
    This function implements cubic Bezier curve interpolation to create smooth
    curves for visualizing connectivity between electrodes. It's especially
    useful for creating aesthetically pleasing inter-brain connectivity
    visualizations.
    
    Parameters
    ----------
    t : float
        Interpolation parameter between 0 and 1
        0 represents the start point, 1 represents the end point
        
    p0 : array-like
        Starting point coordinates
        
    p1 : array-like
        Ending point coordinates
        
    c0 : array-like
        Control point for the starting point
        
    c1 : array-like, optional
        Control point for the ending point
        If None, c0 is used for both control points (for intra-brain curves)
        (default=None)
    
    Returns
    -------
    array-like
        Point coordinates on the Bezier curve at parameter t
    
    Notes
    -----
    The function implements the cubic Bezier formula:
    B(t) = (1-t)³P₀ + 3(1-t)²t(2P₀-C₀) + 3(1-t)t²(2P₁-C₁) + t³P₁
    
    When visualizing intra-brain connectivity (within one brain), typically
    c1 is set to None, which makes the curve symmetric.
    
    For inter-brain connectivity, different control points are used to create
    a curve that properly arcs between the two head models.
    
    Examples
    --------
    >>> # Calculate 10 points along a Bezier curve for smooth connection
    >>> points = []
    >>> for i in range(10):
    ...     t = i / 9  # 0 to 1
    ...     point = bezier_interpolation(
    ...         t, 
    ...         [0, 0, 0],      # start point
    ...         [1, 1, 0],      # end point
    ...         [0.5, 0, 0]     # control point
    ...     )
    ...     points.append(point)
    """

    if c1 is None:
        c1 = c0
        
    return ((1 - t) ** 3 * p0 +
            3 * (1 - t) ** 2 * t * (2 * p0 - c0) +
            3 * (1 - t) * t ** 2 * (2 * p1 - c1) +
            t ** 3 * p1)


def plot_sensors_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = True, 
                          children: bool=False, child_head: bool=False):
    """
    Plot sensors from two participants in 2D for inter-brain visualizations.
    
    This function plots the sensor positions from two participants in a 2D view,
    with special markers for bad channels. It's typically used as part of
    inter-brain connectivity visualizations.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    lab : bool, optional
        Whether to plot channel labels (default=True)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the sensors in the current matplotlib Axes
    
    Notes
    -----
    The function:
    1. Extracts electrode coordinates from the Epochs objects
    2. Transforms them for proper visualization using transform()
    3. Plots each sensor with:
       - Circle markers for good channels
       - X markers for bad channels (those in info['bads'])
    4. Optionally adds channel labels
    
    This function is typically used as part of a larger visualization
    function like viz_2D_topomap_inter() rather than called directly.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Create a figure
    >>> fig, ax = plt.subplots()
    >>> # Plot sensors
    >>> plot_sensors_2d_inter(epochs_subj1, epochs_subj2, lab=True)
    >>> plt.show()
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


def plot_sensors_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, 
                         children: bool=False, child_head: bool=False):
    """
    Plot sensors from two participants in 2D for intra-brain visualizations.
    
    This function plots the sensor positions from two participants for visualizing
    intra-brain connectivity (connectivity within each brain separately). The sensors 
    are positioned side by side rather than facing each other as in inter-brain plots.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the sensors in the current matplotlib Axes
    
    Notes
    -----
    The function:
    1. Extracts electrode coordinates from the Epochs objects
    2. Transforms them for side-by-side visualization using transform_2d_intra()
    3. Plots each sensor with:
       - Circle markers for good channels
       - X markers for bad channels (those in info['bads'])
    4. Optionally adds channel labels
    
    Unlike plot_sensors_2d_inter(), this function positions the heads side by side
    rather than facing each other, which is appropriate for visualizing intra-brain
    connectivity where comparisons are made between participants.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Create a figure
    >>> fig, ax = plt.subplots()
    >>> # Plot sensors for intra-brain visualization
    >>> plot_sensors_2d_intra(epochs_subj1, epochs_subj2, lab=True)
    >>> plt.show()
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


def plot_sensors_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, 
                         children: bool=False, child_head: bool=False):
    """
    Plot sensors from two participants in 3D for inter-brain visualizations.
    
    This function plots the sensor positions from two participants in a 3D view,
    with special markers for bad channels. It's typically used as part of
    inter-brain connectivity 3D visualizations.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib 3D axis (created with projection='3d')
        
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the sensors in the provided matplotlib 3D Axes
    
    Notes
    -----
    The function:
    1. Extracts electrode coordinates from the Epochs objects
    2. Transforms them for proper 3D visualization using transform()
    3. Plots each sensor in 3D space with:
       - Sphere markers for good channels
       - X markers for bad channels (those in info['bads'])
    4. Optionally adds channel labels in 3D space
    
    This function is typically used as part of a larger visualization
    function like viz_3D_inter() rather than called directly.
    
    The main difference from the 2D equivalent is that this function requires
    a 3D axis and uses 3D scatter plots instead of 2D plots.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> # Create a 3D figure
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # Plot sensors in 3D
    >>> plot_sensors_3d_inter(ax, epochs_subj1, epochs_subj2, lab=True)
    >>> plt.show()
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



def plot_sensors_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False, 
                         children: bool = False, child_head: bool = False):
    """
    Plot sensors from two participants in 3D for intra-brain visualizations.
    
    This function plots the sensor positions from two participants in a 3D view
    for visualizing intra-brain connectivity. The sensors are positioned side by side 
    rather than facing each other as in inter-brain plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib 3D axis (created with projection='3d')
        
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the sensors in the provided matplotlib 3D Axes
    
    Notes
    -----
    The function:
    1. Extracts electrode coordinates from the Epochs objects
    2. Transforms them for side-by-side 3D visualization using transform()
    3. Plots each sensor in 3D space with:
       - Sphere markers for good channels
       - X markers for bad channels (those in info['bads'])
    4. Optionally adds channel labels in 3D space
    
    Unlike plot_sensors_3d_inter(), this function positions the heads side by side
    rather than facing each other, which is appropriate for visualizing intra-brain
    connectivity where comparisons are made between participants.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> # Create a 3D figure
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # Plot sensors in 3D for intra-brain visualization
    >>> plot_sensors_3d_intra(ax, epochs_subj1, epochs_subj2, lab=True)
    >>> plt.show()
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


def plot_links_2d_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, 
                       threshold: str='auto', steps: int=10, children: bool=False, 
                       child_head: bool=False):
    """
    Plot inter-brain connectivity links between two participants in 2D.
    
    This function visualizes the connectivity between electrodes of two participants
    using bezier curves. Only connections above a specified threshold are displayed,
    with line color and thickness indicating the strength and sign of the connection.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_2) containing
        connectivity values between all pairs of electrodes across participants
        
    threshold : float or str, optional
        Threshold for displaying connections (default='auto')
        - If float: Only connections with absolute value above this threshold are displayed
        - If 'auto': Threshold is set to the maximum median by column plus
          the maximum standard error by column
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        - If steps < 3: Straight lines are drawn instead of curves
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the connectivity links in the current matplotlib Axes
    
    Notes
    -----
    The function:
    1. Transforms electrode coordinates for proper visualization
    2. Calculates control points for bezier curves based on head centers
    3. Determines connectivity threshold (automatically or from parameter)
    4. Plots connections with:
       - Red color scale for positive connectivity values
       - Blue color scale for negative connectivity values
       - Line thickness proportional to connection strength
    
    This function is typically used as part of a larger visualization
    function like viz_2D_topomap_inter() rather than called directly.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # Create random connectivity matrix
    >>> n_channels = len(epochs_subj1.ch_names)
    >>> connectivity = np.random.rand(n_channels, n_channels) * 2 - 1  # Values between -1 and 1
    >>> # Create a figure
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> # Plot 2D connectivity links
    >>> plot_links_2d_inter(
    ...     epochs_subj1, 
    ...     epochs_subj2, 
    ...     connectivity, 
    ...     threshold=0.5, 
    ...     steps=15
    ... )
    >>> plt.show()
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
    cmap_p = matplotlib.colormaps['Reds']

    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.colormaps['Blues_r']

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


def plot_links_2d_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, 
                       threshold: str='auto', steps: int=10, children: bool=False, 
                       child_head: bool=False):
    """
    Plot intra-brain connectivity links for two participants in 2D.
    
    This function visualizes the connectivity within each participant's brain
    separately using bezier curves. Only connections above a specified threshold 
    are displayed, with line color and thickness indicating the strength and sign 
    of the connection.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C1 : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_1) containing
        connectivity values between all pairs of electrodes for participant 1
        
    C2 : np.ndarray
        Connectivity matrix with shape (n_channels_2, n_channels_2) containing
        connectivity values between all pairs of electrodes for participant 2
        
    threshold : float or str, optional
        Threshold for displaying connections (default='auto')
        - If float: Only connections with absolute value above this threshold are displayed
        - If 'auto': Threshold is set to the maximum of the median values plus
          the maximum of standard errors across both participants
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        - If steps < 3: Straight lines are drawn instead of curves
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the connectivity links in the current matplotlib Axes
    
    Notes
    -----
    The function:
    1. Transforms electrode coordinates for side-by-side visualization
    2. Calculates control points for bezier curves based on each head's center
    3. Determines connectivity threshold (automatically or from parameter)
    4. Plots connections with:
       - Red color scale for positive connectivity values
       - Blue color scale for negative connectivity values
       - Line thickness proportional to connection strength
    
    Unlike plot_links_2d_inter(), this function:
    - Handles two separate connectivity matrices (one per participant)
    - Uses the same color scale for both participants for consistent visualization
    - Uses separate control points for each participant's connectivity curves
    
    This function is typically used as part of a larger visualization
    function like viz_2D_topomap_intra() rather than called directly.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # Create random connectivity matrices
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity_1 = np.random.rand(n_channels_1, n_channels_1) * 2 - 1  # Values between -1 and 1
    >>> connectivity_2 = np.random.rand(n_channels_2, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create a figure
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> # Plot 2D intra-brain connectivity links
    >>> plot_links_2d_intra(
    ...     epochs_subj1, 
    ...     epochs_subj2, 
    ...     connectivity_1,
    ...     connectivity_2,
    ...     threshold=0.5, 
    ...     steps=15
    ... )
    >>> plt.show()
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
    cmap_p = matplotlib.colormaps['Reds']

    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.colormaps['Blues_r']
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


def plot_links_3d_inter(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, 
                       threshold: str='auto', steps: int=10, children: bool=False, 
                       child_head: bool=False):
    """
    Plot inter-brain connectivity links between two participants in 3D.
    
    This function visualizes the connectivity between electrodes of two participants
    using 3D bezier curves. Only connections above a specified threshold are displayed,
    with line color and thickness indicating the strength and sign of the connection.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib 3D axis (created with projection='3d')
        
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_2) containing
        connectivity values between all pairs of electrodes across participants
        
    threshold : float or str, optional
        Threshold for displaying connections (default='auto')
        - If float: Only connections with absolute value above this threshold are displayed
        - If 'auto': Threshold is set to the maximum median by column plus
          the maximum standard error by column
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        - If steps < 3: Straight lines are drawn instead of curves
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the connectivity links in the provided matplotlib 3D Axes
    
    Notes
    -----
    The function:
    1. Transforms electrode coordinates for proper 3D visualization
    2. Calculates 3D control points for bezier curves (offset from head centers)
    3. Determines connectivity threshold (automatically or from parameter)
    4. Plots 3D connections with:
       - Red color scale for positive connectivity values
       - Blue color scale for negative connectivity values
       - Line thickness proportional to connection strength
    
    This function is similar to plot_links_2d_inter() but works in 3D space.
    It is typically used as part of a larger visualization function like
    viz_3D_inter() rather than called directly.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import numpy as np
    >>> # Create random connectivity matrix
    >>> n_channels = len(epochs_subj1.ch_names)
    >>> connectivity = np.random.rand(n_channels, n_channels) * 2 - 1  # Values between -1 and 1
    >>> # Create a 3D figure
    >>> fig = plt.figure(figsize=(10, 8))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # Plot 3D connectivity links
    >>> plot_links_3d_inter(
    ...     ax,
    ...     epochs_subj1, 
    ...     epochs_subj2, 
    ...     connectivity, 
    ...     threshold=0.5, 
    ...     steps=15
    ... )
    >>> plt.show()
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
    cmap_p = matplotlib.colormaps['Reds']

    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.colormaps['Blues_r']
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


def plot_links_3d_intra(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, 
                       threshold: str='auto', steps: int=10, children: bool=False, 
                       child_head: bool=False):
    """
    Plot intra-brain connectivity links for two participants in 3D.
    
    This function visualizes the connectivity within each participant's brain
    separately using 3D bezier curves. Only connections above a specified threshold 
    are displayed, with line color and thickness indicating the strength and sign 
    of the connection.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib 3D axis (created with projection='3d')
        
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C1 : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_1) containing
        connectivity values between all pairs of electrodes for participant 1
        
    C2 : np.ndarray
        Connectivity matrix with shape (n_channels_2, n_channels_2) containing
        connectivity values between all pairs of electrodes for participant 2
        
    threshold : float or str, optional
        Threshold for displaying connections (default='auto')
        - If float: Only connections with absolute value above this threshold are displayed
        - If 'auto': Threshold is set to the maximum of the median values plus
          the maximum of standard errors across both participants
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        - If steps < 3: Straight lines are drawn instead of curves
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    None
        Plots the connectivity links in the provided matplotlib 3D Axes
    
    Notes
    -----
    The function:
    1. Transforms electrode coordinates for side-by-side 3D visualization
    2. Calculates 3D control points for bezier curves (offset vertically from head centers)
    3. Determines connectivity threshold (automatically or from parameter)
    4. Plots 3D connections with:
       - Red color scale for positive connectivity values
       - Blue color scale for negative connectivity values
       - Line thickness proportional to connection strength
    
    Unlike plot_links_3d_inter(), this function:
    - Handles two separate connectivity matrices (one per participant)
    - Uses the same color scale for both participants for consistent visualization
    - Uses separate control points for each participant's connectivity curves
    
    This function is typically used as part of a larger visualization
    function like viz_3D_intra() rather than called directly.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import numpy as np
    >>> # Create random connectivity matrices
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity_1 = np.random.rand(n_channels_1, n_channels_1) * 2 - 1  # Values between -1 and 1
    >>> connectivity_2 = np.random.rand(n_channels_2, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create a 3D figure
    >>> fig = plt.figure(figsize=(10, 8))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # Plot 3D intra-brain connectivity links
    >>> plot_links_3d_intra(
    ...     ax,
    ...     epochs_subj1, 
    ...     epochs_subj2, 
    ...     connectivity_1,
    ...     connectivity_2,
    ...     threshold=0.5, 
    ...     steps=15
    ... )
    >>> plt.show()
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
    cmap_p = matplotlib.colormaps['Reds']

    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=vmax)
    cmap_n = matplotlib.colormaps['Blues_r']
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
    Plot 2D head topomap outlines for inter-brain visualizations.
    
    This function draws the head outlines for two participants facing each other
    in a 2D representation. It creates the basic head shapes including circles 
    for the heads, ellipses for the ears, and polygons for the noses.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis where the topomap will be plotted
        
    children : bool, optional
        Whether to use child head model proportions (default=False)
        If True, one head will be scaled to child size
    
    Returns
    -------
    None
        Plots the head outlines in the provided matplotlib Axes
    
    Notes
    -----
    The function:
    1. Creates head outlines for two facing participants
    2. Adds ears and nose features to each head
    3. If children=True, scales one head (right side) to child proportions
    4. Removes axis ticks and spines for cleaner visualization
    
    This function handles only the head outlines - it does not plot sensors
    or connectivity. It is typically used as part of a larger visualization
    function like viz_2D_topomap_inter() which then adds sensors and links.
    
    Key coordinates:
    - Head centers: [-0.178, 0] and [0.178, 0]
    - Adult head radius: 0.1
    - Child head radius (when children=True): 0.065
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Create a figure
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> # Plot head outlines
    >>> plot_2d_topomap_inter(ax, children=False)
    >>> plt.show()
    >>> 
    >>> # Plot adult-child head outlines
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> plot_2d_topomap_inter(ax, children=True)
    >>> plt.show()
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
    Plot 2D head topomap outlines for intra-brain visualizations.
    
    This function draws the head outlines for two participants positioned side by side
    in a 2D representation. It creates the basic head shapes including circles 
    for the heads, ellipses for the ears, and polygons for the noses.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis where the topomap will be plotted
        
    children : bool, optional
        Whether to use child head model proportions (default=False)
        If True, one head will be scaled to child size
    
    Returns
    -------
    None
        Plots the head outlines in the provided matplotlib Axes
    
    Notes
    -----
    The function:
    1. Creates head outlines for two participants positioned side by side
    2. Adds ears and nose features to each head
    3. If children=True, scales one head (right side) to child proportions
    4. Removes axis ticks and spines for cleaner visualization
    
    Unlike plot_2d_topomap_inter() which positions heads facing each other,
    this function positions heads side by side with noses pointing upward.
    This orientation is appropriate for intra-brain connectivity visualization
    where each participant's brain connectivity is analyzed separately.
    
    Key coordinates:
    - Head centers: [-0.178, 0] and [0.178, 0]
    - Adult head radius: 0.1
    - Child head radius (when children=True): 0.065
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Create a figure
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> # Plot head outlines for intra-brain visualization
    >>> plot_2d_topomap_intra(ax, children=False)
    >>> plt.show()
    >>> 
    >>> # Plot adult-child head outlines
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> plot_2d_topomap_intra(ax, children=True)
    >>> plt.show()
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
    Plot 3D head models using vertices and faces.
    
    This function draws 3D head models defined by vertices and faces in a 
    wireframe representation. It plots the edges between vertices that define
    the faces of the 3D model.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib 3D axis (created with projection='3d')
        
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing the 3D coordinates of vertices
        
    faces : np.ndarray
        Array of shape (n_faces, 4) containing the vertex indices for each face
        Each face is defined by 4 vertex indices
    
    Returns
    -------
    None
        Plots the 3D head model wireframe in the provided matplotlib 3D Axes
    
    Notes
    -----
    The function:
    1. Extracts vertex coordinates (with some reordering of x, y, z for correct orientation)
    2. For each face, plots lines connecting its vertices
    3. Uses a thin gray line style for the wireframe
    
    This function creates a lightweight wireframe representation rather than
    a fully shaded 3D model, which is sufficient for visualization purposes
    while maintaining good performance.
    
    The wireframe approach allows electrode and connectivity visualization to be
    clearly visible in front of or behind the head model.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> # Create a 3D figure
    >>> fig = plt.figure(figsize=(10, 8))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # Get vertices and faces for head models
    >>> vertices, faces = get_3d_heads_inter()
    >>> # Plot 3D heads
    >>> plot_3d_heads(ax, vertices, faces)
    >>> plt.show()
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
    Get vertices and faces for 3D head models in inter-brain configuration.
    
    This function returns the vertices and faces defining two 3D head models
    positioned facing each other, as needed for inter-brain connectivity visualization.
    
    Parameters
    ----------
    children : bool, optional
        Whether to use child head model scaling (default=False)
        
    child_head : bool, optional
        If True and children=True, scales the second head to child size
        If False and children=True, scales the first head to child size
        (default=False)
    
    Returns
    -------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing the 3D coordinates of all vertices
        from both head models
        
    faces : np.ndarray
        Array of shape (n_faces, 4) containing the vertex indices for each face
        from both head models
    
    Notes
    -----
    The function:
    1. Loads a base head model from an OBJ file included in the package
    2. Creates two copies of the model with appropriate scaling
    3. Rotates and translates the models to position them facing each other
    4. If children=True and child_head=True, applies scaling to make the second head smaller
    5. Combines vertices and faces into unified arrays
    
    For the faces array in the second head, vertex indices are offset by
    the number of vertices in the first head to maintain correct indexing
    in the combined array.
    
    This function is typically used to provide head models for the 
    viz_3D_inter() visualization function.
    
    Examples
    --------
    >>> # Get vertices and faces for two adult heads
    >>> vertices, faces = get_3d_heads_inter()
    >>> print(f"Number of vertices: {len(vertices)}")
    >>> print(f"Number of faces: {len(faces)}")
    >>> 
    >>> # Get vertices and faces for adult and child heads
    >>> vertices, faces = get_3d_heads_inter(children=True, child_head=True)
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
    Get vertices and faces for 3D head models in intra-brain configuration.
    
    This function returns the vertices and faces defining two 3D head models
    positioned side by side, as needed for intra-brain connectivity visualization.
    
    Parameters
    ----------
    children : bool, optional
        Whether to use child head model scaling (default=False)
        
    child_head : bool, optional
        If True and children=True, scales the second head to child size
        If False and children=True, scales the first head to child size
        (default=False)
    
    Returns
    -------
    vertices : np.ndarray
        Array of shape (n_vertices, 3) containing the 3D coordinates of all vertices
        from both head models
        
    faces : np.ndarray
        Array of shape (n_faces, 4) containing the vertex indices for each face
        from both head models
    
    Notes
    -----
    The function:
    1. Loads a base head model from an OBJ file included in the package
    2. Creates two copies of the model with appropriate scaling
    3. Translates the models to position them side by side along the X-axis
    4. If children=True and child_head=True, applies scaling to make the second head smaller
    5. Combines vertices and faces into unified arrays
    
    Unlike get_3d_heads_inter() which rotates one head to face the other, this function
    simply positions the heads side by side without rotation. This configuration is
    appropriate for comparing intra-brain connectivity between participants.
    
    For the faces array in the second head, vertex indices are offset by
    the number of vertices in the first head to maintain correct indexing
    in the combined array.
    
    This function is typically used to provide head models for the 
    viz_3D_intra() visualization function.
    
    Examples
    --------
    >>> # Get vertices and faces for two adult heads side by side
    >>> vertices, faces = get_3d_heads_intra()
    >>> print(f"Number of vertices: {len(vertices)}")
    >>> print(f"Number of faces: {len(faces)}")
    >>> 
    >>> # Get vertices and faces for adult and child heads side by side
    >>> vertices, faces = get_3d_heads_intra(children=True, child_head=True)
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
    Plot statistical values on a topographic map for significant sensors.
    
    This function visualizes the results of statistical tests (such as t-tests or 
    cluster-corrected t-tests) performed on sensor data across participants or conditions.
    It creates a topographic plot with statistical values displayed for significant sensors.
    
    Parameters
    ----------
    T_obs_plot : np.ndarray
        Array of shape (n_sensors,) containing the statistical values (T or F statistics)
        to plot for significant sensors. Typically, this array has zeros for non-significant
        sensors and the actual statistical values for significant ones.
        
    epochs : mne.Epochs
        Epochs object to get channel positions from
        
    significant : np.ndarray, optional
        Optional mask array of shape (n_sensors,) where non-zero values indicate
        significant sensors to highlight (default=None)
    
    Returns
    -------
    None
        Creates a topographic plot of the statistical values
    
    Notes
    -----
    The function:
    1. Extracts sensor positions from the epochs object
    2. Determines appropriate color scale limits (symmetric around zero)
    3. Creates a topographic plot using MNE's plot_topomap function
    4. If significant mask is provided, adds white circles around the significant sensors
    
    This function is designed for single-participant statistical visualization.
    For inter-brain connectivity statistics, use the plot_links_* functions instead.
    
    Examples
    --------
    >>> import numpy as np
    >>> import mne
    >>> # Create array with statistical values (zeros for non-significant sensors)
    >>> n_channels = len(epochs.ch_names)
    >>> # Example: t-values for a statistical test, with zeros for non-significant channels
    >>> t_values = np.zeros(n_channels)
    >>> # Set some random significant values
    >>> significant_indices = [3, 7, 12, 20, 25]
    >>> t_values[significant_indices] = np.random.randn(len(significant_indices)) * 3
    >>> # Plot the significant sensors
    >>> plot_significant_sensors(t_values, epochs)
    >>> 
    >>> # Alternatively, use a significance mask
    >>> significance_mask = np.zeros(n_channels)
    >>> significance_mask[significant_indices] = 1
    >>> plot_significant_sensors(t_values, epochs, significant=significance_mask)
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


def viz_2D_headmodel_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, 
                         threshold: float=0.95, steps: int=10, lab: bool = True, 
                         children: bool=False, child_head: bool=False):
    """
    Create a complete 2D head model visualization with inter-brain connectivity.
    
    This function provides a high-level interface for creating a 2D visualization
    of two head models with sensors and inter-brain connectivity links.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_2) containing
        connectivity values between all pairs of electrodes across participants
        
    threshold : float, optional
        Threshold for displaying connections (default=0.95)
        Only connections with absolute value above this threshold are displayed
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        If steps < 3, straight lines are drawn instead of curves
        
    lab : bool, optional
        Whether to plot channel labels (default=True)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the visualization
    
    Notes
    -----
    This function combines several lower-level visualization functions:
    1. Creates a figure and axis with appropriate settings
    2. Generates 3D head models and applies a camera transformation for 2D view
    3. Plots sensors using plot_sensors_2d_inter()
    4. Plots connectivity links using plot_links_2d_inter()
    
    Unlike viz_2D_topomap_inter() which uses simpler 2D head outlines,
    this function uses a 3D head model rendered in a 2D view, which
    can provide a more realistic representation.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create random connectivity matrix
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity = np.random.rand(n_channels_1, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create visualization
    >>> ax = viz_2D_headmodel_inter(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     connectivity,
    ...     threshold=0.7,
    ...     steps=15,
    ...     lab=True
    ... )
    >>> # Add a title
    >>> ax.set_title('Inter-brain connectivity')
    >>> plt.show()
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


def viz_2D_topomap_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, 
                       threshold: float=0.95, steps: int=10, lab: bool = False, 
                       children: bool=False, child_head: bool=False):
    """
    Create a 2D topographic map visualization with inter-brain connectivity.
    
    This function provides a high-level interface for creating a 2D topographic
    visualization of two head outlines with sensors and inter-brain connectivity links.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_2) containing
        connectivity values between all pairs of electrodes across participants
        
    threshold : float, optional
        Threshold for displaying connections (default=0.95)
        Only connections with absolute value above this threshold are displayed
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        If steps < 3, straight lines are drawn instead of curves
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the visualization
    
    Notes
    -----
    This function combines several lower-level visualization functions:
    1. Creates a figure and axis with appropriate settings
    2. Draws 2D head outlines using plot_2d_topomap_inter()
    3. Plots sensors using plot_sensors_2d_inter()
    4. Plots connectivity links using plot_links_2d_inter()
    
    This function uses simpler 2D head outlines compared to viz_2D_headmodel_inter()
    which uses a projected 3D model. The 2D outlines are computationally lighter
    and may be preferred for quick visualizations or publications.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create random connectivity matrix
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity = np.random.rand(n_channels_1, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create visualization
    >>> ax = viz_2D_topomap_inter(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     connectivity,
    ...     threshold=0.7,
    ...     steps=15,
    ...     lab=True
    ... )
    >>> # Add a title
    >>> ax.set_title('Inter-brain connectivity')
    >>> plt.show()
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


def viz_2D_topomap_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, 
                       threshold: float=0.95, steps: int=2, lab: bool = False, 
                       children: bool=False, child_head: bool=False):
    """
    Create a 2D topographic map visualization with intra-brain connectivity.
    
    This function provides a high-level interface for creating a 2D topographic
    visualization of two head outlines side by side, displaying intra-brain
    connectivity within each participant separately.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C1 : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_1) containing
        connectivity values between pairs of electrodes for participant 1
        
    C2 : np.ndarray
        Connectivity matrix with shape (n_channels_2, n_channels_2) containing
        connectivity values between pairs of electrodes for participant 2
        
    threshold : float, optional
        Threshold for displaying connections (default=0.95)
        Only connections with absolute value above this threshold are displayed
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=2)
        If steps < 3, straight lines are drawn instead of curves
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the visualization
    
    Notes
    -----
    This function combines several lower-level visualization functions:
    1. Creates a figure and axis with appropriate settings
    2. Draws 2D head outlines using plot_2d_topomap_intra()
    3. Plots sensors using plot_sensors_2d_intra()
    4. Plots connectivity links using plot_links_2d_intra()
    
    Unlike viz_2D_topomap_inter() which visualizes connections between
    participants, this function displays connections within each participant's
    brain separately, allowing for comparison of intra-brain connectivity patterns.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create random connectivity matrices
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity_1 = np.random.rand(n_channels_1, n_channels_1) * 2 - 1  # Values between -1 and 1
    >>> connectivity_2 = np.random.rand(n_channels_2, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create visualization
    >>> ax = viz_2D_topomap_intra(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     connectivity_1,
    ...     connectivity_2,
    ...     threshold=0.7,
    ...     steps=10,
    ...     lab=True
    ... )
    >>> # Add a title
    >>> ax.set_title('Intra-brain connectivity comparison')
    >>> plt.show()
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


def viz_3D_inter(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, 
                threshold: float=0.95, steps: int=10, lab: bool = False, 
                children: bool=False, child_head: bool=False):
    """
    Create a complete 3D visualization with inter-brain connectivity.
    
    This function provides a high-level interface for creating a 3D visualization
    of two head models with sensors and inter-brain connectivity links.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_2) containing
        connectivity values between all pairs of electrodes across participants
        
    threshold : float, optional
        Threshold for displaying connections (default=0.95)
        Only connections with absolute value above this threshold are displayed
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        If steps < 3, straight lines are drawn instead of curves
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib 3D Axes object containing the visualization
    
    Notes
    -----
    This function combines several lower-level visualization functions:
    1. Creates a 3D figure and axis with appropriate settings
    2. Gets 3D head model vertices and faces using get_3d_heads_inter()
    3. Plots 3D head models using plot_3d_heads()
    4. Plots sensors using plot_sensors_3d_inter()
    5. Plots connectivity links using plot_links_3d_inter()
    
    The 3D visualization allows for interactive rotation and zoom, providing
    a more comprehensive view of the connectivity patterns compared to
    2D visualizations.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create random connectivity matrix
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity = np.random.rand(n_channels_1, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create 3D visualization
    >>> ax = viz_3D_inter(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     connectivity,
    ...     threshold=0.7,
    ...     steps=15,
    ...     lab=False
    ... )
    >>> plt.show()
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


def viz_3D_intra(epo1: mne.Epochs, epo2: mne.Epochs, C1: np.ndarray, C2: np.ndarray, 
                threshold: float=0.95, steps: int=10, lab: bool = False, 
                children: bool=False, child_head: bool=False):
    """
    Create a complete 3D visualization with intra-brain connectivity.
    
    This function provides a high-level interface for creating a 3D visualization
    of two head models side by side, displaying intra-brain connectivity
    within each participant separately.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Epochs object for participant 1, containing channel information
        
    epo2 : mne.Epochs
        Epochs object for participant 2, containing channel information
        
    C1 : np.ndarray
        Connectivity matrix with shape (n_channels_1, n_channels_1) containing
        connectivity values between pairs of electrodes for participant 1
        
    C2 : np.ndarray
        Connectivity matrix with shape (n_channels_2, n_channels_2) containing
        connectivity values between pairs of electrodes for participant 2
        
    threshold : float, optional
        Threshold for displaying connections (default=0.95)
        Only connections with absolute value above this threshold are displayed
        
    steps : int, optional
        Number of steps for bezier curve interpolation (default=10)
        If steps < 3, straight lines are drawn instead of curves
        
    lab : bool, optional
        Whether to plot channel labels (default=False)
        
    children : bool, optional
        Whether to apply transformations for child head models (default=False)
        
    child_head : bool, optional
        If True and children=True, treats participant 2 as a child
        If False and children=True, treats participant 1 as a child
        (default=False)
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib 3D Axes object containing the visualization
    
    Notes
    -----
    This function combines several lower-level visualization functions:
    1. Creates a 3D figure and axis with appropriate settings
    2. Gets 3D head model vertices and faces using get_3d_heads_intra()
    3. Plots 3D head models using plot_3d_heads()
    4. Plots sensors using plot_sensors_3d_intra()
    5. Plots connectivity links using plot_links_3d_intra()
    
    Unlike viz_3D_inter() which visualizes connections between participants,
    this function displays connections within each participant's brain separately,
    allowing for comparison of intra-brain connectivity patterns in 3D.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create random connectivity matrices
    >>> n_channels_1 = len(epochs_subj1.ch_names)
    >>> n_channels_2 = len(epochs_subj2.ch_names)
    >>> connectivity_1 = np.random.rand(n_channels_1, n_channels_1) * 2 - 1  # Values between -1 and 1
    >>> connectivity_2 = np.random.rand(n_channels_2, n_channels_2) * 2 - 1  # Values between -1 and 1
    >>> # Create 3D visualization
    >>> ax = viz_3D_intra(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     connectivity_1,
    ...     connectivity_2,
    ...     threshold=0.7,
    ...     steps=15,
    ...     lab=False
    ... )
    >>> plt.show()
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
    Plot the results of Cross-Wavelet Transform (XWT) analysis.
    
    This function visualizes time-frequency representations of cross-wavelet
    transform results between two participants' EEG signals. It can display
    phase angles, power, or wavelet coherence.
    
    Parameters
    ----------
    sig1 : mne.Epochs
        EEG data for the first participant
        
    sig2 : mne.Epochs
        EEG data for the second participant
        
    sfreq : int or float
        Sampling frequency of the EEG data in Hz
        
    freqs : int, float, or np.ndarray
        Frequencies of interest in Hz
        
    time : int
        Time duration of the sample in seconds
        
    analysis : str
        Type of analysis to visualize, options:
        - 'phase': Phase angle differences
        - 'power': Cross-wavelet power
        - 'wtc': Wavelet coherence
        
    figsize : tuple, optional
        Figure size in inches (default=(30, 8))
        
    tmin : int, optional
        Start time for the plot in seconds (default=0)
        
    x_units : int or float, optional
        Distance between x-axis ticks in data points (default=100)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the visualization
    
    Notes
    -----
    The function:
    1. Calculates the appropriate time and frequency axes for plotting
    2. Computes the cone of influence (COI) to indicate regions with edge effects
    3. Creates a colormap-based visualization of the XWT results
    4. Masks regions outside the COI with hatching to indicate less reliable values
    
    Different colormaps are used based on the analysis type:
    - 'phase': 'hsv' colormap for phase angles
    - 'power': 'viridis' colormap for power values
    - 'wtc': 'plasma' colormap for coherence values
    
    This function is not meant to be called independently but is usually called 
    from higher-level functions like plot_xwt_crosspower or plot_xwt_phase_angle.
    
    Examples
    --------
    >>> # Plot cross-wavelet power
    >>> fig = plot_xwt(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     sfreq=256,
    ...     freqs=np.arange(4, 40, 1),  # 4-40 Hz
    ...     time=10,  # 10 seconds
    ...     analysis='power',
    ...     figsize=(12, 8)
    ... )
    >>> plt.show()
    >>> 
    >>> # Plot wavelet coherence
    >>> fig = plot_xwt(
    ...     epochs_subj1,
    ...     epochs_subj2,
    ...     sfreq=256,
    ...     freqs=np.arange(4, 40, 1),
    ...     time=10,
    ...     analysis='wtc',
    ...     figsize=(12, 8)
    ... )
    >>> plt.show()
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
        data = xwt(sig1, sig2, freqs, mode='phase')
        analysis_title = 'Cross Wavelet Transform (Phase Angle)'
        cbar_title = 'Phase Difference'
        my_cm = matplotlib.colormaps['hsv']

        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='nearest')

    elif analysis == 'power':
        data = xwt(sig1, sig2, freqs, mode='power')
        normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        analysis_title = 'Cross Wavelet Transform (Power)'
        cbar_title = 'Cross Power'
        my_cm = matplotlib.colormaps['viridis']

        plt.imshow(normed_data, aspect='auto', cmap=my_cm,
                   interpolation='lanczos')

    elif analysis == 'wtc':
        data = xwt(sig1, sig2, freqs, mode='wtc')
        analysis_title = 'Wavelet Coherence'
        cbar_title = 'Coherence'
        my_cm = matplotlib.colormaps['plasma']

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