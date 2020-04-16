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


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne


def transform(locs: np.ndarray, traY: float=0.25, rotZ: float=np.pi):
    """
    Calculates new locations for the EEG locations.

    Arguments:
        locs: array of shape (n_sensors, 3)
          3d coordinates of the sensors
        traY: float
          Y translation to apply to the sensors
        rotZ: float
          Z rotation to apply to the sensors

    Returns:
        result: array (n_sensors, 3)
          new 3d coordinates of the sensors
    """
    newX = locs[:, 0] * np.cos(rotZ) - locs[:, 1] * np.sin(rotZ)
    newY = locs[:, 0] * np.sin(rotZ) + locs[:, 1] * np.cos(rotZ)
    locs[:, 0] = newX
    locs[:, 1] = newY
    locs[:, 1] = locs[:, 1] + traY
    return locs


def plot_sensors_2d(loc1: np.ndarray, loc2: np.ndarray, lab1: list=[], lab2: list=[]):
    """
    Plots sensors in 2D.

    Arguments:
        loc1: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        loc2: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        lab1: lists of strings
          sensor labels
        lab2: lists of strings
          sensor labels

    Returns:
        None: plot the sensors in 2D within the current axis.
    """
    for idx1 in range(len(loc1)):
        x1, y1, z1 = loc1[idx1, :]
        plt.plot(x1, y1, marker='o', color='blue')
        if lab1:
            plt.text(x1, y1, lab1[idx1],
                     horizontalalignment='center',
                     verticalalignment='center')
    for idx2 in range(len(loc2)):
        x2, y2, z2 = loc2[idx2, :]
        plt.plot(x2, y2, marker='o', color='red')
        if lab2:
            plt.text(x2, y2, lab2[idx2],
                     horizontalalignment='center',
                     verticalalignment='center')


def plot_links_2d(loc1: np.ndarray, loc2: np.ndarray, C: np.ndarray, threshold: float=0.95, steps: int=10):
    """
    Plots hyper-conenctivity in 2D.

    Arguments:
        loc1: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        loc2: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        C: array, (len(loc1), len(loc2))
          matrix with the values of hyper-connectivity
        threshold: float
          threshold for the links
          only those above will be ploted
        steps: int
          number of steps for the Bezier curves
          if <3 equivalent to ploting straight lines
        weight: numpy.float
          Connectivity weight to determine the thickness
          of the link

    Returns:
        None: plot the links in 2D within the current axis.
    """
    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)

    cmap = matplotlib.cm.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(vmin=threshold, vmax=np.max(C[:]))


    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            color = cmap(norm(C[e1, e2]))  
            if C[e1, e2] >= threshold:
                if steps <= 2:
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.max(C[:]-threshold)))
                    plt.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             '-', color=color, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.max(C[:]-threshold)))
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
                                 '-', color=color, linewidth=weight)


def plot_sensors_3d(ax: str, loc1: np.ndarray, loc2: np.ndarray, lab1: list=[], lab2: list=[]):
    """
    Plots sensors in 3D.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        loc1: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        loc2: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        lab1: lists of strings
          sensor labels
        lab2: lists of strings
          sensor labels

    Returns:
        None: plot the sensors in 3D within the current axis.
    """
    for idx1 in range(len(loc1)):
            x1, y1, z1 = loc1[idx1, :]
            ax.scatter(x1, y1, z1, marker='o', color='blue')
            if lab1:
                ax.text(x1, y1 ,z1, lab1[idx1],
                        horizontalalignment='center',
                        verticalalignment='center')

    for idx2 in range(len(loc2)):
        x2, y2, z2 = loc2[idx2, :]
        ax.scatter(x2, y2, z2, marker='o', color='red')
        if lab2:
            ax.text(x2, y2, z2, lab2[idx2],
                    horizontalalignment='center',
                    verticalalignment='center')


def plot_links_3d(ax: str, loc1: np.ndarray, loc2: np.ndarray, C: np.ndarray, threshold: float=0.95, steps: int=10):
    """
    Plots hyper-conenctivity in 3D.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        loc1: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        loc2: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        C: array, (len(loc1), len(loc2))
          matrix with the values of hyper-connectivity
        threshold: float
          threshold for the links
          only those above will be ploted
        steps: int
          number of steps for the Bezier curves
          if <3 equivalent to ploting straight lines
        weight: numpy.float
          Connectivity weight to determine the thickness
          of the link

    Returns:
        None: plot the links in 3D within the current axis.
    """
    ctr1 = np.nanmean(loc1, 0)
    ctr1[2] -= 0.2
    ctr2 = np.nanmean(loc2, 0)
    ctr2[2] -= 0.2

    cmap = matplotlib.cm.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(vmin=threshold, vmax=np.max(C[:]))

    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        z1 = loc1[e1, 2]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            z2 = loc2[e2, 2]
            color = cmap(norm(C[e1, e2])) 
            if C[e1, e2] >= threshold:
                if steps <= 2:
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.max(C[:]-threshold)))
                    ax.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             [loc1[e1, 2], loc2[e2, 2]],
                             '-', color=color, linewidth=weight)
                else:
                    alphas = np.linspace(0, 1, steps)
                    weight = 0.2 +1.6*((C[e1, e2]-threshold)/(np.max(C[:]-threshold)))
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
                                 '-', color=color, linewidth=weight)


def plot_significant_sensors(T_obs_plot: np.ndarray, epochs: mne.Epochs):
    """
    Plots the significant sensors from a statistical test (simple t test or
    clusters corrected t test) computed between groups or conditions, on power
    or connectivity values, across simple subjects. For satistics with
    interbrains connectivity values on dyads (merge data), use the
    plot_links_3d function of the toolbox.

    Arguments:
        T_obs_plot: satistical values to plot, from sensors above alpha threshold,
          array of shape (n_tests,).
        epochs: one subject Epochs object to sample channels information in info.

    Returns:
        None, plots topomap with the T or F statistics for significant sensors.
    """

    # getting sensors position
    pos = np.array([[0, 0]])
    for i in range(0, len(epochs.info['ch_names'])):
        cor = np.array([epochs.info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis = 0)
    pos = pos[1:]
    # topoplot of significant sensors
    if np.max(np.abs(T_obs_plot)) != 0:
        vmax = np.max(np.abs(T_obs_plot))
        vmin = -vmax
    else:
        vmax = None
        vmin = None
        mne.viz.plot_topomap(T_obs_plot, pos, vmin=vmin, vmax=vmax,
                             sensors=True)

    return None
