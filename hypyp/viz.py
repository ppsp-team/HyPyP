#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : viz.py
# description     : basic visualization functions
# author          : Guillaume Dumas, Amir Djalovski
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt


def transform(locs, traY=0.25, rotZ=np.pi):
    """Calculating new locations for the EEG locations.
    
    Parameters
    ----------
    locs : array of shape (n_sensors, 3)
           3d coordinates of the sensors
    traY : float
        Y translation to apply to the sensors
    rotZ : float
        Z rotation to apply to the sensors

    Returns
    -------
    result : array (n_sensors, 3)
        new 3d coordinates of the sensors
    """
    newX = locs[:, 0] * np.cos(rotZ) - locs[:, 1] * np.sin(rotZ)
    newY = locs[:, 0] * np.sin(rotZ) + locs[:, 1] * np.cos(rotZ)
    locs[:, 0] = newX
    locs[:, 1] = newY
    locs[:, 1] = locs[:, 1] + traY
    return locs


def plot_sensors_2d(loc1, loc2, lab1=[], lab2=[]):
    """Plot sensors in 2D.
    
    Parameters
    ----------
    loc1, loc2 : arrays of shape (n_sensors, 3)
                 3d coordinates of the sensors
    lab1, lab2 : lists of strings
                 sensor labels
    Returns
    -------
    None : plot the sensors in 2D within the current axis.
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


def plot_links_2d(loc1, loc2, C, threshold=0.95, steps=10):
    """Plot hyper-conenctivity in 2D.
    
    Parameters
    ----------
    loc1, loc2 : arrays of shape (n_sensors, 3)
                 3d coordinates of the sensors
    C : array, (len(loc1), len(loc2))
        matrix with the values of hyper-connectivity
    threshold : float
                threshold for the links
                only those above will be ploted
    steps : int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines           

    Returns
    -------
    None : plot the links in 2D within the current axis.
    """
    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)

    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            if C[e1, e2] >= threshold:
                if steps <= 2:
                    plt.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             '-', color='black')
                else:
                    alphas = np.linspace(0, 1, steps)
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
                                 '-', color='black')


def plot_sensors_3d(ax, loc1, loc2, lab1=[], lab2=[]):
    """Plot sensors in 3D.
    
    Parameters
    ----------
    loc1, loc2 : arrays of shape (n_sensors, 3)
                 3d coordinates of the sensors
    lab1, lab2 : lists of strings
                 sensor labels
    Returns
    -------
    None : plot the sensors in 3D within the current axis.
    """
    for idx1 in range(len(loc1)):
            x1, y1, z1 = loc1[idx1, :]
            ax.scatter(x1, y1, z1, marker='o', color='blue')
            if lab1:
                plt.text(x1, y1 ,z1, lab1[idx1],
                        horizontalalignment='center',
                        verticalalignment='center')

    for idx2 in range(len(loc2)):
        x2, y2, z2 = loc2[idx2, :]
        ax.scatter(x2, y2, z2, marker='o', color='red')
        if lab2:
            plt.text(x2, y2, z2, lab2[idx2],
                    horizontalalignment='center',
                    verticalalignment='center')


def plot_links_3d(loc1, loc2, C, threshold=0.95, steps=10):
    """Plot hyper-conenctivity in 3D.
    
    Parameters
    ----------
    loc1, loc2 : arrays of shape (n_sensors, 3)
                 3d coordinates of the sensors
    C : array, (len(loc1), len(loc2))
        matrix with the values of hyper-connectivity
    threshold : float
                threshold for the links
                only those above will be ploted
    steps : int
            number of steps for the Bezier curves
            if <3 equivalent to ploting straight lines           

    Returns
    -------
    None : plot the links in 3D within the current axis.
    """
    ctr1 = np.nanmean(loc1, 0)
    ctr2 = np.nanmean(loc2, 0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for e1 in range(len(loc1)):
        x1 = loc1[e1, 0]
        y1 = loc1[e1, 1]
        z1 = loc1[e1, 2]
        for e2 in range(len(loc2)):
            x2 = loc2[e2, 0]
            y2 = loc2[e2, 1]
            z1 = loc2[e1, 2]
            if C[e1, e2] >= threshold:
                if steps <= 2:
                    ax.plot([loc1[e1, 0], loc2[e2, 0]],
                             [loc1[e1, 1], loc2[e2, 1]],
                             [loc1[e1, 2], loc2[e2, 2]],
                             '-', color='black')
                else:
                    alphas = np.linspace(0, 1, steps)
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
                              3 * (1-a)**2 * a * (2 * z1 - ctr1[1]) +
                              3 * (1-a) * a**2 * (2 * z2 - ctr2[1]) +
                              a**3 * z2)
                        znn = ((1-b)**3 * z1 +
                               3 * (1-b)**2 * b * (2 * z1 - ctr1[1]) +
                               3 * (1-b) * b**2 * (2 * z2 - ctr2[1]) +
                               b**3 * z2)
                        ax.plot([xn, xnn], [yn, ynn], [zn, znn],
                                 '-', color='black')