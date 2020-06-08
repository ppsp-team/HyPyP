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


from copy import copy
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import meshio


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


def plot_sensors_2d(epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = True):
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
    bads_epo1 = []
    bads_epo1 = epo1.info['bads']
    bads_epo2 = []
    bads_epo2 = epo2.info['bads']

    # extract sensor info and transform loc to fit with headmodel
    loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
    loc1 = transform(loc1, traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi/12), rotZ=(-np.pi/2))
    lab1 = [ch for ch in epo1.ch_names]

    loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
    loc2 = transform(loc2, traX=0.17, traY=0, traZ=0.08, rotY=(np.pi/12), rotZ=np.pi/2)
    lab2 = [ch for ch in epo2.ch_names]

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

def plot_links_2d(epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10):
    """
    Plots hyper-connectivity in 2D.

    Arguments:
        epo1: mne.Epochs
          Epochs object to get channels information
        epo2: mne.Epochs
          Epochs object to get channels information
        C: array, (len(loc1), len(loc2))
          matrix with the values of hyper-connectivity
        threshold: float
          threshold for the inter-brain links;
          only those above the set value will be plotted
        steps: int
          number of steps for the Bezier curves
          if <3 equivalent to ploting straight lines
        weight: numpy.float
          Connectivity weight to determine the thickness
          of the link

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

    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

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


def plot_sensors_3d(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, lab: bool = False):
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

def plot_links_3d(ax: str, epo1: mne.Epochs, epo2: mne.Epochs, C: np.ndarray, threshold: float=0.95, steps: int=10):
    """
    Plots hyper-connectivity in 3D.

    Arguments:
        ax: Matplotlib axis created with projection='3d'
        loc1: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        loc2: arrays of shape (n_sensors, 3)
          3d coordinates of the sensors
        C: array, (len(loc1), len(loc2))
          matrix with the values of hyper-connectivity
        threshold: float
          threshold for the inter-brain links;
          only those above the set value will be plotted
        steps: int
          number of steps for the Bezier curves
          if <3 equivalent to ploting straight lines
        weight: numpy.float
          Connectivity weight to determine the thickness
          of the link

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

    cmap_p = matplotlib.cm.get_cmap('Reds')
    norm_p = matplotlib.colors.Normalize(vmin=threshold, vmax=np.nanmax(C[:]))
    cmap_n = matplotlib.cm.get_cmap('Blues_r')
    norm_n = matplotlib.colors.Normalize(vmin=np.min(C[:]), vmax=-threshold)

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
        mne.viz.plot_topomap(T_obs_plot, pos, vmin=vmin, vmax=vmax,
                             sensors=True)

    return None

def get_3d_heads():
    """
    Returns Vertices and Faces of a 3D OBJ representing two facing heads.
    """

    # Extract vertices and faces for the first head
    mesh = meshio.read(os.path.join(os.path.dirname(__file__),
           os.pardir,'data',"Basehead.obj"))
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
    """Plot heads models in 3D.

    Arguments:
        ax : Matplotlib axis created with projection='3d'
        vertices : arrays of shape (V, 3)
            3d coordinates of the vertices
        faces : arrays of shape (F, 4)
            vertices number of face

    Returns:
        None : plot the head faces in 3D within the current axis.
    """
    x_V = vertices[:, 2]
    y_V = vertices[:, 0]
    z_V = vertices[:, 1]
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
