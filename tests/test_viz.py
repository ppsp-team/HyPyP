#!/usr/bin/env python
# coding=utf-8

import pytest
import numpy as np
import matplotlib.pyplot as plt
import mne
from hypyp import viz
from hypyp import utils


def test_child_head_transform():
    """Test that the child head transformations work correctly"""
    # Create a sample array of coordinates
    locs = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    
    # Transform with adult parameters
    adult_locs = viz.transform(locs.copy())
    
    # Transform with child parameters
    child_locs = viz.transform(locs.copy(), children=True, child_head=True)
    
    # Verify that child coordinates are scaled appropriately
    # Child head should be smaller than adult head
    assert np.linalg.norm(child_locs[1] - child_locs[0]) < np.linalg.norm(adult_locs[1] - adult_locs[0])


def test_transform_2d_intra():
    """Test the transform_2d_intra function scaling for child heads"""
    # Create a sample array of coordinates
    locs = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    
    # Transform with adult parameters
    adult_locs = viz.transform_2d_intra(locs.copy())
    
    # Transform with child parameters
    child_locs = viz.transform_2d_intra(locs.copy(), children=True, child_head=True)
    
    # Verify that child coordinates are scaled appropriately
    # Child head should be smaller than adult head
    assert np.linalg.norm(child_locs[2]) < np.linalg.norm(adult_locs[2])


def test_bezier_interpolation():
    """Test the bezier_interpolation function both with and without second control point"""
    # Test parameters
    t = 0.5  # Interpolation parameter
    p0 = 0.0  # Starting point
    p1 = 1.0  # Ending point
    c0 = 0.3  # Control point for p0
    c1 = 0.7  # Control point for p1
    
    # Test with both control points
    result_with_c1 = viz.bezier_interpolation(t, p0, p1, c0, c1)
    
    # Test with only one control point (the second one should be ignored)
    result_without_c1 = viz.bezier_interpolation(t, p0, p1, c0)
    
    # They should produce different results
    assert result_with_c1 != result_without_c1
    
    # The result should be between the start and end points
    assert 0.0 <= result_with_c1 <= 1.0
    assert 0.0 <= result_without_c1 <= 1.0


def test_get_3d_heads_inter():
    """Test the get_3d_heads_inter function for child head models"""
    # Get 3D heads with normal parameters
    vertices_adult, faces_adult = viz.get_3d_heads_inter()
    
    # Get 3D heads with child parameters
    vertices_child, faces_child = viz.get_3d_heads_inter(children=True, child_head=True)
    
    # Check that the faces have the same shape for both (same mesh topology)
    assert faces_adult.shape == faces_child.shape
    
    # Check that the vertices have the same number of points
    assert vertices_adult.shape == vertices_child.shape
    
    # But the child model should have smaller overall size
    adult_size = np.max(vertices_adult) - np.min(vertices_adult)
    child_size = np.max(vertices_child) - np.min(vertices_child)
    
    # The difference should be small but noticeable
    size_ratio = child_size / adult_size
    assert 0.85 <= size_ratio <= 0.99  # Not exact due to transformations


def test_get_3d_heads_intra():
    """Test the get_3d_heads_intra function for child head models"""
    # Get 3D heads with normal parameters
    vertices_adult, faces_adult = viz.get_3d_heads_intra()
    
    # Get 3D heads with child parameters
    vertices_child, faces_child = viz.get_3d_heads_intra(children=True, child_head=True)
    
    # Check that the faces have the same shape for both (same mesh topology)
    assert faces_adult.shape == faces_child.shape
    
    # Check that the vertices have the same number of points
    assert vertices_adult.shape == vertices_child.shape
    
    # But the child model should have smaller overall size for the second head
    # Extract vertices for the second head (using the face indices)
    second_head_adult_vertices = vertices_adult[len(vertices_adult)//2:]
    second_head_child_vertices = vertices_child[len(vertices_child)//2:]
    
    adult_size = np.max(second_head_adult_vertices) - np.min(second_head_adult_vertices)
    child_size = np.max(second_head_child_vertices) - np.min(second_head_child_vertices)
    
    # The difference should be small but noticeable
    size_ratio = child_size / adult_size
    assert 0.85 <= size_ratio <= 0.99  # Not exact due to transformations


def test_plot_2d_topomap_inter(mocker):
    """Test the plot_2d_topomap_inter function with child head parameter"""
    # Mock the matplotlib add_patch to avoid actual plotting
    mocker.patch.object(plt.Axes, 'add_patch')
    
    # Create a figure and ax for testing
    fig, ax = plt.subplots()
    
    # Call the function with adult parameters
    viz.plot_2d_topomap_inter(ax)
    
    # Call the function with child parameters
    viz.plot_2d_topomap_inter(ax, children=True)
    
    # Since we're mocking add_patch, we can just verify it was called enough times 
    # (we expect more calls for the child version due to the different sized heads)
    assert plt.Axes.add_patch.call_count >= 8  # At least 8 calls (2 heads × 4 components)


def test_plot_2d_topomap_intra(mocker):
    """Test the plot_2d_topomap_intra function with child head parameter"""
    # Mock the matplotlib add_patch to avoid actual plotting
    mocker.patch.object(plt.Axes, 'add_patch')
    
    # Create a figure and ax for testing
    fig, ax = plt.subplots()
    
    # Call the function with adult parameters
    viz.plot_2d_topomap_intra(ax)
    
    # Call the function with child parameters
    viz.plot_2d_topomap_intra(ax, children=True)
    
    # Since we're mocking add_patch, we can just verify it was called enough times
    assert plt.Axes.add_patch.call_count >= 8  # At least 8 calls (2 heads × 4 components)


def test_viz_2D_topomap_inter(epochs, mocker):
    """Test that viz_2D_topomap_inter runs with child head parameters"""
    # Mock plt.show to avoid displaying the figure
    mocker.patch('matplotlib.pyplot.show')
    # Mock the underlying plot functions to prevent actual plotting
    mocker.patch('hypyp.viz.plot_2d_topomap_inter')
    mocker.patch('hypyp.viz.plot_sensors_2d_inter')
    mocker.patch('hypyp.viz.plot_links_2d_inter')
    
    # Create some dummy connectivity data
    C = np.random.rand(len(epochs.epo1.info['ch_names']), len(epochs.epo2.info['ch_names']))
    
    # Call the function with adult parameters
    viz.viz_2D_topomap_inter(epochs.epo1, epochs.epo2, C)
    
    # Call the function with child parameters
    viz.viz_2D_topomap_inter(epochs.epo1, epochs.epo2, C, 
                            children=True, child_head=True)
    
    # Check the functions were called
    assert viz.plot_2d_topomap_inter.call_count == 2
    assert viz.plot_sensors_2d_inter.call_count == 2
    assert viz.plot_links_2d_inter.call_count == 2
    
    # Check that the children parameter was passed correctly
    viz.plot_2d_topomap_inter.assert_any_call(mocker.ANY, children=True)
    viz.plot_sensors_2d_inter.assert_any_call(mocker.ANY, mocker.ANY, 
                                             lab=False, children=True, child_head=True)
    viz.plot_links_2d_inter.assert_any_call(mocker.ANY, mocker.ANY, C=C,
                                          threshold=0.95, steps=10, 
                                          children=True, child_head=True)


def test_viz_2D_topomap_intra(epochs, mocker):
    """Test that viz_2D_topomap_intra runs with child head parameters"""
    # Mock plt.show to avoid displaying the figure
    mocker.patch('matplotlib.pyplot.show')
    # Mock the underlying plot functions to prevent actual plotting
    mocker.patch('hypyp.viz.plot_2d_topomap_intra')
    mocker.patch('hypyp.viz.plot_sensors_2d_intra')
    mocker.patch('hypyp.viz.plot_links_2d_intra')
    
    # Create some dummy connectivity data
    n_channels = len(epochs.epo1.info['ch_names'])
    C1 = np.random.rand(n_channels, n_channels)
    C2 = np.random.rand(n_channels, n_channels)
    
    # Call the function with adult parameters
    viz.viz_2D_topomap_intra(epochs.epo1, epochs.epo2, C1, C2)
    
    # Call the function with child parameters
    viz.viz_2D_topomap_intra(epochs.epo1, epochs.epo2, C1, C2,
                            children=True, child_head=True)
    
    # Check the functions were called
    assert viz.plot_2d_topomap_intra.call_count == 2
    assert viz.plot_sensors_2d_intra.call_count == 2
    assert viz.plot_links_2d_intra.call_count == 2
    
    # Check that the children parameter was passed correctly
    viz.plot_2d_topomap_intra.assert_any_call(mocker.ANY, children=True)
    viz.plot_sensors_2d_intra.assert_any_call(mocker.ANY, mocker.ANY, 
                                            lab=False, children=True, child_head=True)
    viz.plot_links_2d_intra.assert_any_call(mocker.ANY, mocker.ANY, C1=C1, C2=C2,
                                          threshold=0.95, steps=2, 
                                          children=True, child_head=True)


def test_viz_3D_inter(epochs, mocker):
    """Test that viz_3D_inter runs with child head parameters"""
    # Mock plt.show to avoid displaying the figure
    mocker.patch('matplotlib.pyplot.show')
    
    # Mock the underlying functions with return values
    mock_vertices = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    mock_faces = np.array([[0, 1, 2, 3]])
    
    # Configure the mocks to return these values
    mocker.patch('hypyp.viz.get_3d_heads_inter', return_value=(mock_vertices, mock_faces))
    mocker.patch('hypyp.viz.plot_3d_heads')
    mocker.patch('hypyp.viz.plot_sensors_3d_inter')
    mocker.patch('hypyp.viz.plot_links_3d_inter')
    
    # Create some dummy connectivity data
    C = np.random.rand(len(epochs.epo1.info['ch_names']), len(epochs.epo2.info['ch_names']))
    
    # Call the function with adult parameters
    viz.viz_3D_inter(epochs.epo1, epochs.epo2, C)
    
    # Call the function with child parameters
    viz.viz_3D_inter(epochs.epo1, epochs.epo2, C, children=True, child_head=True)
    
    # Check the functions were called
    assert viz.get_3d_heads_inter.call_count == 2
    assert viz.plot_3d_heads.call_count == 2
    assert viz.plot_sensors_3d_inter.call_count == 2
    assert viz.plot_links_3d_inter.call_count == 2


def test_viz_3D_intra(epochs, mocker):
    """Test that viz_3D_intra runs with child head parameters"""
    # Mock plt.show to avoid displaying the figure
    mocker.patch('matplotlib.pyplot.show')
    
    # Create mock data for the 3D head model
    mock_vertices = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    mock_faces = np.array([[0, 1, 2, 3]])
    
    # Configure the mocks to return these values
    mocker.patch('hypyp.viz.get_3d_heads_intra', return_value=(mock_vertices, mock_faces))
    mocker.patch('hypyp.viz.plot_3d_heads')
    mocker.patch('hypyp.viz.plot_sensors_3d_intra')
    mocker.patch('hypyp.viz.plot_links_3d_intra')
    
    # Create some dummy connectivity data
    n_channels = len(epochs.epo1.info['ch_names'])
    C1 = np.random.rand(n_channels, n_channels)
    C2 = np.random.rand(n_channels, n_channels)
    
    # Call the function with adult parameters
    viz.viz_3D_intra(epochs.epo1, epochs.epo2, C1, C2)
    
    # Call the function with child parameters
    viz.viz_3D_intra(epochs.epo1, epochs.epo2, C1, C2, children=True, child_head=True)
    
    # Check the functions were called
    assert viz.get_3d_heads_intra.call_count == 2
    assert viz.plot_3d_heads.call_count == 2
    assert viz.plot_sensors_3d_intra.call_count == 2
    assert viz.plot_links_3d_intra.call_count == 2