# Draft

## Underliying libraries

HyPyP supports both MNE and Cedalion for loading and preprocessing fNIRS data. Since Cedalion is not yet on PyPI, it is an optional dependency and must be installed manually.

### Use optional Cedalion preprocessing

`poetry shell`  
`pip install pint-xarray trimesh vtk pyvista strenum nibabel click trame`  
`cd /path/to/cedalion`  
`pip install -e .`

To test installation, open and run the Cedalion `examples/00_test_installation.ipynb` notebook. Don't forget to use the same poetry virtual environment.
