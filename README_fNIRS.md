# Draft

## Underliying libraries

HyPyP supports both MNE and Cedalion for loading and preprocessing fNIRS data. Since Cedalion is not yet on PyPI, it is an optional dependency and must be installed manually.

### Use optional Cedalion preprocessing

`poetry shell`  
`pip install pint-xarray trimesh vtk pyvista strenum nibabel click trame`  
`cd /path/to/cedalion`  
`pip install -e .`

To test installation, open and run the Cedalion `examples/00_test_installation.ipynb` notebook. Don't forget to use the same poetry virtual environment.

### Use optional pycwt wavelets

`poetry run pip install pycwt`

### Use optional matlab engine wavelets

To install matlab engine, you need to specify the version that matches the matlab installation. Also, depending on the installation, you might need to specify where to find the matlab executables

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/matlab/R2023b/bin/glnxa64 poetry run pip install matlabengine==23.2.3
```
