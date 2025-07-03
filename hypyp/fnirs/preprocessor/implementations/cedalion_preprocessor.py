# This Cedalion Preprocessor is a sandbox and is provided as inspiration only for further development
# See https://openfnirs.org/training/cedalion/ for more info about the Cedalion project
#
# Here is how to have cedalion available as a python package for this project
#   poetry shell
#   pip install pint-xarray trimesh vtk pyvista strenum nibabel click trame
#   mkdir /path/where/you/want/to/install
#   cd /path/where/you/want/install
#   git clone https://github.com/ibs-lab/cedalion
#   cd cedalion
#   pip install -e .
#

try:
    import xarray as xr
    import cedalion
    import cedalion.dataclasses as cdc
    from cedalion.typing import NDTimeSeries
    import matplotlib.pyplot as plt

    from ...data_browser import DataBrowser
    from ..base_step import *
    from ..base_preprocessor import BasePreprocessor

    class CedalionStep(BaseStep[cdc.Recording]):
        @property
        def n_times(self):
            return len(self.obj['time'])
            
        @property
        def sfreq(self):
            # TODO We probably have this information directly
            return 1 / float(self.obj['time'][1] - self.obj['time'][0])
            
        @property
        def ch_names(self):
            # TODO learn how to use xarray efficiently
            # TODO maybe we should send channel and  wavelength
            return list(self.obj.channel.to_series())
            
        def plot(self, **_kwargs):

            # select which time series we work with
            ts = self.obj

            # Thanks to the xarray DataArray structure, we can easily select the data we want to plot
            # plot four channels and their stim markers
            f, ax = plt.subplots(len(self.ch_names), 1, sharex=True)
            for i, ch in enumerate(self.ch_names):
                ax[i].plot(ts.time, ts.sel(channel=ch, chromo="HbO"), "r-", label="HbO")
                ax[i].plot(ts.time, ts.sel(channel=ch, chromo="HbR"), "b-", label="HbR")
                ax[i].set_title(f"Ch. {ch}")
                # add stim markers using Cedalion's plot_stim_markers function
                #cedalion.plots.plot_stim_markers(ax[i], rec.stim, y=1)
                ax[i].set_ylabel(r"$\Delta$ c / uM")

            ax[0].legend(ncol=6)
            ax[3].set_label("time / s")
            ax[3].set_xlim(0,100)
            #plt.tight_layout()
            return f

    class CedalionPreprocessor(BasePreprocessor[cdc.Recording]):
        def read_file(self, path, verbose:bool=False) -> cdc.Recording:
            # TODO honor verbose arg
            if not DataBrowser.is_path_snirf(path):
                raise RuntimeError('Not implemented: only snirf file is supported for now')

            recordings = cedalion.io.read_snirf(path)
            rec: cdc.Recording  = recordings[0]

            # TODO we should force units only if they are not set
            rec['amp'] = rec['amp'].pint.dequantify().pint.quantify("V")
            rec['amp']['time'] = rec['amp']['time'].pint.dequantify().pint.quantify("second")
            return rec

        def run(self, rec, verbose: bool = False) -> list[CedalionStep]:
            # TODO honor verbose
            steps = []
            amp = rec['amp']
            steps.append(CedalionStep(amp, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC))

            od = cedalion.nirs.int2od(amp)
            steps.append(CedalionStep(od, PREPROCESS_STEP_OD_KEY, PREPROCESS_STEP_OD_DESC))


            dpf = xr.DataArray(
                [6., 6.],
                dims="wavelength",
                coords={"wavelength" : [760., 850.]}) # TODO unhardcode wavelengts

            haemo = cedalion.nirs.beer_lambert(rec['amp'], rec.geo3d, dpf)
            steps.append(CedalionStep(haemo, PREPROCESS_STEP_HAEMO_KEY, PREPROCESS_STEP_HAEMO_DESC))

            return steps
            

except: 
    CedalionPreprocesStep = None
    CedalionPreprocessor = None