from importlib.metadata import version
from hypyp import analyses, prep, stats, utils, viz, fnirs, eeg, multimodal, ext, signal

__version__ = version("hypyp")
__all__ = ["analyses", "prep", "stats", "utils", "viz", "fnirs", "eeg", "multimodal", "ext", "signal"]
