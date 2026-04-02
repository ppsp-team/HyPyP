from importlib.metadata import version
from hypyp import analyses, datasets, prep, stats, utils, viz

__version__ = version("hypyp")
__all__ = ["analyses", "datasets", "prep", "stats", "utils", "viz", "fnirs", "ext"]
