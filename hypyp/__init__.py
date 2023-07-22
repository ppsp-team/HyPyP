from importlib.metadata import version
from hypyp import analyses, prep, stats, utils, viz

__version__ = version("hypyp")
__all__ = ["analyses", "prep", "stats", "utils", "viz"]
