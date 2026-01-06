from importlib.metadata import version
from hypyp import analyses, prep, stats, utils, viz, analyses_it, analyses_pid

__version__ = version("hypyp")
__all__ = ["analyses", "prep", "stats", "utils", "viz", "fnirs", "ext",
           "analyses_it", "analyses_pid"]
