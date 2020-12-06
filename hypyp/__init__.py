import pkg_resources
from hypyp import analyses, prep, stats, utils, viz

__version__ = pkg_resources.get_distribution("hypyp").version
__all__ = ["analyses", "prep", "stats", "utils", "viz"]
