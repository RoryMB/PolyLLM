from importlib.metadata import version

__version__ = version("polyllm")

from .polyllm import *  # noqa: F403
from .utils import *  # noqa: F403
