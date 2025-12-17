import importlib.metadata

__version__ = importlib.metadata.version("tinyllm")

from . import cpp_extensions  # type: ignore
from . import tokenize
from . import network
from . import optimize
from . import train

# from .network import layers
# from .network import functional
