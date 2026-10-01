"""Continual Learning strategies for class-incremental malware detection."""

from .base import BaseContinualStrategy
from .baseline import BaselineStrategy
from .ewc import EWCStrategy
from .lwf import LwFStrategy
from .lgr import LGRStrategy
from .bir import BIRStrategy
from .gc import GCStrategy
from .der import DERStrategy
from .foster import FOSTERStrategy
from .riscmal import RISCMalStrategy
from .registry import STRATEGY_REGISTRY, get_strategy

__all__ = [
    "BaseContinualStrategy",
    "BaselineStrategy",
    "EWCStrategy",
    "LwFStrategy",
    "LGRStrategy",
    "BIRStrategy",
    "GCStrategy",
    "DERStrategy",
    "FOSTERStrategy",
    "RISCMalStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
