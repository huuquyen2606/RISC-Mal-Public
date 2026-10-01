"""Strategy registry and factory for Continual Learning methods."""

from typing import Any, Dict, Type
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

STRATEGY_REGISTRY: Dict[str, Type[BaseContinualStrategy]] = {
    "baseline": BaselineStrategy,
    "ewc": EWCStrategy,
    "lwf": LwFStrategy,
    "lgr": LGRStrategy,
    "bir": BIRStrategy,
    "bi-r": BIRStrategy,
    "gc": GCStrategy,
    "der": DERStrategy,
    "foster": FOSTERStrategy,
    "riscmal": RISCMalStrategy,
    "risc-mal": RISCMalStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> BaseContinualStrategy:
    """Factory function instantiating a Continual Learning strategy by name.
    
    Args:
        name: Strategy name (case-insensitive).
        **kwargs: Strategy-specific hyperparameters.
        
    Returns:
        Instance of BaseContinualStrategy subclass.
    """
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}")

    strategy_cls = STRATEGY_REGISTRY[key]
    return strategy_cls(**kwargs)
