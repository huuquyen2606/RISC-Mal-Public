"""RISC-Mal: Risk-Informed Selective Class-Incremental Learning for Imbalanced Windows Malware Detection.

A modular, reproducible continual learning framework for Windows malware classification.
"""

__version__ = "0.1.0"
__author__ = "RISC-Mal Research Team"

from riscmal import data, evaluation, memory, utils

__all__ = [
    "data",
    "evaluation",
    "memory",
    "utils",
    "__version__",
]
