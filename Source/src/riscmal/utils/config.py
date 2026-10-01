"""Declarative YAML and JSON configuration parser and schema validator.

Loads protocol, task, and method-specific hyperparameter configurations
with comprehensive schema validation and sensible defaults.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import os
import yaml


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file from disk.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML parsing fails or file is empty.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML file {config_path}: {exc}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to be a dictionary, got {type(config).__name__}")
    return config


def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Saves a configuration dictionary to disk in YAML format.

    Args:
        config: Configuration dictionary to serialize.
        config_path: Destination path for the YAML file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON configuration file from disk.

    Args:
        config_path: Path to the JSON file.

    Returns:
        Dictionary containing the parsed JSON structure.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], config_path: Union[str, Path], indent: int = 2) -> None:
    """Saves a configuration dictionary to disk in JSON format.

    Args:
        config: Dictionary to serialize.
        config_path: Destination path for the JSON file.
        indent: Indentation spaces for pretty printing.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=indent)


def merge_configs(base: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deep merges two configuration dictionaries without mutating the inputs.

    Args:
        base: Base dictionary containing default parameters.
        override: Optional dictionary containing overridden values.

    Returns:
        Merged configuration dictionary.
    """
    merged = dict(base)
    if not override:
        return merged

    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged


def validate_protocol_config(config: Dict[str, Any]) -> bool:
    """Validates the structure of the global incremental learning protocol configuration.

    Args:
        config: Configuration dictionary loaded from protocol.yaml.

    Returns:
        True if the configuration satisfies schema requirements.

    Raises:
        ValueError: If mandatory parameters are missing or invalid.
    """
    required_keys = ["seeds", "tasks", "training", "evaluation"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Protocol configuration is missing required top-level keys: {missing}")

    if not isinstance(config["seeds"], list) or len(config["seeds"]) == 0:
        raise ValueError("Protocol config 'seeds' must be a non-empty list of integers.")

    if not isinstance(config["tasks"], (list, dict)):
        raise ValueError("Protocol config 'tasks' must define task split sequence.")

    training_cfg = config.get("training", {})
    if "batch_size" in training_cfg and training_cfg["batch_size"] <= 0:
        raise ValueError("Protocol config 'training.batch_size' must be positive.")

    return True


def validate_method_config(method_name: str, config: Dict[str, Any]) -> bool:
    """Validates method-specific hyperparameters against expected strategy schemas.

    Args:
        method_name: Name of continual learning strategy (e.g. 'riscmal', 'der', 'ewc').
        config: Hyperparameter dictionary.

    Returns:
        True if valid.

    Raises:
        ValueError: If required hyperparameters are missing or have invalid ranges.
    """
    m_name = method_name.lower().replace("-", "").replace("_", "")

    if m_name in ("riscmal", "risc_mal"):
        if "alpha" in config and not (0.0 <= config["alpha"] <= 1.0):
            raise ValueError(f"RISC-Mal alpha must be in [0.0, 1.0], got {config['alpha']}")
        if "replay_ratio" in config and not (0.0 < config["replay_ratio"] <= 1.0):
            raise ValueError(f"RISC-Mal replay_ratio must be in (0.0, 1.0], got {config['replay_ratio']}")

    elif m_name == "ewc":
        if "lambda" in config and config["lambda"] < 0:
            raise ValueError(f"EWC lambda penalty must be non-negative, got {config['lambda']}")

    elif m_name == "lwf":
        if "lambda" in config and config["lambda"] < 0:
            raise ValueError(f"LwF lambda penalty must be non-negative, got {config['lambda']}")
        if "temperature" in config and config["temperature"] <= 0:
            raise ValueError(f"LwF temperature must be positive, got {config['temperature']}")

    return True
