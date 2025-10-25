"""
YAML strategy loader.
Maps YAML specification to Python strategy classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.strategies.base import Strategy
from src.strategies.breakout import BreakoutFromYAML, BreakoutStrategy
from src.strategies.earnings_iv_short import EarningsIVFromYAML, EarningsIVShortStrategy
from src.strategies.sma_cross import SMACrossFromYAML, SMACrossStrategy

logger = logging.getLogger(__name__)

# Strategy registry: maps YAML 'type' to Python class
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossStrategy,
    "sma_cross": SMACrossStrategy,
    "breakout": BreakoutStrategy,
    "donchian": BreakoutStrategy,
    "earnings_iv_short": EarningsIVShortStrategy,
    "earnings_iv": EarningsIVShortStrategy,
}

# YAML-driven strategy classes (for parameterized strategies)
YAML_STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossFromYAML,
    "sma_cross": SMACrossFromYAML,
    "breakout": BreakoutFromYAML,
    "donchian": BreakoutFromYAML,
    "earnings_iv_short": EarningsIVFromYAML,
    "earnings_iv": EarningsIVFromYAML,
}


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dict of YAML contents
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    if not spec:
        raise ValueError(f"Empty YAML file: {file_path}")

    return spec


def load_strategy_from_yaml(file_path: str | Path) -> tuple[Strategy, dict[str, Any]]:
    """
    Load strategy from YAML file.

    Args:
        file_path: Path to YAML strategy file

    Returns:
        Tuple of (Strategy instance, full YAML spec dict)
    """
    spec = load_yaml_file(file_path)

    # Extract strategy metadata
    if "strategy" not in spec:
        raise ValueError(f"YAML missing 'strategy' section: {file_path}")

    strategy_section = spec["strategy"]
    strategy_type = strategy_section.get("type", strategy_section.get("name", ""))

    if not strategy_type:
        raise ValueError(f"Strategy type not specified in: {file_path}")

    # Get parameters
    parameters = spec.get("parameters", {})

    # Look up strategy class
    if strategy_type in YAML_STRATEGY_REGISTRY:
        # Use YAML-driven strategy class
        strategy_class = YAML_STRATEGY_REGISTRY[strategy_type]
        strategy = strategy_class(params=spec)  # Pass entire spec
        logger.info(f"Loaded YAML-driven strategy: {strategy_type} from {file_path}")
    elif strategy_type in STRATEGY_REGISTRY:
        # Use hardcoded strategy class with parameters
        strategy_class = STRATEGY_REGISTRY[strategy_type]
        strategy = strategy_class(**parameters)
        logger.info(f"Loaded strategy: {strategy_type} from {file_path} with params {parameters}")
    else:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    return strategy, spec


def validate_yaml_strategy(file_path: str | Path) -> tuple[bool, str]:
    """
    Validate a YAML strategy file.

    Args:
        file_path: Path to YAML strategy file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        spec = load_yaml_file(file_path)

        # Check required sections
        required_sections = ["strategy"]
        for section in required_sections:
            if section not in spec:
                return False, f"Missing required section: {section}"

        # Check strategy metadata
        strategy_section = spec["strategy"]
        if "type" not in strategy_section and "name" not in strategy_section:
            return False, "Strategy section missing 'type' or 'name'"

        # Try to load the strategy (validates type exists)
        strategy_type = strategy_section.get("type", strategy_section.get("name", ""))
        if (
            strategy_type not in STRATEGY_REGISTRY
            and strategy_type not in YAML_STRATEGY_REGISTRY
        ):
            return False, f"Unknown strategy type: {strategy_type}"

        return True, "Valid"

    except Exception as e:
        return False, str(e)


def list_strategies(directory: str | Path = "strategies") -> list[dict[str, Any]]:
    """
    List all YAML strategies in a directory.

    Args:
        directory: Directory to scan for YAML files

    Returns:
        List of dicts with strategy metadata
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Strategy directory not found: {directory}")
        return []

    strategies = []

    for file_path in directory.glob("**/*.yaml"):
        try:
            spec = load_yaml_file(file_path)
            is_valid, msg = validate_yaml_strategy(file_path)

            strategies.append(
                {
                    "file": str(file_path),
                    "name": spec.get("strategy", {}).get("name", file_path.stem),
                    "type": spec.get("strategy", {}).get("type", "unknown"),
                    "universe": spec.get("universe", "unknown"),
                    "valid": is_valid,
                    "validation_msg": msg,
                }
            )
        except Exception as e:
            strategies.append(
                {
                    "file": str(file_path),
                    "name": file_path.stem,
                    "type": "error",
                    "universe": "unknown",
                    "valid": False,
                    "validation_msg": str(e),
                }
            )

    return strategies


if __name__ == "__main__":
    # Test strategy loader
    logging.basicConfig(level=logging.INFO)

    # List available strategies
    print("Available strategy types:")
    for name in STRATEGY_REGISTRY.keys():
        print(f"  - {name}")

    # Test validation
    test_spec = {
        "strategy": {"name": "test_sma", "type": "sma_crossover", "description": "Test strategy"},
        "parameters": {"fast_period": 50, "slow_period": 200},
        "universe": "demo",
    }

    print("\nTest spec validation:", validate_yaml_strategy.__doc__)
