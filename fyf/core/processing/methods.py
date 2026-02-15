"""Processing method dispatch for FYF.

This module provides a lightweight registry that allows the CLI to route
processing to different backends (INLA, MCMC, convolution, etc.) without
changing call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from fyf.config import CosmicConfig, INLAConfig, SatelliteConfig
from fyf.core.processing.fits_processor import FitsProcessor


def _run_inla(
    data: np.ndarray,
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
) -> Dict[str, object]:
    """Run the existing INLA pipeline and return normalized output keys."""
    processor = FitsProcessor(CosmicConfig(fraction=0.0), SatelliteConfig(num_trails=0, trail_width=1))
    variants = {"original": data}
    processed = processor.process_variants(variants, inla_config, str(output_dir))

    return {
        "restored": processed.get("original"),
        "uncertainty": processed.get("original_uncertainty"),
        "meta": {"method": "inla"},
    }


def _run_mcmc(
    data: np.ndarray,
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
) -> Dict[str, object]:
    """Placeholder for a future MCMC backend."""
    raise NotImplementedError("Method 'mcmc' is not implemented yet.")


def _run_convolution(
    data: np.ndarray,
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
) -> Dict[str, object]:
    """Placeholder for a future convolution backend."""
    raise NotImplementedError("Method 'convolution' is not implemented yet.")


METHOD_REGISTRY: Dict[str, Callable[..., Dict[str, object]]] = {
    "inla": _run_inla,
    "mcmc": _run_mcmc,
    "convolution": _run_convolution,
}

ALIASES = {
    "conv": "convolution",
}


def normalize_method(method: str) -> str:
    """Normalize method names and aliases to canonical registry keys."""
    canonical = method.lower().strip()
    return ALIASES.get(canonical, canonical)


def get_supported_methods() -> tuple[str, ...]:
    """Return supported method names for CLI choices."""
    return tuple(METHOD_REGISTRY.keys())


def ensure_method_available(method: str) -> None:
    """Validate method availability and fail fast for placeholders."""
    canonical = normalize_method(method)
    if canonical not in METHOD_REGISTRY:
        options = ", ".join(get_supported_methods())
        raise ValueError(f"Unknown method '{method}'. Available methods: {options}")

    if canonical != "inla":
        raise NotImplementedError(f"Method '{canonical}' is recognized but not implemented yet.")


def run_processing_method(
    method: str,
    data: np.ndarray,
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
) -> Dict[str, object]:
    """Dispatch processing to the selected backend."""
    canonical = normalize_method(method)
    if canonical not in METHOD_REGISTRY:
        options = ", ".join(get_supported_methods())
        raise ValueError(f"Unknown method '{method}'. Available methods: {options}")

    runner = METHOD_REGISTRY[canonical]
    return runner(data=data, output_dir=output_dir, inla_config=inla_config)
