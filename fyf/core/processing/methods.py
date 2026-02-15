"""Processing method dispatch for FYF.

This module provides a lightweight registry that allows the CLI to route
processing to different backends (INLA, MCMC, convolution, etc.) without
changing call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans

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
    """Fill NaNs using astropy convolution interpolation."""
    if data.ndim != 2:
        raise ValueError("Convolution backend expects 2D image data.")

    data_float = data.astype(np.float64, copy=True)
    nan_mask = np.isnan(data_float)

    if not np.any(nan_mask):
        restored = data_float
    else:
        kernel = Gaussian2DKernel(x_stddev=1.0, y_stddev=1.0)
        restored = interpolate_replace_nans(data_float, kernel)

        # If any NaNs remain (e.g., large disconnected masked regions),
        # use a smoothed fallback from finite pixels.
        if np.isnan(restored).any():
            finite_mask = np.isfinite(data_float).astype(np.float64)
            weighted_sum = convolve(
                np.nan_to_num(data_float, nan=0.0),
                kernel,
                boundary="extend",
                nan_treatment="fill",
                normalize_kernel=False,
            )
            weights = convolve(
                finite_mask,
                kernel,
                boundary="extend",
                nan_treatment="fill",
                normalize_kernel=False,
            )
            fallback = np.divide(
                weighted_sum,
                weights,
                out=np.copy(data_float),
                where=weights > 0,
            )
            restored = np.where(np.isnan(restored), fallback, restored)

    # Lightweight uncertainty proxy: absolute local residual from smoothed field.
    smoothed = convolve(
        restored,
        Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5),
        boundary="extend",
        nan_treatment="interpolate",
    )
    uncertainty = np.abs(restored - smoothed)
    uncertainty[~nan_mask] = 0.0

    variant_output_dir = Path(output_dir) / "original"
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(variant_output_dir / "out.npy", restored.astype(np.float32))
    np.save(variant_output_dir / "outsd.npy", uncertainty.astype(np.float32))

    return {
        "restored": restored.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "meta": {"method": "convolution"},
    }


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

    if canonical in {"mcmc"}:
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
