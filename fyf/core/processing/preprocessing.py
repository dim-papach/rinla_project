"""Preprocessing dispatch for 3D FITS volumes.

This module defines preprocessing strategies that run before the selected
processing backend (INLA, convolution, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from astropy.io import fits

from fyf.config import INLAConfig
from fyf.core.processing.methods import run_processing_method

PREPROCESS_REGISTRY = {
    "split2d": "split2d",
    "pca": "pca",
    "svd": "svd",
}


def normalize_preprocess(preprocess: str) -> str:
    """Normalize preprocess strategy name."""
    return preprocess.lower().strip()


def get_supported_preprocessors() -> tuple[str, ...]:
    """Return supported preprocessing choices for CLI/config."""
    return tuple(PREPROCESS_REGISTRY.keys())


def _run_split2d(
    data: np.ndarray,
    method: Optional[str],
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
    header: Optional[fits.Header] = None,
) -> Dict[str, object]:
    """Preprocess a 3D cube slice-by-slice and optionally process each slice."""
    restored_slices = []
    uncertainty_slices = []
    preprocess_dir = output_dir / "preprocessed"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    for index in range(data.shape[0]):
        slice_data = np.asarray(data[index], dtype=np.float32)
        fits.writeto(
            preprocess_dir / f"slice_{index:04d}.fits",
            slice_data,
            header=header,
            overwrite=True,
        )

        if method is None:
            continue

        slice_output_dir = output_dir / f"slice_{index:04d}"
        slice_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_processing_method(
            method=method,
            data=slice_data,
            output_dir=slice_output_dir,
            inla_config=inla_config,
        )

        restored = result.get("restored")
        if restored is None:
            raise RuntimeError(f"Slice {index} returned no restored data.")

        restored_slices.append(np.asarray(restored, dtype=np.float32))

        uncertainty = result.get("uncertainty")
        if uncertainty is None:
            uncertainty_slices.append(np.zeros_like(restored_slices[-1], dtype=np.float32))
        else:
            uncertainty_slices.append(np.asarray(uncertainty, dtype=np.float32))

    if method is None:
        return {
            "restored": None,
            "uncertainty": None,
            "meta": {
                "preprocess": "split2d",
                "slice_count": int(data.shape[0]),
                "preprocess_only": True,
            },
        }

    return {
        "restored": np.stack(restored_slices, axis=0),
        "uncertainty": np.stack(uncertainty_slices, axis=0),
        "meta": {
            "preprocess": "split2d",
            "slice_count": int(data.shape[0]),
            "preprocess_only": False,
        },
    }


def run_preprocessed_processing(
    data: np.ndarray,
    method: Optional[str],
    preprocess: str,
    output_dir: Path,
    inla_config: Optional[INLAConfig] = None,
    header: Optional[fits.Header] = None,
) -> Dict[str, object]:
    """Run processing on 2D/3D data with an optional 3D preprocessing step."""
    if data.ndim == 2:
        if method is None:
            raise ValueError("A processing method is required for 2D inputs.")
        return run_processing_method(
            method=method,
            data=data,
            output_dir=output_dir,
            inla_config=inla_config,
        )

    if data.ndim != 3:
        raise ValueError(f"Unsupported FITS dimensionality: {data.ndim}. Only 2D or 3D are supported.")

    canonical = normalize_preprocess(preprocess)
    if canonical not in PREPROCESS_REGISTRY:
        options = ", ".join(get_supported_preprocessors())
        raise ValueError(f"Unknown preprocess '{preprocess}'. Available preprocessors: {options}")

    if canonical == "split2d":
        return _run_split2d(
            data=data,
            method=method,
            output_dir=output_dir,
            inla_config=inla_config,
            header=header,
        )

    raise NotImplementedError(
        f"Preprocess '{canonical}' is recognized for 3D data but not implemented yet."
    )
