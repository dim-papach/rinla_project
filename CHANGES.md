# FYF Change Summary

This document summarizes the updates we made in this collaboration session.

## CLI and Help Updates

- Added method selection to processing:
  - `fyf process --method [inla|mcmc|convolution]`
  - Defined in `fyf/cli.py`.
- Fixed help/flag exposure so `--method` appears in `process --help`.
- Added 3D preprocessing flag:
  - `fyf process --preprocess [split2d|pca|svd]`
  - Defined in `fyf/cli.py`.

## Processing Methods

- Added method dispatch registry for processing backends in `fyf/core/processing/methods.py`.
- Implemented `convolution` backend using `astropy` convolution/interpolation.
- Kept `mcmc` as a recognized placeholder with explicit error:
  - `"Method 'mcmc' is recognized but not implemented yet."`

## 3D FITS Support (Initial)

- Added preprocessing dispatch module: `fyf/core/processing/preprocessing.py`.
- Added support for 2D and 3D input handling in `process` command:
  - 2D: processed directly by selected backend.
  - 3D: routed through selected preprocessor.
- Implemented `split2d` preprocessor:
  - Splits 3D cube into slices and processes each slice with selected method.
- `pca` and `svd` preprocessors are recognized placeholders with explicit not-implemented errors.

## Output Structure and Artifacts

- Changed per-file output folder naming to:
  - `{original-name}_{method}`
  - Implemented in `fyf/cli.py`.
- Ensured original FITS file is copied into:
  - `<output>/<original-name>_<method>/original/<original-filename>.fits`
  - Implemented in `fyf/cli.py`.
- Existing `.npy` outputs continue to be saved under the `original` variant folder (backend-dependent).

## Config Updates

- Added processing method to config usage:
  - `process.method`
- Added 3D preprocessing setting to config:
  - `process.preprocess`
- Updated config template generation in `fyf/config/config_manager.py`:
  - Includes `"method"` and `"preprocess"` under `"process"`.
- Updated repo config file `fyf-config.json`:
  - `"process": { "method": "convolution", "preprocess": "split2d", ... }`

## Module Exports

- Exported new preprocessing helpers from `fyf/core/processing/__init__.py`:
  - `get_supported_preprocessors`
  - `run_preprocessed_processing`

## Validation Notes

- Verified CLI help now includes:
  - `--method [inla|mcmc|convolution]`
  - `--preprocess [split2d|pca|svd]`
- Verified 3D run with `--preprocess split2d --method convolution` succeeds.
- Verified `--preprocess pca` fails with clear not-implemented message.

## Docker Path + 3D INLA Robustness Fixes

- Fixed duplicate file processing in Docker path resolution:
  - `fyf process test2.fits` no longer expands to multiple mounted paths (`/app`, `/data`) for the same logical file.
  - Implemented in `validate_fits_files` in `fyf/cli.py` by selecting the first valid candidate per input pattern and deduplicating by inode.
- Added explicit warning for unresolved direct FITS paths:
  - `Warning: Skipping invalid file: <name>.fits`
  - Implemented in `fyf/cli.py`.
- Improved INLA scaling behavior for non-positive pixel data:
  - If `scaling=log` and input contains `<= 0`, processing automatically falls back to `scaling=none` for that input with a warning.
  - Implemented in `fyf/core/processing/methods.py`.

## 3D Preprocess Output + Preprocess-Only Mode

- Added preprocess-only execution mode for 3D FITS:
  - You can now run `fyf process --preprocess split2d <cube.fits>` without `--method`.
  - In this mode, FYF performs preprocessing and exits without running INLA/convolution.
  - For 2D input, a processing method is still required.
- Updated `process --help` to describe optional `--method` behavior for preprocess-only runs.
- During `split2d` preprocessing, each preprocessed slice is now saved as FITS in:
  - `<output>/<original-name>_<method-or-preprocess>/preprocessed/slice_XXXX.fits`
- Existing behavior is preserved when a method is provided:
  - Slices are preprocessed, then processed, and restored outputs are saved as before.
