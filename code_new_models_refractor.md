# Multi-Method Processing Refactor (Implemented)

This project now includes a minimal method-dispatch architecture so processing is no longer hardcoded to INLA in the CLI.

## What Was Changed

1. Added method registry and dispatcher:
- `fyf/core/processing/methods.py`
- Canonical methods: `inla`, `mcmc`, `convolution`
- Alias support: `conv -> convolution`
- Unified dispatch entrypoint: `run_processing_method(...)`

2. Updated CLI process command:
- File: `fyf/cli.py`
- New option: `--method [inla|mcmc|convolution]` (default: `inla`)
- Process flow now dispatches via method registry instead of directly calling INLA logic in the command body.
- INLA installation checks run only when `--method inla` is selected.

3. Preserved INLA behavior through dispatch:
- INLA runner uses existing `FitsProcessor.process_variants(...)`
- Return is normalized to:
  - `restored`
  - `uncertainty` (optional)
  - `meta`

4. Exported method utilities from processing package:
- File: `fyf/core/processing/__init__.py`
- Exports:
  - `ensure_method_available`
  - `get_supported_methods`
  - `run_processing_method`

## Current Method Status

- `inla`: Implemented and active.
- `mcmc`: Recognized but not implemented yet.
- `convolution`: Recognized but not implemented yet.

For unimplemented methods, the CLI fails fast with a clear error message.

## Why This Is Simpler

This uses a lightweight registry instead of a full abstract class hierarchy. It gives immediate extensibility with minimal disruption:

- No broad rewrite of `FitsProcessor` required.
- Existing INLA path remains functional.
- New methods can be added incrementally by implementing one runner function and registering it.

## Next Step (Recommended)

Implement `convolution` as the first non-INLA backend in `fyf/core/processing/methods.py`, then add method-specific CLI/config options only for that backend.
