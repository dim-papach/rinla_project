---
output:
  pdf_document: default
  html_document: default
---
# INLA Variable Report: Astronomical Image Reconstruction

This report describes the configuration parameters for the INLA (Integrated Nested Laplace Approximation) backend in the FYF package.

## 1. Basic Parameters
Controls the fundamental behavior of the spatial model and data preparation.

| Variable | Description | Effect on INLA |
| :--- | :--- | :--- |
| `shape` | Spatial trend type (`none`, `radius`, `ellipse`). | Determines the global background model. `radius` is best for stars, `ellipse` for galaxies, and `none` for uniform fields. |
| `scaling` | Data transformation (`log` or `none`). | `log` is recommended for astronomical data to handle high dynamic range and ensure positivity in the reconstructed field. |
| `tolerance` | Convergence threshold (default: `1e-4`). | Controls how precise the optimization must be. Lower values increase accuracy but may prevent convergence or increase time. |
| `restart` | Number of retries (default: `0`). | If the model fails to converge, INLA will restart with different initial values. Increase this for complex images. |
| `nonstationary` | Toggle for non-stationary model. | If `True`, the spatial correlation (range and variance) can vary across the image. Useful for complex nebulae but significantly slower. |

## 2. Mesh Parameters
Controls the Triangulated Mesh (the "grid") used for the SPDE approximation.

| Variable | Description | Effect on INLA |
| :--- | :--- | :--- |
| `mesh_resolution` | Internal resolution factor (default: `30`). | Higher values result in a finer mesh. This is the primary "quality vs speed" knob for the user. |
| `max_edge_factor` | Triangle size control. | Determines the maximum size of a triangle relative to image dimensions. Larger factors = smaller triangles = more detail. |
| `mesh_cutoff` | Minimum point distance. | Prevents triangles from becoming too small in dense data areas, which preserves numerical stability. |
| `outer_edge_factor` | Boundary triangle size. | Controls the size of triangles in the "buffer" zone outside the image. Coarser boundaries speed up computation. |
| `offset_inner/outer` | Boundary extension. | Controls how far the mesh extends beyond the image. Proper offsets prevent "edge effects" where the model acts strangely at the borders. |

## 3. SPDE & Prior Parameters
Defines the statistical properties of the spatial field and our "prior" beliefs.

| Variable | Description | Effect on INLA |
| :--- | :--- | :--- |
| `alpha` | Smoothness parameter (1 or 2). | `2` (default) produces smoother, differentiable fields (standard for images). `1` produces more "jagged" or "noisy" fields. |
| `prior_range_lower` | Minimum expected correlation distance. | Tells the model that features (like stars or galaxy arms) are likely at least this many pixels wide. |
| `prior_sigma_upper` | Maximum expected field variability. | Controls how much the model is allowed to "wiggle" to fit the data. High values allow for sharper features but may fit noise. |
| `prior_range/sigma_prob` | PC-Prior probabilities. | Controls the "strength" of our belief in the bounds above. Usually kept at `0.2`. |

## 4. Computation & Advanced
Performance tuning for the R-INLA solver.

| Variable | Description | Effect on INLA |
| :--- | :--- | :--- |
| `num_threads` | CPU core allocation. | Parallelizes the linear algebra. Diminishing returns after 6-8 threads for most image sizes. |
| `openmp_strategy` | Solver strategy (`small` to `huge`). | `huge` is default for images. It optimizes memory access patterns for large spatial meshes. |
| `nbasis` | Basis functions (Non-stationary only). | Number of spline components used to model the change in spatial parameters across the image. |
| `spline_degree` | B-spline degree (Non-stationary only). | `3` (Cubic) provides smooth transitions in spatial properties. High degrees can cause overfitting. |
