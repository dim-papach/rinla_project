---
output:
  html_document: default
  pdf_document: default
---
# Technical Report: R-INLA Pipeline for Astronomical Image Reconstruction

This document provides a comprehensive overview of the mathematical foundations and implementation details of the INLA (Integrated Nested Laplace Approximation) pipeline used in the FYF package.

## 1. Problem Definition: Spatial Inpainting
The goal of the pipeline is to reconstruct missing or corrupted pixels (cosmic rays, satellite trails, or masks) in an astronomical image $Y$. We treat the image as a realization of a continuous spatial process $x(s)$ observed at discrete pixel locations $s_i$:

$$y_i = \mu(s_i) + x(s_i) + \epsilon_i$$

Where:
*   $\mu(s_i)$ is the deterministic spatial trend (e.g., galaxy profile).
*   $x(s_i)$ is a Latent Gaussian Field representing the correlated astronomical structure.
*   $\epsilon_i \sim N(0, \sigma^2_\epsilon)$ is the measurement noise.

## 2. Mathematical Foundations

### 2.1 The Matérn Covariance
We model the spatial correlation $x(s)$ using the Matérn covariance function, which defines the correlation between two points separated by distance $d$:

$$C(d) = \frac{\sigma^2}{2^{
u-1}\Gamma(
u)}(\kappa d)^
u K_
u(\kappa d)$$

Parameters:
*   $\sigma^2$: Marginal variance.
*   $
u$: Smoothness parameter (controlled by `alpha` in FYF).
*   $\kappa$: Scale parameter, related to the spatial range $
ho \approx \frac{\sqrt{8
u}}{\kappa}$.
*   $K_
u$: Modified Bessel function of the second kind.

### 2.2 The SPDE Approach
Traditional Gaussian Processes scale poorly ($O(N^3)$). The R-INLA pipeline uses the **Stochastic Partial Differential Equation (SPDE)** approach to represent the Matérn field as a Gaussian Markov Random Field (GMRF) with sparse precision matrices, scaling at $\approx O(N^{1.5})$.

The field $x(s)$ is the solution to the following SPDE:
$$(\kappa^2 - \Delta)^{\alpha/2} (	au x(s)) = \mathcal{W}(s)$$

Where:
*   $\Delta = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator.
*   $\alpha = 
u + d/2$ (in 2D, $\alpha = 
u + 1$).
*   $\mathcal{W}(s)$ is Gaussian white noise.
*   $	au$ controls the variance: $\sigma^2 = \frac{\Gamma(
u)}{\Gamma(\alpha)(4\pi)^{d/2}\kappa^{2
u}	au^2}$.

### 2.3 Finite Element Method (FEM) Approximation
To solve the SPDE, we discretize the continuous domain using a **Triangulated Mesh**. The field is approximated as a linear combination of basis functions $\psi_j$ defined on the mesh vertices:

$$x(s) \approx \sum_{j=1}^{m} \psi_j(s) w_j$$

Where $w = (w_1, \dots, w_m)$ is a GMRF with a sparse precision matrix $Q$.

## 3. Implementation Details in FYF

### 3.1 Mesh Generation
The mesh is the computational grid. Its quality directly impacts the approximation:
*   **`max_edge`**: Limits the size of triangles. Smaller triangles $
ightarrow$ finer approximation of high-frequency features (stars).
*   **`cutoff`**: Prevents "numerical clusters" by ensuring no two vertices are closer than this distance.
*   **Boundary Extensions**: We extend the mesh beyond the image boundaries (`offset_outer`) to avoid "boundary leakage" where the correlation is artificially inflated at the edges.

### 3.2 Trend Models (Fixed Effects)
The `shape` parameter determines the deterministic component $\mu(s)$:
*   **`none`**: $\mu(s) = \beta_0$ (Constant background).
*   **`radius`**: $\mu(s) = \beta_0 + \beta_1 r + \beta_2 r^2$, where $r = \|s - s_{center}\|$.
*   **`ellipse`**: $\mu(s) = \beta_0 + \beta_1 \mathcal{M} + \beta_2 \mathcal{M}^2$, where $\mathcal{M}$ is the Mahalanobis distance based on the image's second moments.

### 3.3 Prior Specification (PC Priors)
We use **Penalized Complexity (PC) Priors** to constrain the spatial parameters. These priors penalize moving away from a "simpler" model (e.g., infinite range or zero variance):
*   **Range Prior**: $P(
ho < 
ho_{lower}) = p_
ho$
*   **Sigma Prior**: $P(\sigma > \sigma_{upper}) = p_\sigma$

In FYF, these are mapped to `prior_range_lower` and `prior_sigma_upper`.

### 3.4 Non-Stationary Model
When `nonstationary=True`, we allow $\kappa$ and $	au$ to vary spatially:
$$\log(\kappa(s)) = \sum B_i^\kappa(s) 	heta_i^\kappa, \quad \log(	au(s)) = \sum B_i^	au(s) 	heta_i^	au$$
We use B-spline basis functions (defined by `nbasis` and `spline_degree`) to model this variation across the image.

## 4. Computation: The INLA Algorithm
Instead of MCMC sampling, INLA uses nested Laplace approximations to compute the marginal posteriors:
1.  **Exploration**: Find the mode of the hyperparameters $	heta$ (range, variance).
2.  **Integration**: Numerically integrate the latent field $x$ over the hyperparameter grid.

This results in deterministic, highly accurate approximations of the full posterior distribution for every pixel, including the standard deviation (`outsd.npy`).

## 5. Summary of Parameters

| Component | Variable | Math Symbol |
| :--- | :--- | :--- |
| **Smoothness** | `alpha` | $\alpha$ |
| **Mesh Density** | `mesh_resolution` | $h$ (mesh size) |
| **Spatial Range** | `prior_range_lower` | $
ho_0$ |
| **Field Variance** | `prior_sigma_upper` | $\sigma_0$ |
| **Trend** | `shape` | $\mu(s)$ |
| **Computational Threads**| `num_threads` | Parallelization |
