# INLA_pipeline.R - Updated with Extended Parameters
library(INLA)
library(reshape2)
library(ggplot2)
library(viridis)
library(optparse)
library(rlang)
library(reticulate)

cat("Debug: Loaded required libraries\n")

# ========== COMMAND LINE ARGUMENT PARSING ==========
option_list <- list(
  # Basic parameters
  make_option("--input-file", type="character", default=NULL,
              help="Path to input NPY file"),
  make_option("--path-file", type="character", default="/tmp/fyf_variants/path.txt",
            help="Path to the file containing NPY path"),
  make_option("--shape", type="character", default="none", 
              help="Shape parameter: none, radius, or ellipse [default: %default]"),
  make_option("--scaling", type = "character", default="log",
              help="Scaling transformation: 'log' or 'none' [default: %default]"),
  make_option("--tolerance", type="double", default=1e-4,
              help="INLA convergence tolerance [default: %default]"),
  make_option("--restart", type="integer", default=0L,
              help="Number of INLA restarts [default: %default]"),
  make_option("--nonstationary",type = "logical" ,action="store_true", default = FALSE,
              help="Use non-stationary SPDE model [default: %default]"),
              
  # Mesh parameters
  make_option("--mesh-cutoff", type="double", default=NULL,
              help="Minimum distance between mesh points [default: %default]"),
  make_option("--mesh-resolution", type="integer", default=30L,
              help="Mesh resolution factor [default: %default]"),
  make_option("--max-edge-factor", type="double", default=10.0,
              help="Max edge factor for mesh [default: %default]"),
  make_option("--outer-edge-factor", type="double", default=1.5,
              help="Outer edge factor for mesh [default: %default]"),
  make_option("--offset-inner-factor", type="double", default=0.5,
              help="Inner offset factor for mesh [default: %default]"),
  make_option("--offset-outer-factor", type="double", default=2.0,
              help="Outer offset factor for mesh [default: %default]"),
              
  # SPDE parameters
  make_option("--alpha", type="integer", default=2L,
              help="SPDE smoothness parameter (1 or 2) [default: %default]"),
  make_option("--prior-range-prob", type="double", default=0.2,
              help="Prior probability for range parameter [default: %default]"),
  make_option("--prior-range-lower", type="double", default=2.0,
              help="Lower bound for range prior [default: %default]"),
  make_option("--prior-sigma-prob", type="double", default=0.2,
              help="Prior probability for sigma parameter [default: %default]"),
  make_option("--prior-sigma-upper", type="double", default=2.0,
              help="Upper bound for sigma prior [default: %default]"),
              
  # Computation parameters
  make_option("--num-threads", type="integer", default=6L,
              help="Number of threads for INLA [default: %default]"),
  make_option("--openmp-strategy", type="character", default="huge",
              help="OpenMP strategy: small, medium, large, huge [default: %default]"),
              
  # Non-stationary parameters
  make_option("--nbasis", type="integer", default=2L,
              help="Number of basis functions for non-stationary model [default: %default]"),
  make_option("--spline-degree", type="integer", default=3L,
              help="Degree of B-spline basis functions [default: %default]")
)

# Parse arguments
opt_parser <- OptionParser(option_list=option_list)
opts <- parse_args(opt_parser)

# Set INLA options based on parsed arguments
inla.setOption(num.threads = opts$`num-threads`)

# Display configuration
cat("=== INLA CONFIGURATION ===\n")
cat("Shape:", opts$shape, "\n")
cat("Scaling:", opts$scaling, "\n")
cat("Tolerance:", opts$tolerance, "\n")
cat("Restart:", opts$restart, "\n")
cat("Non-stationary:", opts$nonstationary, "\n")
cat("Mesh resolution:", opts$`mesh-resolution`, "\n")
cat("Max edge factor:", opts$`max-edge-factor`, "\n")
cat("Prior range lower:", opts$`prior-range-lower`, "\n")
cat("Prior sigma upper:", opts$`prior-sigma-upper`, "\n")
cat("Threads:", opts$`num-threads`, "\n")
cat("OpenMP strategy:", opts$`openmp-strategy`, "\n")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

#' Load data from NPY file and transpose to correct orientation
load_npy <- function(file_path) {
  np <- reticulate::import("numpy")
  data <- t(np$load(file_path))  # Transpose for correct orientation
  return(data)
}

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

#' Prepare image data for INLA analysis
prepare_data <- function(img, scaling = opts$scaling) {
  if (is.null(dim(img)) || length(dim(img)) != 2) {
    stop("Error: The image data is not a 2D matrix.")
  }

  dims <- dim(img)
  cat("Image dimensions:", dims, "\n")

  # Create coordinate matrices
  x <- matrix(rep(1:dims[1], dims[2]), nrow = dims[1], ncol = dims[2])
  y <- matrix(rep(1:dims[2], each = dims[1]), nrow = dims[1], ncol = dims[2])
  
  # Clean image data
  img[img == "BLANK" | img == "blank"] <- NA
  
  # Apply scaling transformation
  if (scaling == "log") {
    logimg <- log10(img)
    logimg[is.infinite(logimg)] <- 0  # Replace -Inf and Inf with 0
    cat("Applied log10 scaling\n")
  } else if (scaling == "none") {
    logimg <- img
    cat("No scaling applied\n")
  } else {
    stop("Unknown scaling option: ", scaling)
  }
  
  # Identify valid data points
  valid <- which(!is.na(img) & !is.nan(img) & img != 0 & 
                 !is.infinite(logimg) & !is.na(logimg) & !is.nan(logimg))

  if (length(valid) == 0) {
    stop("Error: No valid data points found in the image.")
  }

  return(list(
    x = x, y = y, valid = valid, 
    xsize = dims[1], ysize = dims[2],
    xfin = dims[1], yfin = dims[2], 
    logimg = logimg, img = img
  ))
}

#' Validate data before INLA processing
check_data_validity <- function(valid, tx, ty, logimg, img) {
  cat("Data validation:\n")
  cat("  Valid points:", length(valid), "\n")
  cat("  Expected non-NA/non-zero:", sum(!is.na(img) & img != 0), "\n")

  if (length(valid) == 0) {
    stop("Error: No valid data points found for INLA analysis.")
  }
  if (length(valid) != sum(!is.na(img) & img != 0)) {
    stop("Error: Mismatch between valid indices and non-NA/non-zero values.")
  }
}

#' Compute parameters for INLA model
compute_parameters <- function(valid, tx, ty, logimg, weight, tepar = NULL) {
  x <- tx[valid]
  y <- ty[valid]
  par <- logimg[valid]
  
  if (any(!is.finite(par))) {
    stop("Error: Non-finite values detected in parameters.")
  }

  cat("Parameter summary:", summary(par), "\n")

  # Error parameters (if available)
  epar <- if (!is.null(tepar)) tepar^2 else NULL

  # Compute weighted centers
  xcenter <- sum(x * weight) / sum(weight)
  ycenter <- sum(y * weight) / sum(weight)
  cat("Centers: xcenter =", xcenter, ", ycenter =", ycenter, "\n")
  
  return(list(
    x = x, y = y, par = par, epar = epar,
    xcenter = xcenter, ycenter = ycenter
  ))
}

# ============================================================================
# MESH CREATION WITH NEW PARAMETERS
# ============================================================================

#' Create INLA mesh with configurable parameters
create_inla_mesh <- function(x, y) {
  if (length(x) == 0 || length(y) == 0) {
    stop("Error: Insufficient points to create a mesh.")
  }
  
  # Calculate data range
  x_range <- diff(range(x))
  y_range <- diff(range(y))
  max_range <- max(x_range, y_range)

  # Calculate max.edge using the factor
  max_edge <- max_range / opts$`max-edge-factor`

  # Calculate cutoff (use provided value or auto-calculate)
  cutoff <- if (!is.null(opts$`mesh-cutoff`)) {
    opts$`mesh-cutoff`
  } else {
    max(max_edge / opts$`mesh-resolution`, 1e-5)
  }
  
  cat("Mesh parameters:\n")
  cat("  max_edge:", max_edge, "\n")
  cat("  cutoff:", cutoff, "\n")
  cat("  mesh_resolution:", if (is.null(opts$`mesh-cutoff`)) opts$`mesh-resolution` else "N/A (manual cutoff)", "\n")

  # Create mesh with new parameters
  mesh <- tryCatch(
    INLA::inla.mesh.2d(
      loc = cbind(x, y),
      max.edge = c(max_edge, max_edge * opts$`outer-edge-factor`),
      cutoff = cutoff,
      offset = c(max_edge * opts$`offset-inner-factor`, max_edge * opts$`offset-outer-factor`)
    ),
    error = function(e) stop("Mesh creation failed: ", e$message))

  cat("Mesh created with", mesh$n, "vertices\n")
  return(mesh)
}

# ============================================================================
# SPDE MODEL WITH NEW PARAMETERS
# ============================================================================

#' Define SPDE model with configurable parameters
define_spde_model <- function(mesh) {
  # Create prior vectors
  p_range <- c(opts$`prior-range-lower`, opts$`prior-range-prob`)
  p_sigma <- c(opts$`prior-sigma-upper`, opts$`prior-sigma-prob`)
  
  cat("SPDE parameters:\n")
  cat("  nonstationary:", opts$nonstationary, "\n")
  cat("  alpha:", opts$alpha, "\n")
  cat("  p_range:", p_range, "\n")
  cat("  p_sigma:", p_sigma, "\n")
  
  if (opts$nonstationary) {
    cat("  nbasis:", opts$nbasis, "\n")
    cat("  spline_degree:", opts$`spline-degree`, "\n")
    
    # Non-stationary model
    basis.T <- inla.mesh.basis(mesh, type = "b.spline", n = opts$nbasis, degree = opts$`spline-degree`)
    basis.K <- inla.mesh.basis(mesh, type = "b.spline", n = opts$nbasis, degree = opts$`spline-degree`)

    spde <- inla.spde2.matern(mesh = mesh, alpha = opts$alpha,
                              B.tau = cbind(0, basis.T, basis.K * 0),
                              B.kappa = cbind(0, basis.T * 0, basis.K / 2))
  } else {
    # Stationary model with PC priors
    spde <- inla.spde2.pcmatern(mesh = mesh, alpha = opts$alpha,
                                prior.range = p_range,
                                prior.sigma = p_sigma)
  }
  return(spde)
}

# ============================================================================
# MODEL STACK PREPARATION
# ============================================================================

#' Prepare model stack for INLA
prepare_model_stack <- function(shape = opts$shape, x, y, par, A, spde, weight, xcenter, ycenter) {
  eigens <- NULL  # Initialize eigens
  
  if (shape == 'radius') {
    radius <- sqrt((x - xcenter)^2 + (y - ycenter)^2)
    radius_2 <- radius^2
    stk <- inla.stack(
      data = list(par = par),
      A = list(A, 1, 1, 1),
      effects = list(
        i = 1:spde$n.spde, 
        m = rep(1, length(x)),
        radius = radius, 
        radius_2 = radius_2
      ),
      tag = 'est'
    )
    return(list(stk = stk, eigens = NULL))
    
  } else if (shape == 'ellipse') {
    # Compute weighted covariance for ellipse
    m_weights <- rep(weight, length(x))
    covar <- cov.wt(cbind(x, y), wt = m_weights)
    eigens <- eigen(covar$cov)
    
    # Mahalanobis distance
    ellipse <- (cbind(x - xcenter, y - ycenter) %*% eigens$vectors[,1])^2 / eigens$values[1] +
               (cbind(x - xcenter, y - ycenter) %*% eigens$vectors[,2])^2 / eigens$values[2]
    ellipse_2 <- ellipse^2
    
    stk <- inla.stack(
      data = list(par = par),
      A = list(A, 1, 1, 1),
      effects = list(
        i = 1:spde$n.spde, 
        m = rep(1, length(x)),
        ellipse = ellipse, 
        ellipse_2 = ellipse_2
      ),
      tag = 'est'
    )
    return(list(stk = stk, eigens = eigens))
    
  } else if (shape == 'none') {
    # No spatial covariates
    stk <- inla.stack(
      data = list(par = par),
      A = list(A, 1),
      effects = list(
        i = 1:spde$n.spde, 
        m = rep(1, length(x))
      ),
      tag = 'est'
    )
    return(list(stk = stk, eigens = NULL))
  }
}

# ============================================================================
# INLA MODEL EXECUTION WITH NEW PARAMETERS
# ============================================================================

#' Run INLA model with configurable parameters
run_inla_model <- function(stk, par, epar, spde) {
  if (is.null(stk) || !inherits(stk, "inla.data.stack")) {
    stop("'stack' must inherit from class \"inla.data.stack\".")
  }
  
  # Define formula based on shape
  formula <- switch(opts$shape,
    'radius' = par ~ 0 + m + radius + radius_2 + f(i, model = spde),
    'ellipse' = par ~ 0 + m + ellipse + ellipse_2 + f(i, model = spde),
    'none' = par ~ 0 + m + f(i, model = spde),
    stop("Error: Invalid shape parameter.")
  )

  cat("Running INLA with:\n")
  cat("  tolerance:", opts$tolerance, "\n")
  cat("  restart:", opts$restart, "\n")
  cat("  openmp_strategy:", opts$`openmp-strategy`, "\n")

  # Run the INLA model
  res <- inla(formula,
              data = inla.stack.data(stk),
              control.predictor = list(A = inla.stack.A(stk)),
              scale = epar,
              control.compute = list(openmp.strategy = opts$`openmp-strategy`),
              control.inla = list(tolerance = opts$tolerance, restart = opts$restart),
              verbose = TRUE
              )

  return(res)
}

# ============================================================================
# PROJECTION FUNCTIONS (KEEPING YOUR EXISTING LOGIC)
# ============================================================================

create_projector <- function(mesh, xlim, ylim, zoom, xsize, ysize) {
  cat("=== PROJECTION FUNCTIONS LOADED ===\n")

  fmesher::fm_evaluator(mesh,
                      xlim = xlim,
                      ylim = ylim,
                      dims = c(zoom * xsize, zoom * ysize))
}

compute_spatial_term <- function(projector, shape = opts$shape, res, xcenter, ycenter, eigens) {
  cat("=== COMPUTING SPATIAL TERM ===\n")
  if (shape == 'radius') {
    radius_proj <- sqrt((projector$x - xcenter)^2 + (projector$y - ycenter)^2)
    radius_2_proj <- radius_proj^2
    term <- res$summary.fixed$mean[1] + 
            res$summary.fixed$mean[2] * c(radius_proj) + 
            res$summary.fixed$mean[3] * c(radius_2_proj)
  } else if (shape == 'ellipse') {
    coords_proj <- cbind(c(projector$x) - xcenter, c(projector$y) - ycenter)
    transformed_coords_proj <- coords_proj %*% eigens$vectors
    ellipse_proj <- rowSums(transformed_coords_proj^2 / matrix(eigens$values, nrow = nrow(transformed_coords_proj), ncol = 2, byrow = TRUE))
    ellipse_2_proj <- ellipse_proj^2
    term <- res$summary.fixed$mean[1] + 
            res$summary.fixed$mean[2] * ellipse_proj + 
            res$summary.fixed$mean[3] * ellipse_2_proj
  } else {
    term <- rep(res$summary.fixed$mean[1], length(projector$x))
  }
  return(term)
}

project_inla_results <- function(mesh, res, xini, xfin, yini, yfin, xsize, ysize, 
                                zoom, shape, xcenter, ycenter, eigens, spde = NULL) {
  cat("=== PROJECTING INLA RESULTS ===\n")
  if (is.null(mesh) || is.null(res)) {
    stop("Error: Mesh and result inputs cannot be NULL.")
  }
  # Create projector with consistent dimensions
  projector <- create_projector(mesh, c(xini, xfin), c(yini, yfin), zoom, xsize, ysize)

  # Calculate spatial trend component
  spatial_term <- compute_spatial_term(projector, shape, res, xcenter, ycenter, eigens)

  # Project random effects and combine with trend
  random_effects <- fmesher::fm_evaluate(projector, res$summary.random$i$mean)
  output <- random_effects + 
            t(matrix(spatial_term, nrow = zoom * ysize, ncol = zoom * xsize))

  # Project standard deviations
  outputsd <- fmesher::fm_evaluate(projector, res$summary.random$i$sd)

  return(list(
    out = t(output),      # Transpose for correct orientation
    outsd = t(outputsd)   # Transpose for correct orientation
  ))
}

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

#' Apply inverse scaling transformation
unscale_results <- function(results, scaling = opts$scaling) {
  if (scaling == "log") {
    results <- lapply(results, function(x) {
      if (is.numeric(x)) {
        10^x  # Inverse of log10
      } else {
        x
      }
    })
  }
  return(results)
}

#' Save results as NPY files
save_npy <- function(array_list, dir_path) {
  if (!is.list(array_list) || is.null(names(array_list))) {
    stop("Input must be a named list of matrices or arrays.")
  }

  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }

  np <- reticulate::import("numpy")
  saved_files <- character()

  for (name in names(array_list)) {
    arr <- array_list[[name]]
    if (!is.matrix(arr) && !is.array(arr)) {
      warning(sprintf("Skipping '%s': not a matrix or array.", name))
      next
    }

    file_path <- file.path(dir_path, paste0(name, ".npy"))
    np$save(file_path, arr)
    saved_files <- c(saved_files, file_path)
  }

  invisible(saved_files)
}

# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================
shape <- opts$shape
tryCatch({
  cat("=== STARTING INLA PIPELINE ===\n")
  
  # 1. Load input path
  npy_path <- opts$`input-file`
  
  # Check if npy_path is provided via option, else check trailing args
  if (is.null(npy_path)) {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) > 0 && !startsWith(args[1], "-")) {
      npy_path <- args[1]
    }
  }
  
  if (is.null(npy_path) || !file.exists(npy_path)) {
    stop("No valid input file provided. Use --input-file or provide path as argument.")
  }
  
  cat("Loading data from:", npy_path, "\n")

  # 2. Load and prepare data
  raw_data <- load_npy(npy_path)
  cat("Data loaded, dimensions:", paste(dim(raw_data), collapse = "x"), "\n")

  inla_variables <- prepare_data(raw_data)
  
  check_data_validity(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    img = inla_variables$img
  )

  # 3. Compute model parameters
  model_params <- compute_parameters(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    weight = 1
  )

  # 4. Create mesh and SPDE model with new parameters
  inla_mesh <- create_inla_mesh(model_params$x, model_params$y)
  spde_model <- define_spde_model(inla_mesh)

  # 5. Create projection matrix and model stack
  projection_matrix_A <- inla.spde.make.A(
    inla_mesh,
    loc = cbind(model_params$x, model_params$y)
  )

  model_stack <- prepare_model_stack(
    shape = opts$shape,
    x = model_params$x,
    y = model_params$y,
    par = model_params$par,
    A = projection_matrix_A,
    spde = spde_model,
    weight = 1,
    xcenter = model_params$xcenter,
    ycenter = model_params$ycenter
  )

  # 6. Run INLA model
  cat("Running INLA model...\n")
  inla_result <- run_inla_model(
    stk = model_stack$stk,
    par = model_params$par,
    epar = model_params$epar,
    spde = spde_model
  )

  # 7. Project results
  cat("Projecting results...\n")
  projected_results <- project_inla_results(
    mesh = inla_mesh,
    res = inla_result,
    xini = 0,
    xfin = inla_variables$xfin,
    yini = 0,
    yfin = inla_variables$yfin,
    xsize = inla_variables$xsize,
    ysize = inla_variables$ysize,
    zoom = 1,
    shape = opts$shape,
    xcenter = model_params$xcenter,
    ycenter = model_params$ycenter,
    eigens = model_stack$eigens
  )

  # 8. Apply inverse scaling and save
  final_results <- unscale_results(projected_results)
  
  output_dir <- Sys.getenv("FYF_OUTPUT_DIR", unset = "processed")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Using fixed output name "out.npy" to match Python expectation or keep original basename?
  # The Python script expects "out.npy" in the variant-specific folder.
  # Let's check the Python script logic again. It looks for "out.npy".
  # "output_path = os.path.join(variant_output_dir, 'out.npy')"
  
  # The existing R code preserved the basename. The Python script sets FYF_OUTPUT_DIR to a variant-specific folder.
  # If we want to be compatible with the Python script, we should probably save as 'out.npy' OR the Python script should look for the basename.
  # However, the previous code did: fname <- sub("\\.npy$", "", basename(npy_path)); out_path <- file.path(output_dir, fname)
  # And `save_npy` appends `.npy`. So it saved `processed/<basename>/out.npy` or similar?
  
  # Wait, `save_npy` iterates over the list names (out, outsd) and saves them as `name.npy`.
  # So if final_results has `out` and `outsd`, it saves `output_dir/out.npy` and `output_dir/outsd.npy`.
  # This matches `save_npy(final_results, out_path)`.
  # BUT `out_path` passed to `save_npy` acts as the directory path in the original code?
  # Let's check `save_npy` implementation in the previous read.
  
  # save_npy implementation:
  # save_npy <- function(array_list, dir_path) { ... file_path <- file.path(dir_path, paste0(name, ".npy")) ... }
  
  # So `out_path` IS the directory path.
  # Previous code: out_path <- file.path(output_dir, fname)
  # If output_dir was "processed", and fname was "cosmic", it created "processed/cosmic" and saved "out.npy" inside it.
  
  # The Python script sets FYF_OUTPUT_DIR to the variant specific directory.
  # So we just need to use that.
  
  save_npy(final_results, output_dir)
  
  cat("=== PIPELINE COMPLETED SUCCESSFULLY ===\n")
  cat("Results saved to:", output_dir, "\n")

}, error = function(e) {
  cat("=== PIPELINE FAILED ===\n")
  cat("Error:", conditionMessage(e), "\n")
  quit(status = 1)
})