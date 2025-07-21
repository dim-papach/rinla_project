# INLA_pipeline.R - Fixed and Improved Version
library(INLA)
library(reshape2)
library(ggplot2)
library(viridis)
library(optparse)
library(rlang)
library(reticulate)

cat("Debug: Loaded required libraries\n")

# Configuration
SCALING <- TRUE
MAX_EDGE_RESOLUTION <- 30
inla.setOption(num.threads = 6)
cat("Debug: Configuration set - scaling:", SCALING, "resolution:", MAX_EDGE_RESOLUTION, "\n")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

#' Load the path of the npy file from a txt file
load_path <- function(file_path) {
  path <- readLines(file_path, n = 1)
  return(path)
}

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
#' 
#' @param img Numeric matrix representing image data
#' @param scaling Logical, whether to apply log10 transformation
#' @return List containing prepared data for INLA analysis
prepare_data <- function(img, scaling = TRUE) {
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
  if (scaling) {
    logimg <- log10(img)
    logimg[is.infinite(logimg)] <- 0  # Replace -Inf and Inf with 0
  } else {
    logimg <- img
  }
  
  # Identify valid data points (not NA, not NaN, not 0, not infinite)
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

  return(list(x = x, y = y, par = par, epar = epar, 
              xcenter = xcenter, ycenter = ycenter))
}

# ============================================================================
# MESH AND MODEL FUNCTIONS
# ============================================================================

#' Create INLA mesh with adaptive parameters
create_inla_mesh <- function(x, y, max.edge = NULL, resolution = 30) {
  if (length(x) == 0 || length(y) == 0) {
    stop("Error: Insufficient points to create mesh.")
  }
  
  # Calculate data range and set adaptive parameters
  x_range <- diff(range(x))
  y_range <- diff(range(y))
  max_range <- max(x_range, y_range)

  if (is.null(max.edge)) {
    max.edge <- max_range / 10  # 10% of maximum range
  }

  cutoff <- max(max.edge / resolution, 1e-5)

  mesh <- tryCatch(
    INLA::inla.mesh.2d(
      loc = cbind(x, y),
      max.edge = c(max.edge, max.edge * 1.5),
      cutoff = cutoff,
      offset = c(max.edge * 0.5, max.edge * 2)
    ),
    error = function(e) stop("Mesh creation failed: ", e$message)
  )

  return(mesh)
}

#' Define SPDE model (stationary or non-stationary)
define_spde_model <- function(mesh, nonstationary, p_range, p_sigma, 
                             nbasis = 2, degree = 10) {
  if (nonstationary) {
    # Non-stationary model with B-spline basis
    basis.T <- inla.mesh.basis(mesh, type = "b.spline", n = nbasis, degree = degree)
    basis.K <- inla.mesh.basis(mesh, type = "b.spline", n = nbasis, degree = degree)

    spde <- inla.spde2.matern(
      mesh = mesh, alpha = 2,
      B.tau = cbind(0, basis.T, basis.K * 0),
      B.kappa = cbind(0, basis.T * 0, basis.K / 2)
    )
  } else {
    # Stationary model with PC priors
    spde <- inla.spde2.pcmatern(
      mesh = mesh, alpha = 2,
      prior.range = p_range,
      prior.sigma = p_sigma
    )
  }
  return(spde)
}

#' Prepare INLA stack based on shape parameter
prepare_model_stack <- function(shape, x, y, par, A, spde, weight, xcenter, ycenter) {
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
    
  } else {
    stop("Error: Invalid shape parameter. Use 'radius', 'ellipse', or 'none'.")
  }

  return(list(stk = stk, eigens = eigens))
}

#' Run INLA model
run_inla_model <- function(stk, par, epar, spde, tolerance, restart, shape) {
  if (!inherits(stk, "inla.data.stack")) {
    stop("Stack must inherit from class 'inla.data.stack'.")
  }
  
  # Define formula based on shape
  formula <- switch(shape,
    'radius' = par ~ 0 + m + radius + radius_2 + f(i, model = spde),
    'ellipse' = par ~ 0 + m + ellipse + ellipse_2 + f(i, model = spde),
    'none' = par ~ 0 + m + f(i, model = spde),
    stop("Error: Invalid shape parameter.")
  )

  res <- inla(
    formula,
    data = inla.stack.data(stk),
    control.predictor = list(A = inla.stack.A(stk)),
    scale = epar,
    control.compute = list(openmp.strategy = 'huge'),
    control.inla = list(tolerance = tolerance, restart = restart),
    verbose = inla.getOption("verbose")
  )

  return(res)
}

# ============================================================================
# PROJECTION AND OUTPUT FUNCTIONS
# ============================================================================

#' Create mesh projector with consistent dimensions
create_projector <- function(mesh, xlim, ylim, zoom, xsize, ysize) {
  inla.mesh.projector(
    mesh,
    xlim = xlim,
    ylim = ylim,
    dim = zoom * c(xsize, ysize)  # FIXED: Removed +1 for consistency
  )
}

#' Compute spatial trend component for projection
compute_spatial_term <- function(projector, shape, res, xcenter, ycenter, eigens) {
  px <- rep(projector$x, each = length(projector$y))
  py <- rep(projector$y, length(projector$x))

  if (shape == 'radius') {
    projected <- sqrt((px - xcenter)^2 + (py - ycenter)^2)
    term <- res$summary.fixed$mean[1] +
            res$summary.fixed$mean[2] * projected +
            res$summary.fixed$mean[3] * projected^2

  } else if (shape == 'ellipse') {
    centered_coords <- cbind(px - xcenter, py - ycenter)
    projected <- (centered_coords %*% eigens$vectors[, 1])^2 / eigens$values[1] +
                 (centered_coords %*% eigens$vectors[, 2])^2 / eigens$values[2]
    term <- res$summary.fixed$mean[1] +
            res$summary.fixed$mean[2] * projected +
            res$summary.fixed$mean[3] * projected^2

  } else if (shape == 'none') {
    term <- res$summary.fixed$mean[1]
  } else {
    stop("Invalid shape parameter.")
  }

  return(term)
}

#' Main projection function with validation support
project_inla_results <- function(mesh, res, xini, xfin, yini, yfin, xsize, ysize, 
                                zoom, shape, xcenter, ycenter, eigens, spde = NULL) {
  if (is.null(mesh) || is.null(res)) {
    stop("Error: Mesh and result inputs cannot be NULL.")
  }
  
  # Create projector with consistent dimensions
  projector <- create_projector(mesh, c(xini, xfin), c(yini, yfin), zoom, xsize, ysize)

  # Calculate spatial trend component
  spatial_term <- compute_spatial_term(projector, shape, res, xcenter, ycenter, eigens)

  # Project random effects and combine with trend
  random_effects <- inla.mesh.project(projector, res$summary.random$i$mean)
  output <- random_effects + 
            t(matrix(spatial_term, nrow = zoom * ysize, ncol = zoom * xsize))

  # Project standard deviations
  outputsd <- inla.mesh.project(projector, res$summary.random$i$sd)

  return(list(
    out = t(output),      # Transpose for correct orientation
    outsd = t(outputsd)   # Transpose for correct orientation
  ))
}

#' Apply inverse scaling transformation
unscale_results <- function(results, scaling = FALSE) {
  if (scaling) {
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

tryCatch({
  cat("=== STARTING INLA PIPELINE ===\n")
  
  # 1. Load input path
  file_path <- "variants/path.txt"
  if (!file.exists(file_path)) {
    file_path <- commandArgs(trailingOnly = TRUE)[1]
    if (is.na(file_path) || !file.exists(file_path)) {
      stop("No valid path provided.")
    }
  }
  
  npy_path <- readLines(file_path, n = 1)
  cat("Loading data from:", npy_path, "\n")

  # 2. Load and prepare data
  raw_data <- load_npy(npy_path)
  cat("Data loaded, dimensions:", paste(dim(raw_data), collapse = "x"), "\n")

  inla_variables <- prepare_data(raw_data, scaling = SCALING)
  
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

  # 4. Create mesh and SPDE model
  inla_mesh <- create_inla_mesh(
    model_params$x, 
    model_params$y, 
    resolution = MAX_EDGE_RESOLUTION
  )

  spde_model <- define_spde_model(
    inla_mesh,
    nonstationary = FALSE,
    p_range = c(2, 0.2),
    p_sigma = c(2, 0.2)
  )

  # 5. Create projection matrix and model stack
  projection_matrix_A <- inla.spde.make.A(
    inla_mesh,
    loc = cbind(model_params$x, model_params$y)
  )

  model_stack <- prepare_model_stack(
    shape = 'none',
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
    spde = spde_model,
    tolerance = 1e-4,
    restart = 0L,
    shape = 'none'
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
    shape = 'none',
    xcenter = model_params$xcenter,
    ycenter = model_params$ycenter,
    eigens = model_stack$eigens
  )

  # 8. Apply inverse scaling and save
  final_results <- unscale_results(projected_results, scaling = SCALING)
  
  output_dir <- "INLA_output_NPY"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  fname <- sub("\\.npy$", "", basename(npy_path))
  out_path <- file.path(output_dir, fname)
  
  save_npy(final_results, out_path)
  
  cat("=== PIPELINE COMPLETED SUCCESSFULLY ===\n")
  cat("Results saved to:", out_path, "\n")

}, error = function(e) {
  cat("=== PIPELINE FAILED ===\n")
  cat("Error:", conditionMessage(e), "\n")
  quit(status = 1)
})
