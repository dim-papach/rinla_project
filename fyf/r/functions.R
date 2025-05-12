# Load necessary libraries (if not already loaded)
library(INLA)
library(reshape2)
library(rlang)
#load NPY file
library(reticulate)

# ---------------------------------------------------------------------------
# Step 1: Get Data from npy file
# ---------------------------------------------------------------------------

#' Load the path of the npy file from a txt file
#'
#' This function loads the path of the npy file from a text file.
#' #' @param file_path A character string specifying the path to the text file.
#' @return A character string containing the path to the npy file.
#' @examples
#' path <- load_path("path/to/file.txt")
load_path <- function(file_path) {
  # Read the file and extract the path
  path <- readLines(file_path, n = 1)
  return(path)
}

#' Get Data from a NPY File
#'
#' This function loads data from a NPY file and returns it as a matrix.
#'
#' @param file_path A character string specifying the path to the NPY file.
#'
#' @return A numeric matrix containing the data from the NPY file.
#'
#' @examples
#' data <- get_data("path/to/file.npy")
load_npy <- function(file_path) {
  #import numpy as np
  np <- reticulate::import("numpy")
  data <- t(np$load(file_path))
  return(data)
}

# ---------------------------------------------------------------------------
# Step 2: Prepare Data for INLA Analysis
# ---------------------------------------------------------------------------

#' Prepare Data for INLA Analysis
#'
#' This function prepares image data for INLA analysis by extracting dimensions,
#' creating coordinate matrices, identifying valid data points, and normalizing the image data.
#'
#' @param img A numeric matrix representing the image data to be analyzed.
#' @param scaling If scale=TRUE then turn the values to log
#'
#' @return A list containing prepared data for INLA analysis.
#'
#' @examples
#' img <- matrix(runif(100), nrow = 10, ncol = 10)
#' prepared_data <- prepare_data(img)
prepare_data <- function(img, scaling = TRUE) {
  # Check if the input is a valid matrix
  if (is.null(dim(img)) || length(dim(img)) != 2) {
    stop("Error: The image data is not a 2D matrix.")
  }

  # Get dimensions of the img array
  dims <- dim(img)
  cat("Image dimensions: ", dims, "\n") # Debugging output

  # Create x and y arrays using matrix indexing
  x <- matrix(rep(1:dims[1], dims[2]), nrow = dims[1], ncol = dims[2])
  y <- matrix(rep(1:dims[2], each = dims[1]), nrow = dims[1], ncol = dims[2])
  # replace blanks and null values with NA
  img[img == "BLANK"] <- NA
  img[img == "blank"] <- NA
  # Normalize data
  if (scaling==TRUE){
  logimg <- log10(img)
  logimg[is.infinite(logimg)] <- 0 # Replace -Inf and Inf values with 0
  }
  else{
    logimg <- img
  }
  # Identify valid data points
  valid <- which(!is.na(img) & !is.nan(img) & img != 0
                 & !is.infinite(logimg) & !is.na(logimg) & !is.nan(logimg))

  # Check if there are any valid points
  if (length(valid) == 0) {
    stop("Error: No valid data points found in the image.")
  }

  # Set dimensions
  xsize <- dims[1]
  ysize <- dims[2]
  xfin <- xsize
  yfin <- ysize


  return(list(
    x = x, y = y, valid = valid, xsize = xsize, ysize = ysize,
    xfin = xfin, yfin = yfin, logimg = logimg, img = img
  ))
}

# Breakdown of Stationary INLA

#' Check Data Validity for INLA Analysis
#'
#' This function performs checks on the data validity before proceeding with INLA analysis.
#'
#' @param valid A vector of indices indicating valid data points.
#' @param tx A matrix of x-coordinates.
#' @param ty A matrix of y-coordinates.
#'
#' @return NULL if checks pass; stops execution if checks fail.
#'
#' @examples
#' check_data_validity(valid, tx, ty, logimg)
check_data_validity <- function(valid, tx, ty, logimg, img) {
  # Print dimensions for debugging
  cat("Number of valid data points: ", length(valid), "\n")
  cat("Length of valid:", length(valid), "\n")
  cat("Dimensions of tx:", length(tx), "\n")
  cat("Dimensions of valid tx:", length(tx[valid]), "\n")
  cat("Dimensions of valid ty:", length(ty[valid]), "\n")
  cat("Dimensions of logimg:", length(logimg), "\n")

  if (length(valid) == 0) {
    stop("Error: No valid data points found for INLA analysis.")
  }
  if (length(valid) > length(tx)) {
    stop("Error: The valid indices are longer than tx dimensions.")
  }
  if (length(valid) > length(ty)) {
    stop("Error: The valid indices are longer than ty dimensions.")
  }
  if (length(valid) > length(logimg)) {
    stop("Error: The valid indices are longer than logimg dimensions.")
  }
  if (length(valid) != sum(!is.na(img) & img != 0)) {
    cat("Number of valid data points: ", length(valid), "\n")
    cat("Number of non-NA/non-zero values in img: ", sum(!is.na(img) & img != 0), "\n")
    stop("Error: The number of valid indices are  not the same as the Number of non-NA/non-zero values in img.")
  }
}


#' Compute Parameters for INLA Model
#'
#' This function computes the parameters required for the INLA model, including spatial coordinates and response variable.
#'
#' @param valid A vector of indices indicating valid data points.
#' @param tx A matrix of x-coordinates.
#' @param ty A matrix of y-coordinates.
#' @param logimg A matrix of log-transformed image data.
#' @param weight A numeric value representing the weight for the analysis.
#' @param tepar Optional error parameters (uncertainties), default is NULL.
#'
#' @return A list containing x, y, par, epar, xcenter, ycenter.
#'
#' @examples
#' params <- compute_parameters(valid, tx, ty, logimg, weight)
compute_parameters <- function(valid, tx, ty, logimg, weight, tepar = NULL) {
  # Compute x and y vectors
  x <- tx[valid]
  y <- ty[valid]

  # Extract response variable
  par <- logimg[valid]
  if (any(!is.finite(par))) {
    stop("Error: Non-finite values detected in the parameters for INLA.")
  }

  cat("Summary of 'par' values: \n")
  print(summary(par)) # Debugging output
  cat("First few 'par' values: ", head(par), "\n") # More detailed inspection

  # Error parameters (if available)
  if (!is.null(tepar)) {
    epar <- tepar^2
  } else {
    epar <- NULL
  }

  # Compute centers
  xcenter <- sum(x * weight) / sum(weight)
  ycenter <- sum(y * weight) / sum(weight)
  cat("Computed centers: xcenter = ", xcenter, ", ycenter = ", ycenter, "\n")

  return(list(x = x, y = y, par = par, epar = epar, xcenter = xcenter, ycenter = ycenter))
}

#' Create INLA Mesh
#'
#' This function creates an INLA mesh for spatial modeling.
#'
#' @param x A vector of x-coordinates.
#' @param y A vector of y-coordinates.
#'
#' @return An INLA mesh object.
#'
#' @examples
#' mesh <- create_inla_mesh(x, y, cutoff)
create_inla_mesh <- function(x, y, max.edge = NULL) {
  if (length(x) == 0 || length(y) == 0) {
      stop("Error: Insufficient points to create a mesh.")
  }
  # Calculate data range
  x_range <- diff(range(x))
  y_range <- diff(range(y))
  max.range <- max(x_range, y_range)
  
  # Set default max.edge if not provided
  if(is.null(max.edge)) {
    max.edge <- max.range / 10  # Default to 10% of maximum range
  }
  
  # Calculate cutoff as max.edge/6 (ensuring it's ≥ 1e-5)
  cutoff <- max(max.edge / 30, 1e-5)
  
  # Create mesh
  mesh <- tryCatch(
    INLA::inla.mesh.2d(
      loc = cbind(x, y),
      max.edge = c(max.edge, max.edge * 1.5),  # Inner and outer resolution
      cutoff = cutoff,
      offset = c(max.edge * 0.5, max.edge * 2)  # Boundary extensions
    ),
    error = function(e) stop("Mesh creation failed: ", e$message))
  
  return(mesh)
}

#' Define SPDE Model
#'
#' This function defines the SPDE model based on whether it's stationary or non-stationary.
#'
#' @param mesh An INLA mesh object.
#' @param nonstationary A logical value indicating whether to use a non-stationary model.
#' @param p_range A numeric vector representing the prior range for the Gaussian process.
#' @param p_sigma A numeric vector representing the prior sigma for the Gaussian process.
#' @param nbasis Number of basis functions for non-stationary model (if applicable).
#' @param degree Degree of the spline basis (if applicable).
#'
#' @return An SPDE model object.
#'
#' @examples
#' spde <- define_spde_model(mesh, nonstationary, p_range, p_sigma)
define_spde_model <- function(mesh, nonstationary, p_range, p_sigma, nbasis = 2, degree = 10) {
  if (nonstationary) {
    # Inverse scale: degree=10, n=2 (default values)
    basis.T <- inla.mesh.basis(mesh, type = "b.spline", n = nbasis, degree = degree)
    # Inverse range
    basis.K <- inla.mesh.basis(mesh, type = "b.spline", n = nbasis, degree = degree)

    spde <- inla.spde2.matern(mesh = mesh, alpha = 2,
                              B.tau = cbind(0, basis.T, basis.K * 0),
                              B.kappa = cbind(0, basis.T * 0, basis.K / 2))
  } else {
    # Priors for Gaussian process
    spde <- inla.spde2.pcmatern(mesh = mesh, alpha = 2,
                                prior.range = p_range,
                                prior.sigma = p_sigma)
  }
  return(spde)
}

#' Prepare Model Stack for INLA
#'
#' This function prepares the INLA stack based on the specified shape parameter.
#'
#' @param shape A character string representing the shape for the analysis.
#' @param x A vector of x-coordinates.
#' @param y A vector of y-coordinates.
#' @param par A vector of response variable values.
#' @param A The projection matrix from the mesh to the data locations.
#' @param spde The SPDE model object.
#' @param weight A numeric value representing the weight for the analysis.
#' @param xcenter The x-coordinate of the center.
#' @param ycenter The y-coordinate of the center.
#'
#' @return A list containing the INLA stack object and eigenvalues (if applicable).
#'
#' @examples
#' stk <- prepare_model_stack(shape, x, y, par, A, spde, weight, xcenter, ycenter)
prepare_model_stack <- function(shape, x, y, par, A, spde, weight, xcenter, ycenter) {
  if (shape == 'radius') {
    radius <- sqrt((x - xcenter)^2 + (y - ycenter)^2)
    radius_2 <- (x - xcenter)^2 + (y - ycenter)^2

    # Use parametric function of radius and radius^2
    stk <- inla.stack(data = list(par = par), A = list(A, 1, 1, 1),
                      effects = list(i = 1:spde$n.spde, m = rep(1, length(x)),
                                     radius = radius, radius_2 = radius_2), tag = 'est')
    eigens <- NULL
  } else if (shape == 'ellipse') {
    # Compute weighted covariance
    m_weights <- rep(weight, length(x))
    covar <- cov.wt(cbind(x, y), wt = m_weights)

    eigens <- eigen(covar$cov)
    ellipse <- (cbind(x - xcenter, y - ycenter) %*% (eigens$vectors[,1]))^2 / eigens$values[1] +
      (cbind(x - xcenter, y - ycenter) %*% (eigens$vectors[,2]))^2 / eigens$values[2]
    ellipse_2 <- ellipse^2

    # Use parametric function of ellipse and ellipse^2
    stk <- inla.stack(data = list(par = par), A = list(A, 1, 1, 1),
                      effects = list(i = 1:spde$n.spde, m = rep(1, length(x)),
                                     ellipse = ellipse, ellipse_2 = ellipse_2), tag = 'est')
  } else if (shape == 'none') {
    # No additional spatial covariates
    stk <- inla.stack(data = list(par = par), A = list(A, 1),
                      effects = list(i = 1:spde$n.spde, m = rep(1, length(x))), tag = 'est')
    eigens <- NULL
  } else {
    stop("Error: Invalid shape parameter.")
  }

  return(list(stk = stk, eigens = eigens))
}

#' Run INLA Model
#'
#' This function runs the INLA model based on the prepared stack and SPDE model.
#'
#' @param stk The INLA stack object.
#' @param par A vector of response variable values.
#' @param epar Error parameters (uncertainties), default is NULL.
#' @param spde The SPDE model object.
#' @param tolerance A numeric value representing the tolerance for the INLA algorithm.
#' @param restart An integer value representing the number of restarts for the INLA algorithm.
#' @param shape The shape parameter.
#'
#' @return The result object from the INLA model.
#'
#' @examples
#' res <- run_inla_model(stk, par, epar, spde, tolerance, restart, shape)
run_inla_model <- function(stk, par, epar, spde, tolerance, restart, shape) {
  if (is.null(stk) || !inherits(stk, "inla.data.stack")) {
    stop("'stack' must inherit from class \"inla.data.stack\".")
  }
  # Determine the formula based on shape
  if (shape == 'radius') {
    formula <- par ~ 0 + m + radius + radius_2 + f(i, model = spde)
  } else if (shape == 'ellipse') {
    formula <- par ~ 0 + m + ellipse + ellipse_2 + f(i, model = spde)
  } else if (shape == 'none') {
    formula <- par ~ 0 + m + f(i, model = spde)
  } else {
    stop("Error: Invalid shape parameter.")
  }

  # Run the INLA model
  res <- inla(formula,
              data = inla.stack.data(stk),
              control.predictor = list(A = inla.stack.A(stk)),
              scale = epar,
              control.compute = list(openmp.strategy = 'huge'),
              control.inla = list(tolerance = tolerance, restart = restart),
              verbose = inla.getOption("verbose"))

  return(res)
}

#' Project INLA Results
#'
#' This function projects the INLA results onto a grid for visualization and further analysis.
#'
#' @param mesh An INLA mesh object.
#' @param res The result object from the INLA model.
#' @param xini The initial x-coordinate.
#' @param xfin The final x-coordinate.
#' @param yini The initial y-coordinate.
#' @param yfin The final y-coordinate.
#' @param xsize The number of columns in the image data.
#' @param ysize The number of rows in the image data.
#' @param zoom A numeric value representing the zoom factor for the analysis.
#' @param shape The shape parameter.
#' @param xcenter The x-coordinate of the center.
#' @param ycenter The y-coordinate of the center.
#' @param eigens Eigen decomposition object (if applicable).
#' @param spde The SPDE model object.
#'
#' @return A list containing output and outputsd matrices.
#'
#' @examples
#' projections <- project_inla_results(mesh, res, xini, xfin, yini,
#'                                     yfin, xsize, ysize, zoom, shape,
#'                                     xcenter, ycenter, eigens)
project_inla_results <- function(mesh, res, xini, xfin, yini, yfin, xsize,
                                 ysize, zoom, shape, xcenter, ycenter,
                                 eigens, spde) {
  # Create projector
  projector <- inla.mesh.projector(mesh, xlim = c(xini, xfin), ylim = c(yini, yfin),
                                   dim = zoom * c(xsize + 1, ysize + 1))

  if (shape == 'radius') {
    # Projection for radius
    px <- rep(projector$x, each = length(projector$y))
    py <- rep(projector$y, length(projector$x))
    projected_radius <- sqrt((px - xcenter)^2 + (py - ycenter)^2)
    projected_radius_2 <- projected_radius^2

    # Output with matrix to include radius function
    output <- inla.mesh.project(projector, res$summary.random$i$mean) +
      t(matrix(as.numeric(res$summary.fixed$mean[1] +
                            res$summary.fixed$mean[2] * projected_radius +
                            res$summary.fixed$mean[3] * projected_radius_2),
               nrow = zoom * (ysize + 1), ncol = zoom * (xsize + 1)))

  } else if (shape == 'ellipse') {
    # Projection for ellipse
    px <- rep(projector$x, each = length(projector$y))
    py <- rep(projector$y, length(projector$x))
    projected_ellipse <- (cbind(px - xcenter, py - ycenter) %*% (eigens$vectors[,1]))^2 / eigens$values[1] +
      (cbind(px - xcenter, py - ycenter) %*% (eigens$vectors[,2]))^2 / eigens$values[2]
    projected_ellipse_2 <- projected_ellipse^2

    # Output with matrix to include ellipse function
    output <- inla.mesh.project(projector, res$summary.random$i$mean) +
      t(matrix(as.numeric(res$summary.fixed$mean[1] +
                            res$summary.fixed$mean[2] * projected_ellipse +
                            res$summary.fixed$mean[3] * projected_ellipse_2),
               nrow = zoom * (ysize + 1), ncol = zoom * (xsize + 1)))

  } else if (shape == 'none') {
    # Output without additional spatial functions
    output <- inla.mesh.project(projector, res$summary.random$i$mean) +
      t(matrix(as.numeric(res$summary.fixed$mean[1]),
               nrow = zoom * (ysize + 1), ncol = zoom * (xsize + 1)))
  } else {
    stop("Error: Invalid shape parameter.")
  }

  # Output standard deviation
  outputsd <- inla.mesh.project(projector, res$summary.random$i$sd)

  return(list(output = output, outputsd = outputsd))
}

#' Adjust Output for Zoom Factor
#'
#' This function adjusts the output matrices if a zoom factor is applied.
#'
#' @param output The output matrix from the projection.
#' @param outputsd The output standard deviation matrix from the projection.
#' @param zoom The zoom factor applied.
#'
#' @return A list containing adjusted output and outputsd matrices.
#'
#' @examples
#' adjusted_outputs <- adjust_zoom(output, outputsd, zoom)
adjust_zoom <- function(output, outputsd, zoom) {
  if (zoom != 1) {
    # Placeholder for zoom adjustment logic
    # Since zoom_fix function is not defined, we can resample the matrices
    # For now, we'll assume the matrices are already adjusted for zoom
    # Alternatively, we can include code to adjust the matrices
    # For now, we can return the matrices as is
    # TODO: Implement zoom adjustment if necessary
    warning("Zoom adjustment not implemented. Output matrices may not be correctly adjusted for zoom.")
  }
  return(list(output = output, outputsd = outputsd))
}

#' Unscale Collected INLA Results Dynamically
#'
#' This function iterates over all elements of the collected results from `collect_inla_results`
#' and optionally applies 10^ to numerical matrices, vectors, or scalars if `scale = TRUE`.
#'
#' @param collected A list of collected results from `collect_inla_results`.
#' @param scale A logical value. If `TRUE`, applies 10^ to all applicable elements.
#'
#' @return The modified list of collected results with unscaled values if `scale = TRUE`.
#'
#' @examples
#' collected <- collect_inla_results(...)
#' unscaled_results <- unscale_collected(collected, scale = TRUE)
unscale_collected <- function(collected, scaling = FALSE) {
  if (scaling) {
    # Iterate through each element in the list
    collected <- lapply(collected, function(x) {
      if (is.numeric(x)) {
        # Apply 10^ only to numeric elements (matrices, vectors, or scalars)
        10^x
      } else {
        # Leave non-numeric elements unchanged
        x
      }
    })
  }

  return(collected)
}

#'------------------------------------------------------------------------------
#' Save the results in npy files
#'------------------------------------------------------------------------------

#' Save multiple arrays from a named list to .npy files using numpy
#'
#' This function saves each matrix or array in a named list to a `.npy` file.
#' Files are saved in the specified directory with filenames based on the list names.
#'
#' @param array_list A named list where each element is a matrix or array.
#' @param dir_path A string specifying the output directory for the .npy files.
#'
#' @return Invisibly returns a character vector of the full paths to the saved files.
#' @examples
#' \dontrun{
#'   arrays <- list(
#'     a = matrix(1:4, 2, 2),
#'     b = matrix(5:8, 2, 2)
#'   )
#'   save_npy_list(arrays, "output_dir")
#' }

save_npy <- function(array_list, dir_path) {
  if (!is.list(array_list)) {
    stop("Input must be a list of matrices or arrays.")
  }
  if (is.null(names(array_list)) || any(names(array_list) == "")) {
    stop("Input list must have named elements. Ensure all elements are named.")
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


#' Create a Mesh Projector Object
#'
#' Initializes a projector for mapping INLA model results onto a grid.
#'
#' @param mesh An INLA mesh object.
#' @param xlim Numeric vector of length 2: x-axis limits (min, max).
#' @param ylim Numeric vector of length 2: y-axis limits (min, max).
#' @param zoom Numeric: Grid resolution multiplier.
#' @param xsize Integer: Base grid dimension in x-direction.
#' @param ysize Integer: Base grid dimension in y-direction.
#' @return An `inla.mesh.projector` object.
#' @examples
#' # projector <- create_projector(mesh, c(0, 10), c(0, 10), zoom = 2, 100, 100)
create_projector <- function(mesh, xlim, ylim, zoom, xsize, ysize) {
  inla.mesh.projector(
    mesh,
    xlim = xlim,
    ylim = ylim,
    dim = zoom * c(xsize, ysize))
}


#' Compute Spatial Trend Component
#'
#' Calculates fixed-effects spatial trend (radius/ellipse/none) for projection.
#'
#' @param projector An `inla.mesh.projector` object.
#' @param shape Character: Spatial trend type ('radius', 'ellipse', or 'none').
#' @param res INLA model result object.
#' @param xcenter Numeric: X-coordinate of trend center.
#' @param ycenter Numeric: Y-coordinate of trend center.
#' @param eigens List containing eigenvectors/values (required for 'ellipse').
#' @return Numeric vector of spatial trend values at grid locations.
#' @note For 'ellipse', eigenvectors define axes orientation, eigenvalues define scaling.
compute_spatial_term <- function(projector, shape, res, xcenter, ycenter, eigens) {
  # Generate grid coordinates
  px <- rep(projector$x, each = length(projector$y))
  py <- rep(projector$y, length(projector$x))
  
  if (shape == 'radius') {
    # Radial distance from center point
    projected <- sqrt((px - xcenter)^2 + (py - ycenter)^2)
    term <- res$summary.fixed$mean[1] + 
      res$summary.fixed$mean[2] * projected + 
      res$summary.fixed$mean[3] * projected^2
    
  } else if (shape == 'ellipse') {
    # Mahalanobis distance using eigenvalue decomposition
    centered_coords <- cbind(px - xcenter, py - ycenter)
    projected <- (centered_coords %*% eigens$vectors[, 1])^2 / eigens$values[1] +
      (centered_coords %*% eigens$vectors[, 2])^2 / eigens$values[2]
    term <- res$summary.fixed$mean[1] + 
      res$summary.fixed$mean[2] * projected + 
      res$summary.fixed$mean[3] * projected^2
    
  } else if (shape == 'none') {
    # Constant intercept only
    term <- res$summary.fixed$mean[1]
  } else {
    stop("Invalid shape parameter. Use 'radius', 'ellipse', or 'none'.")
  }
  
  return(term)
}


#' Process Validation Data
#'
#' Converts validation data into grid-aligned matrices for comparison.
#'
#' @param valid Logical/numeric vector: Indices of validation points.
#' @param tx Numeric vector: X-coordinates of full dataset.
#' @param ty Numeric vector: Y-coordinates of full dataset.
#' @param logimg Numeric vector: Observed values (log-transformed).
#' @param weight Numeric vector: Observation weights.
#' @param tepar Numeric vector: Tapering parameters.
#' @param mesh INLA mesh object containing spatial locations.
#' @return List with:
#'   - timage: Matrix of validation values in grid space
#'   - terrimage: Matrix of validation errors (if available)
#'   - x,y: Rescaled coordinates
#'   - z,erz: Values and errors at validation points
process_validation <- function(valid, tx, ty, logimg, weight, tepar, mesh) {
  # Compute parameters using user-defined function
  params <- compute_parameters(valid, tx, ty, logimg, weight, tepar)
  
  # Extract validation locations from mesh
  mim <- mesh$loc[valid, ]
  
  # Create gridded observation matrix
  timage <- matrix(NA, 
                   nrow = length(unique(ty)), 
                   ncol = length(unique(tx)))
  timage[cbind(as.numeric(factor(ty[valid])),  # Map to grid indices
               as.numeric(factor(tx[valid])))] <- params$par
  
  # Create error matrix if available
  terrimage <- if (!is.null(params$epar)) {
    matrix(NA, nrow = length(unique(ty)), ncol = length(unique(tx)))
    terrimage[cbind(as.numeric(factor(ty[valid])), 
                    as.numeric(factor(tx[valid])))] <- sqrt(params$epar)
    terrimage
  } else NULL
  
  list(
    timage = timage,
    terrimage = terrimage,
    x = mim$x,
    y = mim$y,
    z = params$par,
    erz = if (!is.null(params$epar)) sqrt(params$epar) else NULL
  )
}


#' Project INLA Results with Validation Support
#'
#' Main function to project INLA model results onto a grid, optionally incorporating validation data.
#'
#' @param mesh INLA mesh object.
#' @param res INLA model result object.
#' @param xini,xfin Numeric: X-axis limits.
#' @param yini,yfin Numeric: Y-axis limits.
#' @param xsize,ysize Integer: Base grid dimensions.
#' @param zoom Numeric: Resolution multiplier.
#' @param shape Character: Spatial trend type.
#' @param xcenter,ycenter Numeric: Center coordinates for spatial trend.
#' @param eigens List: Eigenvectors/values for ellipse (if needed).
#' @param spde SPDE model object (unused in projection, retained for compatibility).
#' @param valid Optional: Validation point indices.
#' @param tx,ty Optional: Coordinate vectors for validation data.
#' @param logimg Optional: Observed values for validation.
#' @param weight Optional: Weights for validation.
#' @param tepar Optional: Taper parameters for validation.
#' @return List containing:
#'   - out: Projected mean field matrix
#'   - outsd: Projected standard deviation matrix
#'   - image: Validation observations in grid space
#'   - erimage: Validation errors in grid space
#'   - x,y,z,erz: Validation data in original coordinates
#' @examples
#' # Basic projection
#' results <- project_inla_results_collect(mesh, res, 0, 10, 0, 10, 100, 100, 2, 'none')
#'
#' # With validation data
#' val_results <- project_inla_results_collect(mesh, res, 0, 10, 0, 10, 100, 100, 2, 'radius',
#'                                            valid = val_indices, tx = xcoords, ty = ycoords)
project_inla_results_collect <- function(mesh, res, xini, xfin, yini, yfin, xsize, ysize, zoom, 
                                         shape, xcenter, ycenter, eigens, spde, valid = NULL, 
                                         tx = NULL, ty = NULL, logimg = NULL, weight = NULL, 
                                         tepar = NULL) {
  if (is.null(mesh) || is.null(res)) {
      stop("Error: Mesh and result inputs cannot be NULL.")
  }
  # 1. Initialize projection grid
  projector <- create_projector(mesh, c(xini, xfin), c(yini, yfin), zoom, xsize, ysize)
  
  # 2. Adjust center coordinates if validation data provided
  if (!is.null(valid)) {
    params <- compute_parameters(valid, tx, ty, logimg, weight, tepar)
    xcenter <- params$xcenter
    ycenter <- params$ycenter
  }
  
  # 3. Calculate spatial trend component
  spatial_term <- compute_spatial_term(projector, shape, res, xcenter, ycenter, eigens)
  
  # 4. Project random effects and combine with trend
  random_effects <- inla.mesh.project(projector, res$summary.random$i$mean)
  output <- (random_effects) + 
   t(matrix(spatial_term, nrow = zoom * (ysize), ncol = zoom * (xsize)))
  
  # 5. Project standard deviations
  outputsd <- inla.mesh.project(projector, res$summary.random$i$sd)
  
  # 6. Process validation data if provided
  validation_data <- if (!is.null(valid)) {
    process_validation(valid, tx, ty, logimg, weight, tepar, mesh)
  } else NULL

  #7. Check if out and image dimensions match
  if (!is.null(validation_data)) {
    if (any(dim(output) != dim(validation_data$timage))) {
      stop("Error: Dimensions of output and validation data do not match.")
    }
    else {
      print("HURRAY")
    }
  }
  else {print("empty")}
  
  # 8. Return comprehensive results
  list(
    out = t(output),
    outsd = t(outputsd),
    image = validation_data$timage %||% NULL,  # Using %||% for null coalescing
    erimage = validation_data$terrimage %||% NULL,
    x = validation_data$x %||% NULL,
    y = validation_data$y %||% NULL,
    z = validation_data$z %||% NULL,
    erz = validation_data$erz %||% NULL
  )
}