# Load necessary libraries (if not already loaded)
library(INLA)
library(FITSio)
library(reshape2)

# ---------------------------------------------------------------------------
# Step 1: Get Data from a FITS File
# ---------------------------------------------------------------------------

#' Get Data from a FITS File
#'
#' This function reads image data from a FITS file.
#'
#' @param file_path A character string specifying the path to the FITS file.
#'
#' @return A matrix containing the image data from the FITS file.
#'
#' @examples
#' image_data <- get_data("path/to/file.fits")
get_data <- function(file_path) {
  # Read the FITS file
  fits_data <- readFITS(file_path)
  
  # Check if the FITS file contains image data
  if (is.null(fits_data$imDat)) {
    stop("Error: No image data found in the FITS file.")
  }
  
  # Extract the image data from the FITS file
  image_data <- fits_data$imDat
  
  return(image_data)
}

#' Get Header from a FITS File
#'
#' This function reads image header from a FITS file.
#'
#' @param file_path A character string specifying the path to the FITS file.
#'
#' @return Header table of the original FITS file
#'
#' @examples
#' image_data <- get_data("path/to/file.fits")
get_header <- function(file_path) {
  # Read the FITS file
  fits_data <- readFITS(file_path)
  
  # Check if the FITS file contains image data
  if (is.null(fits_data$header)) {
    stop("Error: No image data found in the FITS file.")
  }
  
  # Extract the image data from the FITS file
  image_data <- fits_data$header
  
  return(image_data)
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
#'
#' @return A list containing prepared data for INLA analysis.
#'
#' @examples
#' img <- matrix(runif(100), nrow = 10, ncol = 10)
#' prepared_data <- prepare_data(img)
prepare_data <- function(img) {
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
  # Normalize data
  logimg <- log10(img)
  logimg[is.infinite(logimg)] <- 0 # Replace -Inf and Inf values with 0
  
  # Identify valid data points
  valid <- which(!is.na(img) & !is.nan(img) & img != 0
                 & !is.infinite(logimg) & !is.na(logimg) & !is.nan(logimg))
  
  # Check if there are any valid points
  if (length(valid) == 0) {
    stop("Error: No valid data points found in the image.")
  }
  
  # Set dimensions
  xsize <- dims[2]
  ysize <- dims[1]
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
#' @param cutoff A numeric value representing the cutoff for the mesh tessellation.
#'
#' @return An INLA mesh object.
#'
#' @examples
#' mesh <- create_inla_mesh(x, y, cutoff)
create_inla_mesh <- function(x, y, cutoff) {
  # Create a mesh (tessellation)
  mesh <- tryCatch(
    inla.mesh.2d(cbind(x, y), max.n = -1, cutoff = max(cutoff,1e-5)),
    error = function(e) stop("Error creating INLA mesh: ", e$message)
  )
  # Check that the mesh was created successfully and has vertices
  if (is.null(mesh) || length(mesh$loc) == 0) {
    stop("Error: Mesh creation failed or resulted in an empty mesh.")
  }
  
  cat("Number of mesh vertices: ", nrow(mesh$loc), "\n") # Debugging output
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

#' Collect INLA Results
#'
#' This function collects and organizes the INLA results for output, ensuring 
#' that the dimensions of the output matrices match those of the original grid.
#'
#' @param output The output matrix from the projection.
#' @param outputsd The output standard deviation matrix from the projection.
#' @param x The x-coordinates used in the model.
#' @param y The y-coordinates used in the model.
#' @param par The response variable values.
#' @param epar Error parameters (uncertainties), default is NULL.
#' @param xsize The number of columns in the image data.
#' @param ysize The number of rows in the image data.
#' @param xini The initial x-coordinate.
#' @param xfin The final x-coordinate.
#' @param yini The initial y-coordinate.
#' @param yfin The final y-coordinate.
#' @param zoom The zoom factor applied.
#'
#' @return A list containing:
#' - `out`: The original output matrix.
#' - `image`: A matrix with the mapped response variable.
#' - `erimage`: A matrix with mapped uncertainties (if provided).
#' - `outsd`: The original output standard deviation matrix.
#' - `x`, `y`, `z`: Scaled and reshaped coordinates and values.
#' - `erz`: Reshaped standard deviation values.
#'
#' @examples
#' final_results <- collect_inla_results(output, outputsd, x, y, par, epar, 
#'                                       xsize, ysize, xini, xfin, 
#'                                       yini, yfin, zoom)
collect_inla_results <- function(output, outputsd, x, y, par, epar, 
                                 xsize, ysize, xini, xfin, 
                                 yini, yfin, zoom) {
  # Step 1: Calculate bin edges for the original grid
  # Create sequences for x and y bins based on grid boundaries
  xbins <- seq(xini, xfin, length.out = xsize + 1)
  ybins <- seq(yini, yfin, length.out = ysize + 1)
  
  # Step 2: Map coordinates to grid indices
  # Use findInterval to map x and y coordinates to bin indices
  xmat <- findInterval(x, xbins, all.inside = TRUE)
  ymat <- findInterval(y, ybins, all.inside = TRUE)
  
  # Step 3: Initialize response variable image matrix
  # Create a matrix for the response variable with dimensions matching the original grid
  timage <- matrix(NA, nrow = xsize + 1, ncol = ysize + 1)
  for (i in seq_along(x)) {
    timage[xmat[i], ymat[i]] <- par[i]
  }
  
  # Step 4: Initialize uncertainty matrix if provided
  terrimage <- NULL
  if (!is.null(epar)) {
    # Create a matrix for uncertainties with the same dimensions
    terrimage <- matrix(NA, nrow = xsize + 1, ncol = ysize + 1)
    for (i in seq_along(x)) {
      terrimage[xmat[i], ymat[i]] <- epar[i]
    }
  }
  
  # Step 5: Reshape the output matrices for visualization
  # Melt the output matrix and adjust coordinates back to original scale
  mim <- reshape2::melt(output)
  colnames(mim) <- c("x", "y", "value")
  mim$x <- (mim$x - 1) / zoom + xini
  mim$y <- (mim$y - 1) / zoom + yini
  zz <- mim$value  # Extract the reshaped values
  
  # Repeat the process for the standard deviation matrix
  sdmim <- reshape2::melt(outputsd)
  colnames(sdmim) <- c("x", "y", "value")
  erzz <- sdmim$value  # Extract the reshaped standard deviations
  
  # Step 6: Return all collected and reshaped results
  return(list(out = output,         # Original output matrix
              image = timage,       # Response variable matrix
              erimage = terrimage,  # Uncertainty matrix (if provided)
              outsd = outputsd,     # Original output standard deviation matrix
              x = mim$x,            # Rescaled x-coordinates
              y = mim$y,            # Rescaled y-coordinates
              z = zz,               # Reshaped response values
              erz = erzz))          # Reshaped standard deviation values
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


# ---------------------------------------------------------------------------
# Step 3: Save INLA Results as FITS Files
# ---------------------------------------------------------------------------

#' Save INLA Results as FITS Files
#'
#' This function saves the original, reconstructed, and standard deviation images as FITS files.
#'
#' @param imginla A list containing the INLA analysis results.
#' @param output_dir A character string specifying the directory to save the FITS files.
#'
#' @examples
#' save_fits(imginla, output_dir = "INLA_fits_output")
save_fits <- function(imginla, header_data, output_dir = "INLA_fits_output"){
  if(!dir.exists(output_dir)){
    dir.create(output_dir)
  }
  
  # Define file names
  original_image_file <- file.path(output_dir,  "Original.fits")
  reconstructed_image_file <- file.path(output_dir, "Reconstructed.fits")
  sd_image_file <- file.path(output_dir, "SD.fits")
  
  # Save FITS files
  writeFITSim(imginla$image, file = original_image_file, header = header_data)
  writeFITSim(imginla$out, file = reconstructed_image_file, header = header_data)
  writeFITSim(imginla$outsd, file = sd_image_file, header = header_data)
  return(c(original_image_file, reconstructed_image_file, sd_image_file))
}

#' Plot INLA Analysis Results
#'
#' This function plots and saves various plots related to the INLA analysis results.
#'
#' @param inla_result A list containing the INLA analysis results.
#' @param title_prefix A character string to prefix the titles of the plots.
#' @param output_dir A character string specifying the directory to save the plots.
#'
#' @examples
#' plot_inla(imginla, title_prefix = "INLA_Result")
plot_inla <- function(inla_result, title_prefix = "INLA_Result", output_dir = "plots", scaling = TRUE) {
  # Create the output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  # Extract the components from the result
  reconstructed_image <- inla_result$out
  original_image <- inla_result$image
  error_image <- inla_result$erimage
  reconstructed_sd <- inla_result$outsd
  if (scaling) {
    # Iterate through each element in the list
    x <- log10(inla_result$x)
    y <- log10(inla_result$y)
    z <- log10(inla_result$z)
    erz <- log10(inla_result$erz)}
      else {
        x <- inla_result$x
        y <- inla_result$y
        z <- inla_result$z
        erz <- inla_result$erz
      }
    
  

  
  # Define file names
  original_image_file <- file.path(output_dir, paste0(title_prefix, "_Original_Image.png"))
  reconstructed_image_file <- file.path(output_dir, paste0(title_prefix, "_Reconstructed_Image.png"))
  error_image_file <- file.path(output_dir, paste0(title_prefix, "_Error_Image.png"))
  reconstructed_sd_file <- file.path(output_dir, paste0(title_prefix, "_Reconstruction_SD.png"))
  scatter_plot_file <- file.path(output_dir, paste0(title_prefix, "_Scatter_Plot.png"))
  error_scatter_plot_file <- file.path(output_dir, paste0(title_prefix, "_Error_Scatter_Plot.png"))
  
  # Save the original image
  png(original_image_file)
  image(original_image, col = terrain.colors(256), main = paste(title_prefix, " - Original Image"))
  dev.off()
  
  # Save the reconstructed image
  png(reconstructed_image_file)
  image(reconstructed_image, col = terrain.colors(256), main = paste(title_prefix, " - Reconstructed Image"))
  dev.off()
  
  # Save the error image if it exists
  if (!is.null(error_image)) {
    png(error_image_file)
    image(error_image, col = terrain.colors(256), main = paste(title_prefix, " - Error Image"))
    dev.off()
  }
  
  # Save the standard deviation of the reconstruction
  png(reconstructed_sd_file)
  image(reconstructed_sd, col = terrain.colors(256), main = paste(title_prefix, " - Reconstruction SD"))
  dev.off()
  
  # Save the scatter plot of x, y, z values
  png(scatter_plot_file)
  plot(x, y, col = terrain.colors(256)[cut(z, 256)], pch = 19, main = paste(title_prefix, " - Scatter Plot"))
  dev.off()
  
  # Save the scatter plot of x, y, erz values
  png(error_scatter_plot_file)
  plot(x, y, col = terrain.colors(256)[cut(erz, 256)], pch = 19, main = paste(title_prefix, " - Error Scatter Plot"))
  dev.off()
  
  # Optionally, display the plots in the R console
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  image(original_image, col = terrain.colors(256), main = paste(title_prefix, " - Original Image"))
  image(reconstructed_image, col = terrain.colors(256), main = paste(title_prefix, " - Reconstructed Image"))
  if (!is.null(error_image)) {
    image(error_image, col = terrain.colors(256), main = paste(title_prefix, " - Error Image"))
  }
  image(reconstructed_sd, col = terrain.colors(256), main = paste(title_prefix, " - Reconstruction SD"))
  plot(x, y, col = terrain.colors(256)[cut(z, 256)], pch = 19, main = paste(title_prefix, " - Scatter Plot"))
  plot(x, y, col = terrain.colors(256)[cut(erz, 256)], pch = 19, main = paste(title_prefix, " - Error Scatter Plot"))
}
