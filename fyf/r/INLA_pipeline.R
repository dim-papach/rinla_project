# INLA_pipeline.R
library(INLA)
library(reshape2)
library(ggplot2)
library(viridis)
library(optparse)
library(rlang)

cat("Debug: Loaded required libraries\n")

# Source your custom functions from R/functions.R
cat("Debug: Sourcing fyf/r/functions.R\n")
source("fyf/r/functions.R")
cat("Debug: Current working directory before setwd:", getwd(), "\n")

# Set working directory with here to this file
setwd(here::here())
cat("Debug: Current working directory after setwd:", getwd(), "\n")

scalingg <- TRUE
cat("Debug: scalingg set to TRUE\n")
inla.setOption(num.threads = 6)
cat("Debug: INLA num.threads set to 6\n")

# ---- Pipeline Execution ----

tryCatch({
  # 1. Load path
  file_path <- "variants/path.txt"
  cat("Debug: Looking for path file at", file_path, "\n")
  # if no path.txt file, use the first argument as the path
  if (!file.exists(file_path)) {
    cat("Debug: path.txt not found, checking commandArgs\n")
    file_path <- commandArgs(trailingOnly = TRUE)[1]
    if (is.na(file_path) || !file.exists(file_path)) {
      stop("No valid path provided. Please create a path.txt file or provide a path as an argument.")
    }
  }
  npy_path <- readLines(file_path)
  cat("Debug: npy_path loaded:", npy_path, "\n")
  
  # 2. Load and process data
  cat("Debug: Loading npy data from", npy_path, "\n")
  raw_data <- load_npy(npy_path)
  cat("Debug: raw_data loaded, dim:", paste(dim(raw_data), collapse = "x"), "\n")

  # 2.5 Plot the image (optional)
  # image(t(raw_data)[, nrow(raw_data):1], col = heat.colors(256), main = "Corrected Orientation")

  # 3. Prepare data
  print("Prepare_data")
  cat("Debug: Calling prepare_data\n")
  inla_variables <- prepare_data(raw_data, scaling = scalingg)
  cat("Debug: prepare_data returned, names:", paste(names(inla_variables), collapse = ", "), "\n")

  # 4. Validate data
  print("Check_data_validity")
  cat("Debug: Calling check_data_validity\n")
  check_data_validity(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    img = inla_variables$img
  )
  cat("Debug: check_data_validity completed\n")

  # 5. Compute parameters
  print("Compute_parameters")
  cat("Debug: Calling compute_parameters\n")
  model_params <- compute_parameters(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    weight = 1
  )
  cat("Debug: compute_parameters returned, names:", paste(names(model_params), collapse = ", "), "\n")

  # 6. Create mesh
  print("Create_mesh")
  cat("Debug: Calling create_inla_mesh\n")
  inla_mesh <- create_inla_mesh(model_params$x, model_params$y)
  cat("Debug: create_inla_mesh returned\n")

  # 7. Define SPDE model
  print("Define_SPDE_model")
  cat("Debug: Calling define_spde_model\n")
  spde_model <- define_spde_model(
    inla_mesh,
    nonstationary = FALSE,
    p_range = c(2, 0.2),
    p_sigma = c(2, 0.2)
  )
  cat("Debug: define_spde_model returned\n")

  # 8. Create projection matrix
  print("Create_projection_matrix")
  cat("Debug: Calling inla.spde.make.A\n")
  projection_matrix_A <- inla.spde.make.A(inla_mesh,
                                          loc = cbind(model_params$x, model_params$y))
  cat("Debug: inla.spde.make.A returned, dim:", paste(dim(projection_matrix_A), collapse = "x"), "\n")

  # 9. Prepare model stack
  print("Prepare_model_stack")
  cat("Debug: Calling prepare_model_stack\n")
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
  cat("Debug: prepare_model_stack returned, names:", paste(names(model_stack), collapse = ", "), "\n")

  # 10. Run INLA model
  print("INLA results")
  cat("Debug: Calling run_inla_model\n")
  inla_result <- run_inla_model(
    stk = model_stack$stk,
    par = model_params$par,
    epar = model_params$epar,
    spde = spde_model,
    tolerance = 1e-4,
    restart = 0L,
    shape = 'none'
  )
  cat("Debug: run_inla_model returned\n")

  # 11. Project results
  print("Project results")
  cat("Debug: Calling project_inla_results_collect\n")
  projected_results <- project_inla_results_collect(
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
  cat("Debug: project_inla_results_collect returned, dim:", paste(dim(projected_results), collapse = "x"), "\n")

  # 12. Collect results
  print("INLA collect")
  cat("Debug: Calling project_inla_results_collect again for collection\n")
  inla_results_collected <- project_inla_results_collect(
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
  cat("Debug: project_inla_results_collect (collection) returned, dim:", paste(dim(inla_results_collected), collapse = "x"), "\n")

  # 13. Unscale results
  print("Unscale results")
  cat("Debug: Calling unscale_collected\n")
  unscaled_results <- unscale_collected(inla_results_collected, scaling = scalingg)
  cat("Debug: unscale_collected returned, dim:", paste(dim(unscaled_results), collapse = "x"), "\n")

  # 13.5 Plot the image (optional)
  # image(t(unscaled_results)[, nrow(unscaled_results):1], col = heat.colors(256), main = "Corrected Orientation")

  # 14. Save output
  print("Save output")
  output_dir <- "INLA_output_NPY"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
    cat("Debug: Created output directory", output_dir, "\n")
  }
  # fname is npy_path without the npy extension
  fname <- sub("\\.npy$", "", basename(npy_path))
  out_path <- file.path(output_dir, fname)
  cat("Debug: Saving output to", out_path, "\n")
  save_npy(unscaled_results, out_path)
  cat("Debug: Output saved to", out_path, "\n")

  message("\n ✅  Pipeline completed successfully. Results saved to:\n", out_path)

}, error = function(e) {
  message("\n ❌ Pipeline failed with error:\n", conditionMessage(e))
  quit(status = 1)

  # 13. Save output (replace existing steps 13-14)
  print("Saving results")
  output_dir <- "INLA_output"  # Base directory for all outputs
  prefix <- sub("\\.npy$", "", basename(npy_path))  # Use input filename as prefix

  # Save results (NPY + PNG by default)
  saved_files <- save_inla_results(
    results = inla_results_collected,
    base_path = output_dir,
    prefix = prefix,
    save_npy = TRUE,
    save_csv = FALSE,  # Set to TRUE if CSV is needed
    save_png = TRUE,
    scale = FALSE  # Apply 10^x scaling (same as unscale_collected)
  )

  message("\n ✅  Pipeline completed successfully. Results saved to:\n", output_dir)
  message("NPY files: ", paste(saved_files$npy, collapse = "\n"))
  message("PNG plots: ", paste(saved_files$png, collapse = "\n"))
})
