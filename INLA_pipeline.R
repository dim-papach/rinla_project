# INLA_pipeline.R
library(INLA)
library(reshape2)
library(ggplot2)
library(viridis)
library(optparse)
library(rlang)

# Set working directory
# this_file <- dirname(normalizePath(sys.frame(1)$ofile))
# setwd(this_file)
# Source your custom functions from R/functions.R
source("R/functions.R")

inla.setOption(num.threads = 6)

# ---- Pipeline Execution ----


tryCatch({
  # 1. Load path
  file_path <- "variants/path.txt"
  npy_path <- readLines(file_path, n=1)

  # 2. Load and process data
  raw_data <- load_npy(npy_path)

  # 2.5 Plot the image
  #image(t(raw_data)[, nrow(raw_data):1], col = heat.colors(256), main = "Corrected Orientation")
  # 3. Prepare data
  print("Prepare_data")
  inla_variables <- prepare_data(raw_data, scaling = FALSE)
  
  # 4. Validate data
  print("Check_data_validity")
  check_data_validity(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    img = inla_variables$img
  )

  # 5. Compute parameters
  print("Compute_parameters")
  model_params <- compute_parameters(
    valid = inla_variables$valid,
    tx = inla_variables$x,
    ty = inla_variables$y,
    logimg = inla_variables$logimg,
    weight = 1
  )

  # 6. Create mesh
  print("Create_mesh")
  inla_mesh <- create_inla_mesh(model_params$x, model_params$y)

  # 7. Define SPDE model
  print("Define_SPDE_model")
  spde_model <- define_spde_model(
    inla_mesh,
    nonstationary = FALSE,
    p_range = c(2, 0.2),
    p_sigma = c(2, 0.2)
  )

  # 8. Create projection matrix
  print("Create_projection_matrix")
  projection_matrix_A <- inla.spde.make.A(inla_mesh,
                                          loc = cbind(model_params$x, model_params$y))

  # 9. Prepare model stack
  print("Prepare_model_stack")
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

  # 10. Run INLA model
  print("INLA results")
  inla_result <- run_inla_model(
    stk = model_stack$stk,
    par = model_params$par,
    epar = model_params$epar,
    spde = spde_model,
    tolerance = 1e-4,
    restart = 0L,
    shape = 'none'
  )

  # 11. Project results
  print("Project results")
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

  # 12. Collect results
  print("INLA collect")
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

  # 13. Unscale results
  print("Unscale results")
  unscaled_results <- unscale_collected(inla_results_collected, scaling = FALSE)

  # 13.5 Plot the image
  # image(t(unscaled_results)[, nrow(unscaled_results):1], col = heat.colors(256), main = "Corrected Orientation")

  # 14. Save output
  print("Save output")
  output_dir <- "INLA_output_NPY"
  if (!dir.exists(output_dir)) dir.create(output_dir)
  # fname is npy_path without the npy extension
  fname <- sub("\\.npy$", "", basename(npy_path))
  out_path <- file.path(output_dir, fname)
  save_npy(unscaled_results, out_path)

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
