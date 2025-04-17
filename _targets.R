library(targets)
library(tarchetypes) # Load other packages as needed.

# Set target options:
tar_option_set(
  packages = c("INLA", "FITSio", "reshape2", "ggplot2", "viridis", "optparse", "rlang"),  # packages that your targets need to run
  )

tar_source()

# Replace the target list below with your own:
list(

  # 1. Load the .npy file
  # 1.1 read the path from the file variables/path.txt
  tar_target(
    file_path,
    "variants/cosmic.npy",
    format = "file"
  ),

  # 2. Read and process each .npy file using Python
  tar_target(
    raw_data,
    load_npy(file_path),
  ),

  # 2. Prepare Data
  tar_target(
    #prepared_data,
    inla_variables,
    prepare_data(raw_data, scaling = FALSE),
  ),

  tar_target(
    data_validity_check,
    check_data_validity(valid = inla_variables$valid,tx = inla_variables$x,
                        ty = inla_variables$y,logimg = inla_variables$logimg,
                        img = inla_variables$img),
    cue = tar_cue(mode = "always"), # Ensure it stops if data is invalid
  ),

  tar_target(
    model_params,
    compute_parameters(
      valid = inla_variables$valid,
      tx = inla_variables$x,
      ty = inla_variables$y,
      logimg = inla_variables$logimg,
      weight = 1, # Adjust weight as needed
      )
  ),

  tar_target(
    inla_mesh,
    create_inla_mesh(model_params$x, model_params$y) # Adjust cutoff as needed
  ),

  tar_target(
    spde_model,
    define_spde_model(
      inla_mesh,
      nonstationary = FALSE,  # Adjust based on your use case
      p_range = c(2, 0.2),
      p_sigma = c(2, 0.2)
    )
  ),

  tar_target(
    projection_matrix_A,
    inla.spde.make.A(inla_mesh, loc = cbind(model_params$x, model_params$y))
  ),

  tar_target(
    model_stack,
    prepare_model_stack(
      shape = 'none',  # Adjust shape based on your needs: 'radius', 'ellipse', 'none'
      x = model_params$x,
      y = model_params$y,
      par = model_params$par,
      A = projection_matrix_A,
      spde = spde_model,
      weight = 1, # Adjust weight as needed
      xcenter = model_params$xcenter,
      ycenter = model_params$ycenter
    )
  ),

  tar_target(
    inla_result,
    run_inla_model(
      stk = model_stack$stk,
      par = model_params$par,
      epar = model_params$epar,
      spde = spde_model,
      tolerance = 1e-4, # Adjust as necessary
      restart = 0L, # Adjust based on your restart strategy
      shape = 'none' # Adjust shape as needed
    )
  ),

  tar_target(
    projected_results,
    project_inla_results(
      mesh = inla_mesh,
      res = inla_result,
      xini = 0, xfin = inla_variables$xfin,
      yini = 0, yfin = inla_variables$yfin,
      xsize = inla_variables$xsize,
      ysize = inla_variables$ysize,
      spde = spde_model,
      zoom = 1, # Adjust zoom as needed
      shape = 'none', # Adjust shape as needed
      xcenter = model_params$xcenter,
      ycenter = model_params$ycenter,
      eigens = model_stack$eigens
    )
  ),

  tar_target(
    inla_results_collected,
    project_inla_results_collect(
      mesh = inla_mesh,
      res = inla_result,
      xini = 0, xfin = inla_variables$xfin,
      yini = 0, yfin = inla_variables$yfin,
      xsize = inla_variables$xsize,
      ysize = inla_variables$ysize,
      spde = spde_model,
      zoom = 1, # Adjust zoom as needed
      shape = 'none', # Adjust shape as needed
      xcenter = model_params$xcenter,
      ycenter = model_params$ycenter,
      eigens = model_stack$eigens
    )
  ),

  tar_target(
    unscalled_results,
    unscale_collected(inla_results_collected, scaling = FALSE),
  ),
  # 3. Plot the results with image()
  # tar_target(
  #   ploting,
  #   image(t(unscaled_results$out), col = heat.colors(256), main = "Corrected Orientation")
  #   
  # ),
  # 
  
  # 4. Save the result using the same file name
  tar_target(
    save_results,
    {
      fname <- basename("TESTTIIIING")  # extract just the filename
      out_path <- file.path("INLA_output_NPY", fname)
      dir.create("INLA_output_NPY", showWarnings = FALSE)
      save_npy(unscalled_results, out_path)
      out_path
    },
    format = "file"
  )
)
