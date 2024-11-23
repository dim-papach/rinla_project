# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
# library(tarchetypes) # Load other packages as needed.

# Set target options:
tar_option_set(
  packages = c("INLA", "FITSio", "reshape2", "ggplot2", "viridis")  # packages that your targets need to run
  # format = "qs", # Optionally set the default storage format. qs is fast.
  #
  # For distributed computing in tar_make(), supply a {crew} controller
  # as discussed at https://books.ropensci.org/targets/crew.html.
  # Choose a controller that suits your needs. For example, the following
  # sets a controller with 2 workers which will run as local R processes:
  #
  #   controller = crew::crew_controller_local(workers = 2)
  #
  # Alternatively, if you want workers to run on a high-performance computing
  # cluster, select a controller from the {crew.cluster} package. The following
  # example is a controller for Sun Grid Engine (SGE).
  # 
  #   controller = crew.cluster::crew_controller_sge(
  #     workers = 50,
  #     # Many clusters install R as an environment module, and you can load it
  #     # with the script_lines argument. To select a specific verison of R,
  #     # you may need to include a version string, e.g. "module load R/4.3.0".
  #     # Check with your system administrator if you are unsure.
  #     script_lines = "module load R"
  #   )
  #
  # Set other options as needed.
)

# tar_make_clustermq() is an older (pre-{crew}) way to do distributed computing
# in {targets}, and its configuration for your machine is below.
options(clustermq.scheduler = "multicore")

# tar_make_future() is an older (pre-{crew}) way to do distributed computing
# in {targets}, and its configuration for your machine is below.
future::plan(future.callr::callr)

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
# source("other_functions.R") # Source other scripts as needed.

# Replace the target list below with your own:
list(
  # 1. Get Data
  tar_target(
    fits_file_path,
    "Ha_line_map_masked_cropped.fits", # Replace with the actual path to your FITS file
    format = "file"
  ),
  tar_target(
    header_data,
    get_header(fits_file_path)
  ),
  tar_target(
    raw_data,
    get_data(fits_file_path)
  ),
  
  # 2. Prepare Data
  tar_target(
    #prepared_data,
    inla_variables,
    prepare_data(raw_data)
  ),
  
  tar_target(
    data_validity_check,
    check_data_validity(valid = inla_variables$valid,tx = inla_variables$x,
                        ty = inla_variables$y,logimg = inla_variables$logimg,
                        img = inla_variables$img),
    cue = tar_cue(mode = "always") # Ensure it stops if data is invalid
  ),
  
  tar_target(
    model_params,
    compute_parameters(
      valid = inla_variables$valid,
      tx = inla_variables$x,
      ty = inla_variables$y,
      logimg = inla_variables$logimg,
      weight = 1 # Adjust weight as needed
    )
  ),
  
  tar_target(
    inla_mesh,
    create_inla_mesh(model_params$x, model_params$y, cutoff = 5) # Adjust cutoff as needed
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
      zoom = 1, # Adjust zoom as needed
      shape = 'none', # Adjust shape as needed
      xcenter = model_params$xcenter,
      ycenter = model_params$ycenter,
      eigens = model_stack$eigens
    )
  ),
  
  tar_target(
    inla_results_collected,
    collect_inla_results(
      output = projected_results$output,
      outputsd = projected_results$outputsd,
      x = model_params$x,
      y = model_params$y,
      par = model_params$par,
      epar = model_params$epar,
      xsize = inla_variables$xsize,
      ysize = inla_variables$ysize,
      xini = 0,
      xfin = inla_variables$xfin,
      yini = 0,
      yfin = inla_variables$yfin,
      zoom = 1 # Adjust zoom as needed
    )
  ),
  
  # 4. Save INLA Results as FITS Files
  tar_target(
    save_fits_files,
    save_fits(inla_results_collected,header_data = header_data,
              output_dir = "INLA_fits_output"),
    format = "file"
  ),
  
  # 5. Plot and Save INLA Analysis Results
  # tar_target(
  #   plot_inla_images,
  #   plot_and_save_images(prepared_data, inla_results_collected, outfile = "out_g", eroutfile = "error_g")
  # ),
  
  tar_target(
    plot_inla_results,
    plot_inla(inla_results_collected, title_prefix = "INLA_Result", output_dir = "plots"),
    cue = tar_cue(mode = "always")
  )
)
