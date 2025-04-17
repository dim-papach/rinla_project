# R/test_functions.R
library(testthat)
library(reshape2)
library(rlang)
library(INLA)
library(reticulate)

source("functions.R")
setwd("/home/dp/Documents/GitHub/rinla_project/")
# -----------------------------------------------------------------------------
# Test: load_path
# -----------------------------------------------------------------------------
test_that("load_path handles valid and invalid file paths", {
    temp_file <- tempfile()
    writeLines("path/to/file.npy", temp_file)
    expect_equal(load_path(temp_file), "path/to/file.npy")
    unlink(temp_file)  # Clean up the temporary file
    expect_error(load_path(tempfile()), "cannot open the connection")
})

# -----------------------------------------------------------------------------
# Test: load_npy
# -----------------------------------------------------------------------------
test_that("load_npy loads data correctly", {
    np <- import("numpy")
    temp_file <- tempfile(fileext = ".npy")
    np$save(temp_file, np$array(matrix(1:4, 2, 2)))  # Save as numpy array
    data <- load_npy(temp_file)
    expect_equal(as.numeric(data), as.numeric(matrix(1:4, 2, 2)))  # Ensure numeric comparison
})

# -----------------------------------------------------------------------------
# Test: prepare_data
# -----------------------------------------------------------------------------
test_that("prepare_data handles valid and invalid inputs", {
    img <- matrix(1:9, nrow = 3, ncol = 3)
    prepared <- prepare_data(img)
    expect_equal(dim(prepared$logimg), c(3, 3))
    expect_true(length(prepared$valid) > 0)
    expect_error(prepare_data(NULL), "The image data is not a 2D matrix.")
    expect_error(prepare_data(matrix(0, 3, 3)), "No valid data points found in the image.")
})

# -----------------------------------------------------------------------------
# Test: check_data_validity
# -----------------------------------------------------------------------------
test_that("check_data_validity validates data correctly", {
    img <- matrix(1:9, nrow = 3, ncol = 3)
    prepared <- prepare_data(img)
    expect_error(check_data_validity(NULL, prepared$x, prepared$y, prepared$logimg, img),
                             "No valid data points found for INLA analysis.")
})

# -----------------------------------------------------------------------------
# Test: compute_parameters
# -----------------------------------------------------------------------------
test_that("compute_parameters computes parameters correctly", {
    img <- matrix(1:9, nrow = 3, ncol = 3)
    prepared <- prepare_data(img)
    params <- compute_parameters(prepared$valid, prepared$x, prepared$y, prepared$logimg, weight = 1)
    expect_true(is.list(params))
    expect_equal(length(params$x), length(prepared$valid))
})

# -----------------------------------------------------------------------------
# Test: create_inla_mesh
# -----------------------------------------------------------------------------
test_that("create_inla_mesh handles valid and invalid inputs", {
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    mesh <- create_inla_mesh(x, y)
    expect_true(inherits(mesh, "inla.mesh"))
    expect_error(create_inla_mesh(numeric(0), numeric(0)), "Insufficient points to create a mesh.")
})

# -----------------------------------------------------------------------------
# Test: define_spde_model
# -----------------------------------------------------------------------------
test_that("define_spde_model handles stationary and non-stationary models", {
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    mesh <- create_inla_mesh(x, y)
    spde_stationary <- define_spde_model(mesh, nonstationary = FALSE, p_range = c(2, 0.2), p_sigma = c(2, 0.2))
    expect_true(inherits(spde_stationary, "inla.spde"))
    spde_nonstationary <- define_spde_model(mesh, nonstationary = TRUE, p_range = c(2, 0.2), p_sigma = c(2, 0.2))
    expect_true(inherits(spde_nonstationary, "inla.spde"))
})

# -----------------------------------------------------------------------------
# Test: prepare_model_stack
# -----------------------------------------------------------------------------
test_that("prepare_model_stack creates stack correctly", {
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    par <- c(1, 2, 3)
    mesh <- create_inla_mesh(x, y)
    spde <- define_spde_model(mesh, nonstationary = FALSE, p_range = c(2, 0.2), p_sigma = c(2, 0.2))
    A <- inla.spde.make.A(mesh, loc = cbind(x, y))
    stack <- prepare_model_stack("none", x, y, par, A, spde, weight = 1, xcenter = 2, ycenter = 2)
    expect_true(is.list(stack))
    expect_true(inherits(stack$stk, "inla.data.stack"))
})

# -----------------------------------------------------------------------------
# Test: run_inla_model
# -----------------------------------------------------------------------------
test_that("run_inla_model runs correctly with valid inputs", {
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    par <- c(1, 2, 3)
    mesh <- create_inla_mesh(x, y)
    spde <- define_spde_model(mesh, nonstationary = FALSE, p_range = c(2, 0.2), p_sigma = c(2, 0.2))
    A <- inla.spde.make.A(mesh, loc = cbind(x, y))
    stack <- prepare_model_stack("none", x, y, par, A, spde, weight = 1, xcenter = 2, ycenter = 2)
    res <- run_inla_model(stack$stk, par, epar = NULL, spde, tolerance = 1e-4, restart = 0L, shape = "none")
    expect_true(inherits(res, "inla"))
})

# -----------------------------------------------------------------------------
# Test: project_inla_results_collect
# -----------------------------------------------------------------------------
test_that("project_inla_results_collect handles valid and invalid inputs", {
    x <- c(1, 2, 3)
    y <- c(1, 2, 3)
    par <- c(1, 2, 3)
    mesh <- create_inla_mesh(x, y)
    spde <- define_spde_model(mesh, nonstationary = FALSE, p_range = c(2, 0.2), p_sigma = c(2, 0.2))
    A <- inla.spde.make.A(mesh, loc = cbind(x, y))
    stack <- prepare_model_stack("none", x, y, par, A, spde, weight = 1, xcenter = 2, ycenter = 2)
    res <- run_inla_model(stack$stk, par, epar = NULL, spde, tolerance = 1e-4, restart = 0L, shape = "none")
    projections <- project_inla_results_collect(mesh, res, 0, 10, 0, 10, 10, 10, 1, "none", 2, 2, NULL, spde)
    expect_true(is.list(projections))
    expect_true(!is.null(projections$out))
    expect_error(project_inla_results_collect(NULL, NULL, 0, 10, 0, 10, 10, 10, 1, "none", 2, 2, NULL, spde),
                             "Error: Mesh and result inputs cannot be NULL.")
})

# -----------------------------------------------------------------------------
# Test: save_npy
# -----------------------------------------------------------------------------
test_that("save_npy saves arrays correctly", {
    np <- import("numpy")
    temp_dir <- tempdir()
    arrays <- list(a = matrix(1:4, 2, 2), b = matrix(5:8, 2, 2))
    save_npy(arrays, temp_dir)
    expect_true(file.exists(file.path(temp_dir, "a.npy")))
    expect_true(file.exists(file.path(temp_dir, "b.npy")))
    loaded_a <- np$load(file.path(temp_dir, "a.npy"))
    expect_equal(loaded_a, matrix(1:4, 2, 2))
})