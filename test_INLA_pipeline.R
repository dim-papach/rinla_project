# Load the testthat library
library(testthat)
library(INLA)
library(reshape2)
library(testthat)

source("R/functions.R")
# -----------------------------------------------------------------------------
# Test: load_path
# -----------------------------------------------------------------------------
# This test ensures that the `load_path` function handles missing files gracefully.
# It expects an error when attempting to load a non-existent file.
test_that("load_path handles missing file gracefully", {
    expect_error(load_path("nonexistent_file.txt"), "cannot open the connection")
})

# -----------------------------------------------------------------------------
# Test: prepare_data
# -----------------------------------------------------------------------------
# This test checks that the `prepare_data` function handles empty input correctly.
# It ensures that the function returns a list with no valid points when given an empty matrix.
test_that("prepare_data handles empty input", {
    empty_data <- matrix(numeric(0), nrow = 0, ncol = 0)
    expect_error(prepare_data(empty_data, scaling = FALSE), "No valid data points found in the image.")
})

# -----------------------------------------------------------------------------
# Test: check_data_validity
# -----------------------------------------------------------------------------
# This test verifies that the `check_data_validity` function fails when provided with invalid data.
# It expects an error when the input data is NULL or invalid.
test_that("check_data_validity fails on invalid data", {
    invalid_data <- list(valid = NULL, x = NULL, y = NULL, logimg = NULL, img = NULL)
    expect_error(
        check_data_validity(
            valid = invalid_data$valid,
            tx = invalid_data$x,
            ty = invalid_data$y,
            logimg = invalid_data$logimg,
            img = invalid_data$img
        ),
        "No valid data points found for INLA analysis."
    )
})
# -----------------------------------------------------------------------------
# Test: compute_parameters
# -----------------------------------------------------------------------------
# This test ensures that the `compute_parameters` function handles cases with no valid points.
# It verifies that the function returns empty x and y coordinates when no valid points are provided.
test_that("compute_parameters handles missing valid points", {
    mock_data <- matrix(runif(100), nrow = 10, ncol = 10)  # Example mock data
    prepared_data <- prepare_data(mock_data, scaling = FALSE)
    prepared_data$valid <- integer(0)  # No valid points
    params <- compute_parameters(
        valid = prepared_data$valid,
        tx = prepared_data$x,
        ty = prepared_data$y,
        logimg = prepared_data$logimg,
        weight = 1  # Use a valid weight
    )
    expect_equal(length(params$x), 0)
    expect_equal(length(params$y), 0)
    expect_true(is.list(params))
})
# -----------------------------------------------------------------------------
# Test: create_inla_mesh
# -----------------------------------------------------------------------------
# This test checks that the `create_inla_mesh` function fails when provided with insufficient points.
# It expects an error when the input x and y coordinates are empty.
test_that("create_inla_mesh fails with insufficient points", {
    expect_error(create_inla_mesh(numeric(0), numeric(0)), "Insufficient points to create a mesh.")
})
# -----------------------------------------------------------------------------
# Test: define_spde_model
# -----------------------------------------------------------------------------
# This test ensures that the `define_spde_model` function handles invalid mesh inputs.
# It expects an error when the mesh input is NULL.
test_that("define_spde_model handles invalid mesh", {
    expect_error(
        define_spde_model(NULL, nonstationary = FALSE, p_range = c(2, 0.2), p_sigma = c(2, 0.2)),
        "Unknown mesh class 'NULL'."
    )
})
# -----------------------------------------------------------------------------
# Test: run_inla_model
# -----------------------------------------------------------------------------
# This test verifies that the `run_inla_model` function handles invalid stack inputs.
# It expects an error when the stack input is NULL or invalid.
test_that("run_inla_model handles invalid stack", {
    expect_error(
        run_inla_model(
            stk = NULL,
            par = numeric(0),
            epar = NULL,
            spde = NULL,
            tolerance = 1e-4,
            restart = 0L,
            shape = 'none'
        ),
        "'stack' must inherit from class \"inla.data.stack\"."
    )
})
# -----------------------------------------------------------------------------
# Test: project_inla_results_collect
# -----------------------------------------------------------------------------
# This test ensures that the `project_inla_results_collect` function handles invalid inputs.
# It expects an error when the mesh or result inputs are NULL.
test_that("project_inla_results_collect handles invalid inputs", {
    expect_error(
        project_inla_results_collect(
            mesh = NULL,
            res = NULL,
            xini = 0,
            xfin = 10,
            yini = 0,
            yfin = 10,
            xsize = 10,
            ysize = 10,
            zoom = 1,
            shape = 'none',
            xcenter = 0,
            ycenter = 0,
            eigens = NULL
        ),
        "Error: Mesh and result inputs cannot be NULL."
    )
})
# -----------------------------------------------------------------------------
# Test: unscale_collected
# -----------------------------------------------------------------------------
# This test verifies that the `unscale_collected` function handles empty results correctly.
# It ensures that the function returns an empty list when provided with empty input.
test_that("unscale_collected handles empty results", {
    empty_results <- list(output = matrix(numeric(0), nrow = 0, ncol = 0))
    unscaled <- unscale_collected(empty_results, scaling = FALSE)
    expect_true(is.list(unscaled))
    expect_equal(length(unscaled$output), 0)})