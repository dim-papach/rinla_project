#!/usr/bin/env Rscript
# setup_inla.R - Script to install and set up INLA
#
# This script helps users set up the INLA package for R, which is required
# for the FYF tool to function properly.

# Function to check if a package is installed
is_package_installed <- function(package_name) {
  return(package_name %in% rownames(installed.packages()))
}

# Function to install INLA
install_inla <- function() {
  # Print status message
  cat("Checking for INLA package...\n")
  
  # Check if INLA is already installed
  if (is_package_installed("INLA")) {
    cat("INLA is already installed. Loading to check functionality...\n")
    
    # Try to load INLA to ensure it works
    tryCatch({
      library(INLA)
      cat("INLA loaded successfully.\n")
      return(TRUE)
    }, error = function(e) {
      cat("Error loading INLA:", conditionMessage(e), "\n")
      cat("Attempting to reinstall INLA...\n")
    })
  } else {
    cat("INLA is not installed. Installing now...\n")
  }
  
  # Install INLA from the official repository
  tryCatch({
    # Install missing dependencies
    if (!is_package_installed("Matrix")) {
      cat("Installing Matrix package...\n")
      install.packages("Matrix")
    }
    if (!is_package_installed("sp")) {
      cat("Installing sp package...\n")
      install.packages("sp")
    }
    if (!is_package_installed("foreach")) {
      cat("Installing foreach package...\n")
      install.packages("foreach")
    }
    
    # Install INLA
    cat("Installing INLA package...\n")
    install.packages("INLA", 
                     repos = c(INLA = "https://inla.r-inla-download.org/R/stable"), 
                     dependencies = TRUE)
    
    # Check if installation was successful
    if (is_package_installed("INLA")) {
      cat("INLA installed successfully.\n")
      
      # Try to load INLA to ensure it works
      library(INLA)
      cat("INLA loaded successfully.\n")
      return(TRUE)
    } else {
      cat("Failed to install INLA via the standard method.\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("Error installing INLA:", conditionMessage(e), "\n")
    return(FALSE)
  })
}

# Function to install other required packages
install_required_packages <- function() {
  required_packages <- c("reshape2", "rlang", "reticulate", "optparse")
  
  for (pkg in required_packages) {
    if (!is_package_installed(pkg)) {
      cat("Installing", pkg, "package...\n")
      install.packages(pkg, dependencies = TRUE)
    }
  }
}

# Main function
main <- function() {
  cat("===== FYF: Setting up R-INLA =====\n")
  
  # Install required packages
  cat("Installing required R packages...\n")
  install_required_packages()
  
  # Install INLA
  if (install_inla()) {
    cat("\n✅ INLA setup completed successfully!\n")
    cat("\nYou can now use the FYF tool with INLA processing.\n")
    return(TRUE)
  } else {
    cat("\n❌ Failed to set up INLA.\n")
    cat("\nPlease try to install INLA manually:\n")
    cat("1. Open R console\n")
    cat("2. Run: install.packages('INLA', repos=c(INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)\n")
    cat("3. Check if it works by running: library(INLA)\n")
    return(FALSE)
  }
}

# Execute main function
main()