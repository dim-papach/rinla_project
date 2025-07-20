# CMakeLists.txt for FYF (Fill Your FITS) - Simple Version
# Minimal build system that avoids hanging issues

cmake_minimum_required(VERSION 3.16)
project(FYF 
    VERSION 0.1.0 
    DESCRIPTION "Fill Your FITS - Process astronomical FITS images using R-INLA"
)

# =============================================================================
# STEP 1: Basic System Detection
# =============================================================================

message(STATUS "=== FYF Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "Source: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build: ${CMAKE_BINARY_DIR}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")

# Detect Linux distribution (safer version)
set(LINUX_DISTRO "unknown")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/etc/os-release")
    file(STRINGS "/etc/os-release" OS_RELEASE_LINES)
    foreach(line ${OS_RELEASE_LINES})
        if(line MATCHES "^ID=(.*)$")
            set(LINUX_DISTRO ${CMAKE_MATCH_1})
            string(REPLACE "\"" "" LINUX_DISTRO ${LINUX_DISTRO})
            break()
        endif()
    endforeach()
endif()
message(STATUS "Distribution: ${LINUX_DISTRO}")

# =============================================================================
# STEP 2: Find Required Programs (Simplified)
# =============================================================================

message(STATUS "")
message(STATUS "=== Finding Required Programs ===")

# Find Python3
find_program(PYTHON_EXECUTABLE python3)
if(PYTHON_EXECUTABLE)
    message(STATUS "Found Python3: ${PYTHON_EXECUTABLE}")
    
    # Get Python version (simple method)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} --version
        OUTPUT_VARIABLE PYTHON_VERSION_OUTPUT
        ERROR_VARIABLE PYTHON_VERSION_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 5
    )
    
    if(PYTHON_VERSION_OUTPUT)
        message(STATUS "Python: ${PYTHON_VERSION_OUTPUT}")
    elseif(PYTHON_VERSION_ERROR)
        message(STATUS "Python: ${PYTHON_VERSION_ERROR}")
    endif()
else()
    message(FATAL_ERROR "Python3 not found!")
endif()

# Find R (simplified)
find_program(R_EXECUTABLE R)
find_program(RSCRIPT_EXECUTABLE Rscript)

if(R_EXECUTABLE)
    message(STATUS "Found R: ${R_EXECUTABLE}")
else()
    message(FATAL_ERROR "R not found!")
endif()

if(RSCRIPT_EXECUTABLE)
    message(STATUS "Found Rscript: ${RSCRIPT_EXECUTABLE}")
else()
    message(FATAL_ERROR "Rscript not found!")
endif()

# Find pip
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m pip --version
    RESULT_VARIABLE PIP_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
    TIMEOUT 5
)

if(PIP_RESULT EQUAL 0)
    message(STATUS "Found pip: python3 -m pip")
    set(PIP_COMMAND "${PYTHON_EXECUTABLE} -m pip")
else()
    message(WARNING "pip not found")
    set(PIP_COMMAND "${PYTHON_EXECUTABLE} -m pip")  # Try anyway
endif()

# =============================================================================
# STEP 3: Project Structure Check
# =============================================================================

message(STATUS "")
message(STATUS "=== Project Structure ===")

# Check setup.py
if(EXISTS "${CMAKE_SOURCE_DIR}/setup.py")
    message(STATUS "setup.py: ✓")
else()
    message(FATAL_ERROR "setup.py not found!")
endif()

# Check fyf directory
if(EXISTS "${CMAKE_SOURCE_DIR}/fyf")
    message(STATUS "fyf/ directory: ✓")
else()
    message(FATAL_ERROR "fyf/ directory not found!")
endif()

# Check fyf/r directory
if(EXISTS "${CMAKE_SOURCE_DIR}/fyf/r")
    message(STATUS "fyf/r/ directory: ✓")
else()
    message(WARNING "fyf/r/ directory not found")
endif()

# =============================================================================
# STEP 4: Installation Configuration
# =============================================================================

# Set default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install prefix" FORCE)
endif()

message(STATUS "")
message(STATUS "=== Installation Paths ===")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# =============================================================================
# STEP 5: Simple Targets
# =============================================================================

# Status target
add_custom_target(status
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== FYF Status ==="
    COMMAND ${CMAKE_COMMAND} -E echo "System: ${CMAKE_SYSTEM_NAME} - ${LINUX_DISTRO}"
    COMMAND ${CMAKE_COMMAND} -E echo "Python: ${PYTHON_EXECUTABLE}"
    COMMAND ${CMAKE_COMMAND} -E echo "R: ${R_EXECUTABLE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Install to: ${CMAKE_INSTALL_PREFIX}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Available targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  status              - Show this status"
    COMMAND ${CMAKE_COMMAND} -E echo "  check-python        - Test Python setup"
    COMMAND ${CMAKE_COMMAND} -E echo "  check-r             - Test R setup"
    COMMAND ${CMAKE_COMMAND} -E echo "  check-r-version     - Check R version and INLA compatibility"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Installation targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  install-system-deps - Install system dependencies"
    COMMAND ${CMAKE_COMMAND} -E echo "  install-r-deps      - Install R packages"
    COMMAND ${CMAKE_COMMAND} -E echo "  install-inla        - Install R-INLA"
    COMMAND ${CMAKE_COMMAND} -E echo "  install-fyf         - Install FYF package"
    COMMAND ${CMAKE_COMMAND} -E echo "  install-all         - Install everything"
    COMMAND ${CMAKE_COMMAND} -E echo "  verify              - Verify installation"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "Showing FYF status"
)

# Generate Python test script
set(PYTHON_TEST_SCRIPT "${CMAKE_BINARY_DIR}/check_python.py")
file(WRITE ${PYTHON_TEST_SCRIPT}
"#!/usr/bin/env python3
import sys
print('=== Python Environment Check ===')
print('Python version:', sys.version_info.major, '.', sys.version_info.minor, '.', sys.version_info.micro)
print('Python executable:', sys.executable)
print('Platform:', sys.platform)

# Test pip
import subprocess
try:
    result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print('pip:', result.stdout.strip())
    else:
        print('pip failed:', result.stderr.strip())
except Exception as e:
    print('pip test failed:', str(e))

print('Python check complete!')
")

# Test Python using the script
add_custom_target(check-python
    COMMAND ${CMAKE_COMMAND} -E echo "Testing Python setup..."
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_TEST_SCRIPT}
    COMMENT "Testing Python environment"
)

# Generate R test script
set(R_TEST_SCRIPT "${CMAKE_BINARY_DIR}/check_r.R")
file(WRITE ${R_TEST_SCRIPT}
"#!/usr/bin/env Rscript
cat('=== R Environment Check ===', '\n')
cat('R version:', R.version.string, '\n')
cat('R executable:', R.home(), '\n')
cat('Platform:', R.version\$platform, '\n')

# Test basic functionality
cat('Testing basic R functionality...', '\n')
x <- 1:10
y <- x^2
if (length(y) == 10) {
    cat('Success: Basic R operations work', '\n')
} else {
    cat('Error: Basic R operations failed', '\n')
}

cat('R check complete!', '\n')
")

# Test R using the script
add_custom_target(check-r
    COMMAND ${CMAKE_COMMAND} -E echo "Testing R setup..."
    COMMAND ${RSCRIPT_EXECUTABLE} ${R_TEST_SCRIPT}
    COMMENT "Testing R environment"
)

# Generate R version check script
set(R_VERSION_SCRIPT "${CMAKE_BINARY_DIR}/check_r_version.R")
file(WRITE ${R_VERSION_SCRIPT}
"#!/usr/bin/env Rscript
cat('=== R Version Check ===', '\n')
cat('R version string:', R.version.string, '\n')
cat('R major version:', R.version\$major, '\n')
cat('R minor version:', R.version\$minor, '\n')

# Get numeric version for comparison
r_version <- numeric_version(paste(R.version\$major, R.version\$minor, sep='.'))
cat('Numeric version:', as.character(r_version), '\n')

# Check INLA compatibility
cat('\n=== INLA Compatibility Check ===', '\n')
if (r_version >= '4.0.0') {
    cat('✓ R version is compatible with INLA', '\n')
    cat('Recommended INLA version: 22.04.16 or later', '\n')
} else if (r_version >= '3.5.0') {
    cat('⚠ R version is older but should work with INLA', '\n')
    cat('Recommended INLA version: 21.x.x series', '\n')
} else {
    cat('✗ R version may be too old for modern INLA', '\n')
    cat('Consider upgrading R to 4.0 or later', '\n')
}

# Check available INLA versions
cat('\n=== Available INLA Versions ===', '\n')
cat('Checking INLA repository...', '\n')
tryCatch({
    # Try to get available versions from the repository
    available_versions <- available.packages(repos = 'https://inla.r-inla-download.org/R/testing')
    if ('INLA' %in% rownames(available_versions)) {
        inla_version <- available_versions['INLA', 'Version']
        cat('Latest INLA in testing repo:', inla_version, '\n')
    } else {
        cat('Could not fetch INLA version info', '\n')
    }
}, error = function(e) {
    cat('Error checking INLA repository:', conditionMessage(e), '\n')
})

cat('\n=== Platform Information ===', '\n')
cat('Platform:', R.version\$platform, '\n')
cat('OS:', R.version\$os, '\n')
cat('Architecture:', R.version\$arch, '\n')
")

# Check R version and INLA compatibility
add_custom_target(check-r-version
    COMMAND ${CMAKE_COMMAND} -E echo "Checking R version and INLA compatibility..."
    COMMAND ${RSCRIPT_EXECUTABLE} ${R_VERSION_SCRIPT}
    COMMENT "Checking R version and INLA compatibility"
)

# Install system dependencies for Pop!_OS (Ubuntu-based)
add_custom_target(install-system-deps
    COMMAND ${CMAKE_COMMAND} -E echo "Installing system dependencies for ${LINUX_DISTRO}..."
    COMMAND sudo apt update
    COMMAND sudo apt install -y build-essential cmake pkg-config 
    COMMAND sudo apt install -y python3-dev python3-pip r-base-dev
    COMMAND sudo apt install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite3-dev
    COMMAND sudo apt install -y libssl-dev libxml2-dev libcurl4-openssl-dev zlib1g-dev 
    COMMAND sudo apt install -y libfontconfig1-dev libharfbuzz-dev libfribidi-dev libabsl-dev
    COMMAND ${CMAKE_COMMAND} -E echo "System dependencies installation complete!"
    COMMENT "Installing system dependencies"
)



# Generate R dependencies installation script
set(R_DEPS_SCRIPT "${CMAKE_BINARY_DIR}/install_r_deps.R")
file(WRITE ${R_DEPS_SCRIPT}
"#!/usr/bin/env Rscript
cat('=== Installing R Dependencies ===', '\n')

# Set up repositories
repos <- c(CRAN = 'https://cloud.r-project.org/')
options(repos = repos)

# Function to install package if not present
install_if_missing <- function(pkg) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat('Installing', pkg, '...', '\n')
        tryCatch({
            install.packages(pkg, dependencies = TRUE)
            cat('Success: Installed', pkg, '\n')
            return(TRUE)
        }, error = function(e) {
            cat('Error installing', pkg, ':', conditionMessage(e), '\n')
            return(FALSE)
        })
    } else {
        cat('Already installed:', pkg, '\n')
        return(TRUE)
    }
}

# Core packages needed for INLA and spatial analysis
packages <- c(
    #'systemfonts', # Required for textshaping
    'sf',          # Simple features for R
    'optparse',    # Command line parsing  
    'reticulate',  # Python interface
    'remotes',     # Package installation
    #'devtools',    # Development tools
    'dplyr',       # Data manipulation
    'ggplot2',     # Plotting
    'viridis',      # Color scales
    'foreach',     # Parallel computing
    'sp',          # Spatial data classes
    'Matrix'      # Required for INLA
)

# Install packages
success_count <- 0
for (pkg in packages) {
    if (install_if_missing(pkg)) {
        success_count <- success_count + 1
    }
}

cat('\n')
cat('Installation summary:', success_count, '/', length(packages), 'packages', '\n')

if (success_count == length(packages)) {
    cat('Success: All R dependencies installed!', '\n')
} else {
    cat('Warning: Some R packages failed to install', '\n')
}

cat('R dependencies installation complete!', '\n')
")

# Install R dependencies
add_custom_target(install-r-deps
    COMMAND ${CMAKE_COMMAND} -E echo "Installing R dependencies..."
    COMMAND ${RSCRIPT_EXECUTABLE} ${R_DEPS_SCRIPT}
    DEPENDS install-system-deps
    COMMENT "Installing R dependencies"
)

# Generate R-INLA installation script (your specific version)
set(INLA_SCRIPT "${CMAKE_BINARY_DIR}/install_inla.R")
file(WRITE ${INLA_SCRIPT}
"#!/usr/bin/env Rscript
cat('=== Installing R-INLA ===', '\n')

# Check if INLA is already installed and working
if (require('INLA', quietly = TRUE)) {
    cat('INLA already installed, testing functionality...', '\n')
    
    # Test basic INLA functionality
    tryCatch({
        # Simple test data
        n <- 50
        x <- runif(n)
        y <- 2 + 3*x + rnorm(n, 0, 0.1)
        
        # Simple INLA model
        result <- inla(y ~ x, data = data.frame(x=x, y=y), family='gaussian')
        
        if (!is.null(result) && !is.null(result\$summary.fixed)) {
            cat('Success: INLA is working correctly!', '\n')
            quit(status = 0)
        } else {
            cat('Warning: INLA installed but not working correctly, reinstalling...', '\n')
        }
    }, error = function(e) {
        cat('Warning: INLA test failed, reinstalling...', '\n')
    })
}

cat('Installing INLA from testing repository...', '\n')

# Set up repositories (your original configuration)
options(repos = c(
    inlabruorg = 'https://inlabru-org.r-universe.dev',
    INLA = 'https://inla.r-inla-download.org/R/testing',
    CRAN = 'https://cran.rstudio.com'
))

# Install fmesher first
cat('Installing fmesher...', '\n')
tryCatch({
    install.packages('fmesher')
    cat('Success: fmesher installed', '\n')
}, error = function(e) {
    cat('Error installing fmesher:', conditionMessage(e), '\n')
})

# Determine correct INLA version for R version
r_version <- numeric_version(paste(R.version$major, R.version$minor, sep='.'))

if (r_version >= '4.3.0') {
    inla_version <- '23.04.24'
    repo_type <- 'stable'
} else if (r_version >= '4.1.0') {
    inla_version <- '22.04.16'
    repo_type <- 'testing'
} else if (r_version >= '4.0.0') {
    inla_version <- '21.02.23'
    repo_type <- 'stable'
} else if (r_version >= '3.6.0') {
    inla_version <- '20.03.17'
    repo_type <- 'stable'
} else {
    cat('Error: R version', as.character(r_version), 'not supported', '\n')
    quit(status = 1)
}

cat('Installing INLA version', inla_version, 'for R', as.character(r_version), '\n')
tryCatch({
    remotes::install_version('INLA', 
                           version = inla_version,
                           repos = c(getOption('repos'), 
                                   INLA = paste0('https://inla.r-inla-download.org/R/', repo_type)), 
                           dep = TRUE)
    
    # Test the installation
    library(INLA)
    
    # Functionality test
    n <- 30
    x <- runif(n)
    y <- 1 + 2*x + rnorm(n, 0, 0.1)
    result <- inla(y ~ x, data = data.frame(x=x, y=y), family='gaussian')
    
    if (!is.null(result) && !is.null(result\$summary.fixed)) {
        cat('Success: INLA 22.04.16 installed and working!', '\n')
    } else {
        cat('Error: INLA installed but functionality test failed', '\n')
    }
    
}, error = function(e) {
    cat('Error installing INLA:', conditionMessage(e), '\n')
})

cat('INLA installation process complete!', '\n')
")

# Install R-INLA
add_custom_target(install-inla
    COMMAND ${CMAKE_COMMAND} -E echo "Installing R-INLA..."
    COMMAND ${RSCRIPT_EXECUTABLE} ${INLA_SCRIPT}
    DEPENDS install-r-deps
    COMMENT "Installing R-INLA package"
)

# Install FYF package in development mode
add_custom_target(install-fyf
    COMMAND ${CMAKE_COMMAND} -E echo "Installing FYF package in development mode..."
    COMMAND pipx install ${CMAKE_SOURCE_DIR} --force
    COMMAND ${CMAKE_COMMAND} -E echo "FYF package installed successfully!"
    DEPENDS install-inla
    COMMENT "Installing FYF package"
)

# Complete installation target
add_custom_target(install-all
    DEPENDS install-fyf
    COMMENT "Complete FYF installation"
)

# Verification target
add_custom_target(verify
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying FYF installation..."
    COMMAND ${PYTHON_EXECUTABLE} -c "import fyf; print('✓ FYF Python package imported successfully')"
    COMMAND ${RSCRIPT_EXECUTABLE} -e "library(INLA); cat('✓ R-INLA loaded successfully', '\n')"
    COMMAND ${CMAKE_COMMAND} -E echo "✓ FYF installation verified!"
    DEPENDS install-fyf
    COMMENT "Verifying FYF installation"
)

# =============================================================================
# STEP 6: Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Setup Complete ===")
message(STATUS "Basic CMake configuration successful!")
message(STATUS "")
message(STATUS "Try these commands:")
message(STATUS "  make status         - Show system status")
message(STATUS "  make check-python   - Test Python")
message(STATUS "  make check-r        - Test R")
message(STATUS "")