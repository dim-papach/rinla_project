# Use rocker/geospatial as base image to save build time on spatial libraries (GDAL, GEOS, PROJ)
FROM rocker/geospatial:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    RETICULATE_PYTHON=/usr/bin/python3

# Install system dependencies
# Added libraries often needed for rgl, imager, and text shaping
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    cowsay \
    pandoc \
    libglpk-dev \
    libxt-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libfftw3-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    glibc-source \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --break-system-packages --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    astropy \
    scipy \
    scikit-image \
    scikit-learn \
    colorama \
    click \
    seaborn \
    jupyter \
    radian \
    setuptools \
    wheel

# Install R packages - Group 1: Utilities & Data Handling
RUN install2.r --error --skipinstalled -n -1 \
    devtools \
    remotes \
    beepr \
    classInt \
    codetools \
    colorspace \
    crew \
    data.table \
    fields \
    foreach \
    future \
    future.batchtools \
    future.callr \
    gtools \
    here \
    jsonlite \
    optparse \
    parallelly \
    spam \
    stringr \
    yaml

# Install R packages - Group 2: Visualization & Graphics
# Splitting this out helps identify if graphical libs (rgl, imager) fail
RUN install2.r --error --skipinstalled -n -1 \
    ggplot2 \
    lattice \
    latticeExtra \
    viridis \
    visNetwork \
    rasterVis \
    imager

# Install R packages - Group 3: specialized & remaining
# Note: rgl is often tricky in containers. If it fails, we might need to skip it or fix X11 deps.
RUN install2.r --error --skipinstalled -n -1 \
    FITSio \
    IDPmisc \
    jjb \
    languageserver \
    latex2exp \
    MASS \
    RcppCNPy \
    quarto \
    rgl \
    rlang \
    rmarkdown \
    reshape2 \
    reticulate \
    sp \
    spatstat \
    targets \
    testthat \
    tarchetypes

# Install INLA
# We use the testing repository as specified in your nix file
RUN Rscript -e 'install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)'

# Install inlabru AFTER INLA to ensure it uses the correct fmesher version
RUN install2.r --error inlabru

# Install INLAutils from GitHub
RUN Rscript -e 'remotes::install_github("timcdlucas/INLAutils")'

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the local fyf package in editable mode
RUN pip3 install --break-system-packages -e .

# Create directory for data mounting
RUN mkdir -p /data

# Default command
CMD ["bash"]
