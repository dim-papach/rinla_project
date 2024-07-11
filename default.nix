let
# Importing Nixpkgs at a specific commit
 pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/eb090f7b923b1226e8beb954ce7c8da99030f4a8.tar.gz") {};

# List of R packages to be included
 rpkgs = builtins.attrValues {
  inherit (pkgs.rPackages) 
    autokeras
    beepr 
    classInt 
    codetools 
    colorspace 
    crew 
    data_table 
    devtools 
    fields 
    FITSio 
    future
    future_batchtools
    future_callr
    fmesher 
    ggplot2 
    gtools 
    here 
    IDPmisc 
    imager 
    inlabru 
    jjb
    jsonlite 
    languageserver 
    lattice 
    latticeExtra 
    latex2exp 
    MASS
    rgl 
    rlang 
    rmarkdown 
    rasterVis 
    reshape2 
    reticulate 
    spam 
    sp 
    spatstat 
    stringr 
    targets 
    viridis 
    visNetwork
    yaml;};

# Git archive packages to be fetched
 git_archive_pkgs = [(pkgs.rPackages.buildRPackage {
    name = "INLAutils";
    src = pkgs.fetchgit {
      url = "https://github.com/timcdlucas/INLAutils.git";
      branchName = "master";
      rev = "74d769a";
      sha256 = "sha256-0F171f3wuYm83uCcJKBKIrtu2SrSEI29PEe7iZeTYes=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) assertthat raster RColorBrewer sp reshape2 tidyr cowplot ggplot2;
    };
  }) ];

 
# System packages to be included (Downloaded from the nixpkgs repo)
  system_packages = builtins.attrValues {
  inherit (pkgs) R glibcLocales nix gnugrep glibc python3 toybox;
};

# RStudio packages to be included
 rstudio_pkgs = pkgs.rstudioWrapper.override {
  packages = [ git_archive_pkgs rpkgs ];
};

  # Python 3.11 with --enable-shared for reticulate
  python311 = pkgs.python311.overrideAttrs (oldAttrs: {
    enableShared = true;
  });



  # Python packages to be included
  python_pkgs = python311.withPackages (ps: with ps; [
    numpy pandas matplotlib astropy radian
  ]);


 in
  pkgs.mkShell {
    LOCALE_ARCHIVE = if pkgs.system == "x86_64-linux" then  "${pkgs.glibcLocales}/lib/locale/locale-archive" else "";
    LANG = "en_US.UTF-8";
    LC_ALL = "en_US.UTF-8";
    LC_TIME = "en_US.UTF-8";
    LC_MONETARY = "en_US.UTF-8";
    LC_PAPER = "en_US.UTF-8";
    LC_MEASUREMENT = "en_US.UTF-8";

    buildInputs = [ git_archive_pkgs rpkgs  system_packages rstudio_pkgs python_pkgs];
      
  shellHook = ''
    export RETICULATE_PYTHON=$(which python3)
    #export LD_LIBRARY_PATH=${pkgs.glibc}/lib:$LD_LIBRARY_PATH
    Rscript -e 'if (!requireNamespace("INLA", quietly = TRUE)) { if (!requireNamespace("remotes", quietly = TRUE)) { install.packages("remotes") }; remotes::install_version("INLA", version = "23.05.30", repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/testing")) } else { cat("INLA is already installed.\n") }'

  '';

    QT_XCB_GL_INTEGRATION="none";
  }
