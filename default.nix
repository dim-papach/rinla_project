let
# Importing Nixpkgs at a specific commit
 pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/cca4f8e59e9479ced4f02f33530be367220d5826.tar.gz") {};

# List of R packages to be included
 rpkgs = builtins.attrValues {
  inherit (pkgs.rPackages) 
    beepr 
    classInt 
    codetools 
    colorspace 
    crew 
    data_table 
    devtools 
    fields 
    foreach
    FITSio 
    future
    future_batchtools
    future_callr
    fmesher 
    ggplot2 
    gtools 
    here 
    httpgd
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
    optparse
    parallelly
    RcppCNPy
    quarto
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
    testthat
    tarchetypes
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
      inherit (pkgs.rPackages) assertthat raster RColorBrewer sp reshape2 tidyr cowplot ggplot2 viridis optparse;
    };
  }) ];

 
# System packages to be included (Downloaded from the nixpkgs repo)
  system_packages = builtins.attrValues {
  inherit (pkgs) R glibcLocales nix gnugrep glibc python311 jupyter toybox cowsay pandoc openblas;};
# Rebuild R to ensure it uses the specified GLIBC version
R = pkgs.R.overrideAttrs (oldAttrs: {
  buildInputs = oldAttrs.buildInputs ++ [ pkgs.glibc ];
});
# RStudio packages to be included
 rstudio_pkgs = pkgs.rstudioWrapper.override {
  packages = [ git_archive_pkgs rpkgs ];
};

  # Definition for INLA package
    inla = [(pkgs.rPackages.buildRPackage {
        name = "INLA";
        version = "24.12.11";
        src = pkgs.fetchzip{
	        url = "https://inla.r-inla-download.org/R/testing/src/contrib/INLA_24.05.10.tar.gz";
	        sha256 = "sha256-v9hvncV2oAI2JtqXQdU4LaqRQ6W/d6ydFrBrB/k7nqk=";
	        };
}) ];


  # Python 3.11 with --enable-shared for reticulate
  python311 = pkgs.python311.overrideAttrs (oldAttrs: {
    enableShared = true;
  });



  # Python packages to be included
  python_pkgs = python311.withPackages (ps: with ps; [
    numpy pandas matplotlib astropy radian scipy scikit-image scikit-learn colorama click pip
  ]);

/* haskellPkgs = builtins.attrValues {
  inherit (pkgs.haskellPackages) 
  typst-symbols_0_1_7;
};
 */
 in
  pkgs.mkShell {
    LOCALE_ARCHIVE = if pkgs.system == "x86_64-linux" then  "${pkgs.glibcLocales}/lib/locale/locale-archive" else "";
    LANG = "en_US.UTF-8";

    buildInputs = [ git_archive_pkgs rpkgs  system_packages rstudio_pkgs python_pkgs inla];
      
  shellHook = ''
    export RETICULATE_PYTHON=$(which python3)
    echo "Setting up Nix shell environment..."
    echo "LOCALE_ARCHIVE is set to: $LOCALE_ARCHIVE"
    echo "LANG is set to: $LANG"
    echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
    # The rest of your existing shellHook

    Rscript -e 'if (requireNamespace("INLA", quietly = TRUE))  { system("cowsay -f llama INLA is already installed.\n") } else {system("cowsay -e xx -f ghostbusters INLA is NOT installed RIP")}' 
    chmod +x ./run.R
  '';

    QT_XCB_GL_INTEGRATION="none";
  }
