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
  inherit (pkgs) R glibcLocales nix gnugrep glibc python311 toybox cowsay;
};
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
        version = "23.05.30-1";
        src = pkgs.fetchzip{
	        url = "https://inla.r-inla-download.org/R/testing/src/contrib/INLA_23.05.30-1.tar.gz";
	        sha256 = "sha256-LvZRUidUuWzBcgarqnC8i+fWnyJbTtjWGj7yOdFVNDg=";
	        };
}) ];


  # Python 3.11 with --enable-shared for reticulate
  python311 = pkgs.python311.overrideAttrs (oldAttrs: {
    enableShared = true;
  });



  # Python packages to be included
  python_pkgs = python311.withPackages (ps: with ps; [
    numpy pandas matplotlib astropy radian scipy colorama
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

    buildInputs = [ git_archive_pkgs rpkgs  system_packages rstudio_pkgs python_pkgs inla];
      
  shellHook = ''
    export RETICULATE_PYTHON=$(which python3)
    echo "Setting up Nix shell environment..."
    echo "LOCALE_ARCHIVE is set to: $LOCALE_ARCHIVE"
    echo "LANG is set to: $LANG"
    echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
    # The rest of your existing shellHook

    # Safely set LD_LIBRARY_PATH to include the R library path
    if [ -z "$LD_LIBRARY_PATH" ]; then
      export LD_LIBRARY_PATH=/nix/store/130rh1iwc2k0qqksgf09663sdbds5vml-R-4.3.2/lib/R/lib
    else
      export LD_LIBRARY_PATH=/nix/store/130rh1iwc2k0qqksgf09663sdbds5vml-R-4.3.2/lib/R/lib:$LD_LIBRARY_PATH
    fi

    Rscript -e 'if (requireNamespace("INLA", quietly = TRUE))  { system("cowsay -f llama INLA is already installed.\n") } else {system("cowsay -e xx -f ghostbusters INLA is NOT installed RIP")}' 
    chmod +x ./run.R
  '';

    QT_XCB_GL_INTEGRATION="none";
  }
