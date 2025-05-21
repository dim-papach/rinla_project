# package.nix
{ lib
, python3
, R
, rPackages
, fetchzip
, fetchgit
}:

let
  # Define INLA the same way as in your default.nix
  inla = rPackages.buildRPackage {
    name = "INLA";
    version = "24.05.10";
    src = fetchzip {
      url = "https://inla.r-inla-download.org/R/testing/src/contrib/INLA_24.05.10.tar.gz";
      sha256 = "sha256-v9hvncV2oAI2JtqXQdU4LaqRQ6W/d6ydFrBrB/k7nqk="; 
    };
    propagatedBuildInputs = with rPackages; [
      Matrix
      sp
      foreach
      fmesher
    ];
  };

  # Define INLAutils the same way as in your default.nix
  inlautils = rPackages.buildRPackage {
    name = "INLAutils";
    src = fetchgit {
      url = "https://github.com/timcdlucas/INLAutils.git";
      branchName = "master";
      rev = "74d769a";
      sha256 = "sha256-0F171f3wuYm83uCcJKBKIrtu2SrSEI29PEe7iZeTYes=";
    };
    propagatedBuildInputs = with rPackages; [
      assertthat
      raster
      RColorBrewer
      sp
      reshape2
      tidyr
      cowplot
      ggplot2
      viridis
      optparse
    ];
  };

in python3.pkgs.buildPythonPackage rec {
  pname = "fyf";
  version = "0.1.0";
  
  src = ./.;
  
  format = "setuptools"; # Specify that we're using setuptools

  propagatedBuildInputs = with python3.pkgs; [
    numpy
    scipy
    astropy
    matplotlib
    colorama
    scikit-image
    scikit-learn
    click
  ];

  # Runtime dependencies including R and custom R packages
  runtimeDependencies = [
    R
    inla
    inlautils
  ] ++ (with rPackages; [
    class
    classInt
    colorspace
    DBI
    e1071
    fmesher
    KernSmooth
    lattice
    lifecycle
    magrittr
    Matrix
    MatrixModels
    munsell
    parallelly
    plyr
    proxy
    Rcpp
    RcppCNPy
    R6
    reshape2
    rlang
    scales
    sf
    sp
    splines2
    spatstat
    stringr 
    stringi 
    withr
    glue
    gtable
    scales
    ### Suggested packages
    compositions
    Deriv
    Ecdat
    Rgraphviz
    deldir
    devtools
    doParallel 
    dplyr 
    evd 
    fastGHQuad 
    fields 
    ggplot2 
    ggpubr
    gsl 
    graph 
    gridExtra 
    knitr 
    markdown 
    MASS 
    matrixStats
    mlogit 
    mvtnorm 
    numDeriv 
    pixmap 
    rgl 
    rmarkdown 
    runjags 
    sf
    shiny 
    sn  
    spdep 
    splancs 
    terra 
    tidyterra
    tibble
    testthat 
    units
    gtools 
    INLAspacetime
    vctrs
    pillar
    cli
    pkgconfig
    viridis
    viridisLite
    optparse
    getopt
    here
    rprojroot
    reticulate
    jsonlite
    png
  ]);
  
  # Make R packages available during runtime by wrapping the Python executable
  makeWrapperArgs = [
    "--prefix R_LIBS_SITE : ${lib.makeSearchPath "library" runtimeDependencies}"
    "--prefix PATH : ${lib.makeBinPath [ R ]}"
    "--set RETICULATE_PYTHON $(which python3)/bin/python3"
    "--set FYF_INSTALL_DIR $out "
    "--set FYF_DATA_DIR $out/share/fyf" 
    "--set FYF_R_SCRIPTS_DIR $out/lib/fyf/r" 
    "--set FYF_VARIANTS_DIR $out/share/fyf/variants" 
  ];
  
  # Don't run tests by default as they might require additional setup
  doCheck = false;
  
  meta = with lib; {
    description = "Fill Your FITS - Process astronomical FITS images using R-INLA";
    homepage = "https://github.com/dim-papach/rinla_project";
    license = licenses.mit;  # Adjust if your license is different
    platforms = platforms.all;
  };
}