let
# Importing Nixpkgs at a specific commit
 pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/ad7efee13e0d216bf29992311536fce1d3eefbef.tar.gz") {};

# List of R packages to be included
 rpkgs = builtins.attrValues {
  inherit (pkgs.rPackages) devtools lattice latticeExtra rmarkdown FITSio reshape2 classInt imager fields latex2exp viridis rasterVis IDPmisc reticulate yaml gtools data_table spatstat colorspace ggplot2 spam sp stringr rgl beepr codetools inlabru fmesher target crew here;
};

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
  inherit (pkgs) R glibcLocales nix gnugrep glibc python3;
};

# RStudio packages to be included
 rstudio_pkgs = pkgs.rstudioWrapper.override {
  packages = [ git_archive_pkgs rpkgs ];
};

# Python packages to be included
 python_pkgs = builtins.attrValues{
  inherit (pkgs.python311Packages) numpy pandas matplotlib astropy;
};
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
      
    QT_XCB_GL_INTEGRATION="none";
  }
