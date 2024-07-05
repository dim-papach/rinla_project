This project is based on the work of [Spatial field reconstruction with INLA: application to IFU galaxy data](https://academic.oup.com/mnras/article/482/3/3880/5144230) and [Spatial field reconstruction with INLA: Application to simulated galaxies](https://www.aanda.org/10.1051/0004-6361/202244481).

It uses the R-INLA package to fill missing values in FITS images of astrophysical objects. Our main focus is to test the accuracy and efficiency of this method.

To make sure you don&rsquo;t have a dependency problem, follow the instructions

# Quick installation

- Make sure you have `Nix` installed

- Clone the repository
  
  ```shell
  git clone https://github.com/dim-papach/rinla_project.git# clone the repository
  
  cd rinla_project# move into the directory
  ```

- Build the Nix environment
  
  ```shell
  nix-build
  ```

- Initiate and use the environment
  
  ```shell
  nix-shell
  ```
  
  This opens a bash shell in you current directory and you can run your scripts
  
  ---

# Quick Guides

- [Git](./docs/git.md)
- [Nix](./docs/nix.md)
