

It is possible to just download neccesary packages via CRAN or the package manager of your choice. However it is preferred if you used the Nix package manager, since it focuses on making the code reproducible and easier to deploy.

# What is Nix?

Nix is a powerful package manager for Linux and other Unix-like systems. It allows users to install, upgrade, and manage software packages in a reliable and reproducible way. Nix uses a purely functional approach to package management, ensuring that packages are isolated and do not interfere with each other.

# Why Use Nix?

- **Reproducibility:** Nix ensures that your software environment is reproducible across different machines, making it easier to share and collaborate on data analysis projects.
- **Isolation:** Nix isolates packages from each other, reducing the risk of conflicts and dependency issues.
- **Declarative Configuration:** Nix uses a declarative language to define package dependencies, making it easy to manage and reproduce software environments.

# How Nix Works:

Nix uses a functional programming language called Nix expressions to define packages and their dependencies. Each package is built in isolation and stored in a unique directory, ensuring that packages do not interfere with each other. Nix also supports garbage collection to remove unused packages and optimize disk space usage.

# Using Nix for Data Analysis:

Nix is particularly useful for data analysis tasks due to its ability to create isolated environments for different projects. This means you can easily set up and switch between different versions of libraries and tools without worrying about conflicts. Nix also allows you to define your project dependencies in a declarative way, making it easier to reproduce your analysis environment on different machines.

# The `rix` tool

`rix` is a package written in R, that makes the creation of nix environments easier.

The purpose of `rix` is to specify the environment requirements using the `rix()` function provided by the package. When you call `rix()`, it generates a file named `default.nix`. This file is utilized by the Nix package manager to create the specified environment. Subsequently, you can utilize this environment for interactive work or to execute R scripts.

# Quick Tutorial:

## How to use Nix

- Install Nix on your system by following the installation instructions on the official Nix website.
  
  - Linux [Multi-user installation (recommended)]
    
    ```shell
    sh <(curl -L https://nixos.org/nix/install) --daemon
    ```
  
  - MacOS
    
    ```shell
    sh <(curl -L https://nixos.org/nix/install)
    ```
  
  - Windows
    
    You need WSL versions 0.67.6 and above. Follow [Microsoft’s systemd guide](https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl) to configure it
    
    ```shell
    sh <(curl -L https://nixos.org/nix/install) --daemon
    ```

- Create a `shell.nix` or a `default.nix` file in your project directory to define your project dependencies. Here&rsquo;s an example:

```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python
    pkgs.pandas
  ];
}
```

For this project the `default.nix` file already exists

- Run `nix-build` in your project directory to build an environment according to the specifications found in the `default.nix` file. Then run `nix-shell` to use the environment.
  
  ```shell
  nix-build
  
  nix-shell --pure #the --pure flag is so only the packages specified are loaded in the env. If you want some system packages inside the env you can ignore it
  ```

- You can now run your code in this isolated environment.

---

# Useful reads

- <https://b-rodrigues.github.io/rix/index.html>
- <https://nix.dev/manual/nix/2.18/>
- <https://discourse.nixos.org/t/nix-shells-tips-tricks-and-best-practices/22332>
