# FYF Docker Quickstart Guide

This guide provides instructions on how to build and run **FYF (Fill Your FITS)** using Docker. Using Docker ensures that all dependencies (Python, R, and R-INLA) are correctly configured without affecting your local system.

## 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running on your system.

## 2. Build the Docker Image

You can build the image using the provided script or manually.

### Using the Build Script
```bash
chmod +x docker_build_script.sh
./docker_build_script.sh
```

### Manual Build
```bash
docker build -t fyf .
```
*Note: The first build may take 10-15 minutes as it installs heavy spatial and statistical libraries (GDAL, R-INLA, etc.).*

## 3. Running the Container

The Docker image is configured to use `/app` as the working directory (containing the source code) and `/data` as a mounting point for your FITS files.

### Interactive Shell
To explore the environment or run commands manually:
```bash
docker run --rm -it -v /path/to/your/data:/data fyf
```

### Running Commands Directly
You can run `fyf` commands directly from your host machine:

```bash
# Get version info
docker run --rm fyf fyf version

# Show examples
docker run --rm fyf fyf examples
```

## 4. Common Workflows

### Processing Data
To process a FITS file located on your host at `/home/user/astro_data/image.fits`:

```bash
docker run --rm 
  -v /home/user/astro_data:/data 
  fyf fyf process /data/image.fits --method inla --shape radius -o /data/output
```
The results will be saved in `/home/user/astro_data/output` on your host machine.

### Simulating Artifacts
```bash
docker run --rm 
  -v /home/user/astro_data:/data 
  fyf fyf simulate /data/image.fits -c 0.02 -t 1 -o /data/simulated
```

### Validating Results
```bash
docker run --rm 
  -v /home/user/astro_data:/data 
  fyf fyf validate /data/original.fits /data/processed.fits --plot -o /data/validation
```

## 5. Summary of Key Paths
- **`/app`**: Source code and internal scripts.
- **`/data`**: Recommended mount point for your input/output FITS files.

## 6. Troubleshooting

- **Permissions**: If Docker cannot write to your mounted directory, ensure the host directory has appropriate write permissions.
- **Memory**: R-INLA can be memory-intensive. If the container crashes during processing, try increasing the memory limit in your Docker settings.
