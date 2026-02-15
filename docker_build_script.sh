#!/bin/bash
# docker-build.sh - Build script for FYF Docker image

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="fyf"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"

echo -e "${GREEN}=== Building FYF Docker Image ===${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: $DOCKERFILE not found in current directory${NC}"
    exit 1
fi

# Check if required files exist
echo "Checking project structure..."
required_files=("setup.py" "fyf/")
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo -e "${RED}Error: Required file/directory '$file' not found${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ Project structure OK${NC}"
echo ""

# Build the image
echo -e "${YELLOW}Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo "This may take several minutes on first build..."
echo ""

docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file "${DOCKERFILE}" \
    . \
    || { echo -e "${RED}Build failed!${NC}"; exit 1; }

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "Image built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the container:"
echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To process a FITS file:"
echo "  docker run --rm -v /path/to/data:/data ${IMAGE_NAME}:${IMAGE_TAG} <command> /data/file.fits"
echo ""
echo "To get a shell inside the container:"
echo "  docker run --rm -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash"
echo ""