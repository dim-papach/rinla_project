#!/bin/bash
# run.sh - Shell script to run the FYF pipeline
#
# This script provides a simple way to run the FYF pipeline on multiple files
# with predefined configurations.

# Display help
function show_help {
  echo "Usage: $0 [options] <fits_file(s)>"
  echo "Options:"
  echo "  -c, --cosmic FLOAT    Cosmic ray fraction (default: 0.01)"
  echo "  -t, --trails INT      Number of satellite trails (default: 1)"
  echo "  -s, --shape STR       Shape model [none|radius|ellipse] (default: none)"
  echo "  -o, --output DIR      Output directory (default: ./output)"
  echo "  -r, --report          Generate HTML report"
  echo "  -p, --parallel        Process files in parallel"
  echo "  -d, --dry-run         Show commands without executing"
  echo "  -h, --help            Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -c 0.02 -t 2 -s radius -r data/*.fits"
  exit 0
}

# Default values
COSMIC_FRACTION=0.01
TRAILS=1
SHAPE="none"
OUTPUT_DIR="./output"
REPORT=false
PARALLEL=false
DRY_RUN=false

# Parse options
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--cosmic)
      COSMIC_FRACTION="$2"
      shift 2
      ;;
    -t|--trails)
      TRAILS="$2"
      shift 2
      ;;
    -s|--shape)
      SHAPE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -r|--report)
      REPORT=true
      shift
      ;;
    -p|--parallel)
      PARALLEL=true
      shift
      ;;
    -d|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      break
      ;;
  esac
done

# Check if any files are provided
if [ $# -eq 0 ]; then
  echo "Error: No FITS files specified"
  show_help
fi

# Check if fyf is installed
if ! command -v fyf &> /dev/null; then
  echo "Error: fyf command not found. Please install the FYF package."
  exit 1
fi

# Build the command
CMD="fyf pipeline --cosmic-fraction $COSMIC_FRACTION --trails $TRAILS --shape $SHAPE --output-dir $OUTPUT_DIR"

if [ "$REPORT" = true ]; then
  CMD="$CMD --report"
fi

# Process files
if [ "$PARALLEL" = true ]; then
  # Process in parallel using xargs if available
  if command -v xargs &> /dev/null; then
    if [ "$DRY_RUN" = true ]; then
      echo "Would run: echo $@ | xargs -P $(nproc) -I{} $CMD {}"
    else
      echo "Processing files in parallel..."
      echo "$@" | tr " " "\n" | xargs -P $(nproc) -I{} $CMD {}
    fi
  else
    echo "Warning: xargs not found, falling back to serial processing"
    PARALLEL=false
  fi
fi

# Serial processing
if [ "$PARALLEL" = false ]; then
  for file in "$@"; do
    if [ "$DRY_RUN" = true ]; then
      echo "Would run: $CMD $file"
    else
      echo "Processing: $file"
      $CMD "$file"
    fi
  done
fi

echo "All processing complete!"