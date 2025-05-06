#!/bin/bash
# Run matrix multiplication test with low memory preallocation

# Default values
MATRIX_SIZE=2048
DURATION_MINUTES=5

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --size=*)
      MATRIX_SIZE="${1#*=}"
      shift
      ;;
    --duration=*)
      DURATION_MINUTES="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: ./run_matrix_test.sh [OPTIONS]"
      echo "Options:"
      echo "  --size=SIZE       Matrix size (default: 2048)"
      echo "  --duration=MINS   Test duration in minutes (default: 5)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Rename the script for clarity
mv test_chess_ai.py test_matrix_multiply.py 2>/dev/null || true

# Make sure test script is executable
chmod +x test_matrix_multiply.py

# Install required packages if not already installed
pip install psutil

echo "Running matrix multiplication test with size=${MATRIX_SIZE}, duration=${DURATION_MINUTES} minutes"
# Run the test with specified parameters
python test_matrix_multiply.py $MATRIX_SIZE $DURATION_MINUTES 