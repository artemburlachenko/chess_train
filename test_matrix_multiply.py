#!/usr/bin/env python3
"""Simple test for JAX memory optimization with matrix multiplication"""

import os
import time
import sys
import psutil

# Set memory optimization parameters before importing JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"  # Lower than default 0.35
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific memory allocator
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"  # Reduce compilation parallelism

# Now import JAX after setting environment variables
import jax
import jax.numpy as jnp

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def matrix_multiplication_test(matrix_size=2048, duration_minutes=5):
    """Perform matrix multiplication operations once per minute
    
    Args:
        matrix_size: Size of square matrices to multiply
        duration_minutes: How long to run the test in minutes
    """
    print(f"Running matrix multiplication test with {matrix_size}x{matrix_size} matrices")
    print(f"Will run for {duration_minutes} minutes, one operation per minute")
    print(f"JAX devices available: {jax.devices()}")
    
    # Initial memory usage
    print("\nInitial memory state:")
    print_memory_usage()
    
    # JIT-compile the matrix multiplication function
    @jax.jit
    def matmul(x, y):
        return jnp.matmul(x, y)
    
    # Track operation count
    op_count = 0
    end_time = time.time() + (duration_minutes * 60)
    
    # Run the test loop
    while time.time() < end_time:
        op_count += 1
        print(f"\n--- Operation {op_count} ---")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        # Create random matrices using BF16 precision to reduce memory usage
        print("Creating random matrices...")
        key = jax.random.PRNGKey(int(time.time()))
        key, subkey = jax.random.split(key)
        matrix_a = jax.random.normal(key, (matrix_size, matrix_size), dtype=jnp.bfloat16)
        matrix_b = jax.random.normal(subkey, (matrix_size, matrix_size), dtype=jnp.bfloat16)
        
        # Perform matrix multiplication and measure time
        print("Performing matrix multiplication...")
        start_time = time.time()
        
        # First call compiles the function (may be slow)
        result = matmul(matrix_a, matrix_b)
        
        # Force the computation to complete
        result.block_until_ready()
        
        end_time_op = time.time()
        print(f"Matrix multiplication completed in {end_time_op - start_time:.4f} seconds")
        
        # Print memory usage after operation
        print("Memory usage after operation:")
        print_memory_usage()
        
        # Calculate some basic statistics about the result to ensure computation happened
        result_sum = jnp.sum(result).item()
        result_mean = jnp.mean(result).item()
        print(f"Result statistics - Sum: {result_sum:.4f}, Mean: {result_mean:.4f}")
        
        # Sleep until next minute if there's still time left
        if time.time() < end_time:
            # Calculate time to wait until next minute
            next_op_time = time.time() + 60  # One minute from now
            sleep_time = max(0, next_op_time - time.time())
            
            if sleep_time > 0:
                print(f"Waiting {sleep_time:.2f} seconds until next operation...")
                time.sleep(sleep_time)
    
    print("\n--- Test completed ---")
    print(f"Performed {op_count} matrix multiplications")
    print("Final memory usage:")
    print_memory_usage()

if __name__ == "__main__":
    # Set default matrix size and duration
    matrix_size = 2048
    duration_minutes = 5
    
    # Allow custom values from command line arguments
    if len(sys.argv) >= 2:
        try:
            matrix_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid matrix size: {sys.argv[1]}")
            print(f"Using default: {matrix_size}")
    
    if len(sys.argv) >= 3:
        try:
            duration_minutes = int(sys.argv[2])
        except ValueError:
            print(f"Invalid duration: {sys.argv[2]}")
            print(f"Using default: {duration_minutes}")
    
    matrix_multiplication_test(matrix_size, duration_minutes) 