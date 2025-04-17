from numba import cuda
import numpy as np

# Kernel function to add two arrays
@cuda.jit
def add_arrays_gpu(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]
        
# Host code
def main():
    # Init data (input arrays)
    n= 1000000
    a = np.arange(n).astype(np.float32)
    b = np.arange(n).astype(np.float32)
    
    # prepare GPU output array
    result = np.zeros(n).astype(np.float32)
    
    # transfer data to GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.to_device(result)
    
    # launch kernel with one thread per element
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    add_arrays_gpu[blocks_per_grid, threads_per_block](d_a, d_b, d_result)
    
    # copy result back to CPU
    d_result.copy_to_host(result)
    
    # print result
    print("Array a  :", a)
    print("Array b  :", b)
    print("a + b    :", result)
    
if __name__ == "__main__":
    main()