#include <cuda_runtime.h>
#include <stdio.h>

#define GRID_SIZE_X 4
#define GRID_SIZE_Y 4

// Kernel function
__global__ void hello_kernel()
{
    // Calculate the 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Hello from thread (%d, %d)\n", x, y);
}

int main()
{
    // Launch the kernel with a 2D grid of threads
    dim3 grid_size(GRID_SIZE_X, GRID_SIZE_Y);
    dim3 block_size(1, 1);
    hello_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}
