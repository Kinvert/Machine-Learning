#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel()
{
    int i = blockIdx.x;
    printf("Hello, World from thread %d!\n", i);
}

int main()
{
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Hello, World from the host!\n");
    return 0;
}
