#include <iostream>
#include <cuda_runtime.h>

// define the number of threads in each thread block
#define NUM_THREADS_PER_BLOCK 128

// define the size of the input data
#define DATA_SIZE 1024

// define the kernel function
__global__ void add_arrays(int *a, int *b, int *c)
{
    // calculate the thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if the thread index is within bounds
    if (i < DATA_SIZE)
    {
        // perform the addition
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv)
{
    // allocate host and device arrays
    int h_a[DATA_SIZE];
    int h_b[DATA_SIZE];
    int h_c[DATA_SIZE];
    int *d_a;
    int *d_b;
    int *d_c;
    cudaMalloc(&d_a, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_b, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_c, DATA_SIZE * sizeof(int));

    // initialize the input data
    for (int i = 0; i < DATA_SIZE; i++)
    {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    // copy the input data from host to device
    cudaMemcpy(d_a, h_a, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel
    add_arrays<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // copy the output data from device to host
    cudaMemcpy(h_c, d_c, DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // print the output data
    for (int i = 0; i < DATA_SIZE; i++)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
