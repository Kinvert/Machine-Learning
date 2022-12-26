#include <iostream>
#include <cuda_runtime.h>

// define the number of threads in each thread block
#define NUM_THREADS_PER_BLOCK 128

// define the number of blocks
#define NUM_BLOCKS 8

// define the size of the input data
#define DATA_SIZE 1024

__global__ void square_array(float *d_out, float *d_in)
{
    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute the square of the input element
    // and store it in the output array
    d_out[i] = d_in[i] * d_in[i];
}

int main(int argc, char **argv)
{
    // allocate host arrays
    float h_in[DATA_SIZE];
    float h_out[DATA_SIZE];

    // initialize the input array with random values
    for (int i = 0; i < DATA_SIZE; i++)
    {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // allocate device arrays
    float *d_in;
    float *d_out;
    cudaMalloc(&d_in, DATA_SIZE * sizeof(float));
    cudaMalloc(&d_out, DATA_SIZE * sizeof(float));

    // copy the input array to the device
    cudaMemcpy(d_in, h_in, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel with the specified number of thread blocks and threads per block
    square_array<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_out, d_in);

    // copy the output array back to the host
    cudaMemcpy(h_out, d_out, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i = 0; i < DATA_SIZE; i++)
    {
        std::cout << h_in[i] << "^2 = " << h_out[i] << std::endl;
    }

    // free the device arrays
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
