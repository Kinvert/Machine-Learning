#include <cuda_runtime.h>
#include <stdio.h>

#define MESSAGE_LENGTH 11

// Kernel function
__global__ void hello_kernel(char *message)
{
    // Calculate the thread index
    int i = blockIdx.x;

    // Print the character at the thread index
    printf("%c", message[i]);
}

int main()
{
    // Allocate device memory for the message
    char *message;
    cudaMalloc((void **)&message, MESSAGE_LENGTH * sizeof(char));

    // Copy the message from the host to the device
    cudaMemcpy(message, "Hello World", MESSAGE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);

    // Launch the kernel
    hello_kernel<<<MESSAGE_LENGTH, 1>>>(message);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(message);

    printf("\n");

    return 0;
}
