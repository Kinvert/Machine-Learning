## Introduction to CUDA C++: Writing and Executing a Kernel

In this lesson, we will learn how to write and execute a kernel in CUDA C++. A kernel is a function that is executed on the GPU by many threads concurrently. We will also learn how to allocate and transfer data between the host and device, and how to print the contents of an array.

## The Code
Here is the code example:

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000  // Size of the data array

// Kernel function
__global__ void hello_kernel(int *data, int size)
{
    // Calculate the thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Only write to the data array if the thread index is within the bounds of the array
    if (i < size)
    {
        data[i] = i;
    }
}

int main()
{
    // Allocate device memory for the data array
    int *data;
    cudaMalloc((void **)&data, N * sizeof(int));

    // Launch the kernel
    hello_kernel<<<1, N>>>(data, N);
    cudaDeviceSynchronize();

    // Allocate host memory and copy data from the device
    int *host_data = (int *)malloc(N * sizeof(int));
    cudaMemcpy(host_data, data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the contents of the data array
    for (int i = 0; i < N; i++)
    {
        printf("%d ", host_data[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(data);
    free(host_data);

    return 0;
}
```


