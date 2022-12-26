## Introduction

In this code example, we will learn how to execute a kernel in CUDA C++ that asynchronously prints a message to the console. We will also learn how to allocate device memory and transfer data between the host and device.

## The Code

Here is the code example:

```cpp
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

    return 0;
}
```

Let's go through the code line by line to understand how it works.

## Code Explanation

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
```

These lines include the `cuda_runtime.h` header file and the `stdio.h` header file. The `cuda_runtime.h` header file provides access to the CUDA runtime API, which includes functions for managing device memory, launching kernels, and synchronizing the host and device. The `stdio.h` header file provides access to standard input/output functions such as `printf`.

```cpp
#define MESSAGE_LENGTH 11
```

This line defines a macro that specifies the length of the message to be printed.

```cpp
// Kernel function
__global__ void hello_kernel(char *message)
{
    // Calculate the thread index
    int i = blockIdx.x;
    // Print the character at the thread index
    printf("%c", message[i]);
}
```

This is the definition of the `hello_kernel` kernel. The kernel is a function that is executed on the GPU by many threads concurrently. The `global` keyword specifies that the function is a kernel.

The kernel takes a single argument, a pointer to a character array that holds the `message` to be printed. Inside the kernel, the `blockIdx.x` variable holds the block index within the grid, and the `threadIdx.x` variable holds the thread index within the block. In this example, we are using a single thread per block, so the `blockIdx.x` variable is equal to the thread index.

The kernel prints the character at the thread index of the message array.

```cpp
int main()
{
    // Allocate device memory for the message
    char *message;
    cudaMalloc((void **)&message, MESSAGE_LENGTH * sizeof(char));
```

The `main` function is the entry point of the program. It contains the code that is executed on the host.

The first line of the `main` function allocates device memory for the message using the `cudaMalloc` function. The `cudaMalloc` function takes a pointer to a pointer and the size of the memory to be allocated in bytes as arguments. It returns a pointer to the allocated memory in the device.

The second line of the `main` function copies the message from the `host` to the `device` using the `cudaMemcpy` function. The `cudaMemcpy` function takes a pointer to the destination, a pointer to the source, the size of the memory to be copied in bytes, and the direction of the copy as arguments. In this example, we are copying the message from the `host` to the `device`.

```cpp
    // Launch the kernel
    hello_kernel<<<MESSAGE_LENGTH, 1>>>(message);
    cudaDeviceSynchronize();
```

This line launches the kernel using the `<<< >>>` syntax. The `<<<MESSAGE_LENGTH, 1>>>` syntax specifies the grid of thread blocks and the number of threads per block. In this example, we are using `MESSAGE_LENGTH` blocks of threads, each with a single thread.

The `hello_kernel` function is passed a pointer to the `message` on the device as an argument.

The `cudaDeviceSynchronize` function is called to synchronize the host and device. This function waits until all the kernels launched on the device have completed.

```cpp
    // Free device memory
    cudaFree(message);
```

This line frees the device memory that was allocated for the message using the `cudaMalloc` function.

```cpp
    return 0;
}
```

Finally, the main function returns 0 to indicate that the program has completed successfully.

The output of this program should be the message "Hello World" printed to the console, with each character printed by a different thread.

## Output

<img alt="OpenAI ChatGPT CUDA Lesson Hello World Result" width="280px" src="https://github.com/Kinvert/Machine-Learning/blob/master/CUDA/ChatGPT-CUDA_Lessons/010-Hello-World/013-Hello-World-Async/ChatGPT-CUDA-Lesson-013-Output.png" />
