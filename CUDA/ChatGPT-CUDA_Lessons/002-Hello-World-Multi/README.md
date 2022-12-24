One way to showcase the parallel nature of CUDA C++ after the "Hello World" program is to write a program that executes a kernel on multiple threads.

Here is an example of a program that launches a kernel with a 2D grid of threads, and each thread prints a message with its thread index:

```cpp
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
```

This program launches a kernel with a grid of `GRID_SIZE_X` by `GRID_SIZE_Y` threads, and each thread prints a message with its 2D thread index (`x` and `y`). When executed on the GPU, this program will print 16 messages, one for each thread in the grid.

You can adjust the size of the grid and the size of the blocks to control the number of threads that are launched. For example, increasing the block size to (2, 2) will launch a grid of 2x2 blocks, each with 2x2 threads, for a total of 16 threads.

I hope this helps! Let me know if you have any further questions

## Code Explanation

Sure, here is a line-by-line explanation of the code example:

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define GRID_SIZE_X 4
#define GRID_SIZE_Y 4
```

These lines include the `cuda_runtime.h` header file, which provides access to the CUDA runtime API, and the `stdio.h` header file, which provides access to the `printf` function.

Next, we define two constants: `GRID_SIZE_X` and `GRID_SIZE_Y`. These constants control the size of the 2D grid of threads that will be launched by the kernel.

```cpp
// Kernel function
__global__ void hello_kernel()
{
    // Calculate the 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Hello from thread (%d, %d)\n", x, y);
}
```

This is the kernel function, which is executed on the GPU. It has no arguments and no return value.

Inside the kernel function, we calculate the 2D thread index using the `blockIdx` and `threadIdx` variables. These variables are special variables provided by the CUDA runtime API that are unique to each thread in a kernel launch.

The `blockIdx` variable is a 3D vector that represents the index of the thread block within the grid of blocks. The `threadIdx` variable is a 3D vector that represents the index of the thread within its block.

To calculate the 2D thread index, we use the following formulas:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

The `blockDim` variable is a 3D vector that represents the size of the thread block. In this case, we are using blocks of size (1, 1, 1), so we can use `blockDim.x` and `blockDim.y` to calculate the 2D thread index.

Finally, we use the `printf` function to print a message that includes the 2D thread index.

```cpp
int main()
{
    dim3 grid_size(GRID_SIZE_X, GRID_SIZE_Y);
    dim3 block_size(1, 1);

    hello_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

In the `main` function, we define a `dim3` variable grid_size that represents the size of the grid of thread blocks. We use the `GRID_SIZE_X` and `GRID_SIZE_Y` constants to set the size of the grid.

We also define a `dim3` variable `block_size` that represents the size of the thread block. In this case, we are using blocks of size (1, 1, 1).

Then, we launch the kernel by calling the `hello_kernel` function and passing it the `grid_size` and `block_size` variables. The `<<<>>>` operator is used to launch a kernel and specifies the grid and block dimensions.

After the kernel is launched, we call the `cudaDeviceSynchronize` function to wait for the kernel to finish executing. This ensures that all threads have completed before the `main` function returns.

Finally, we return 0 to indicate that the program has run successfully.

This code example demonstrates how to launch a 2D grid of threads in CUDA C++ and how to calculate the 2D thread index using the `blockIdx` and `threadIdx` variables. It also shows how to use the `printf` function to print messages from the kernel.

## Output

Here is an example of the output that you might see when running this code:

```cpp
Hello from thread (0, 0)!
Hello from thread (1, 0)!
Hello from thread (2, 0)!
Hello from thread (3, 0)!
Hello from thread (4, 0)!
Hello from thread (0, 1)!
Hello from thread (1, 1)!
Hello from thread (2, 1)!
Hello from thread (3, 1)!
Hello from thread (4, 1)!
```

Note that the order of the messages may not be exactly as shown above, due to the parallel nature of the threads. However, you should see that the messages are printed in a grid pattern, with the first coordinate representing the column and the second coordinate representing the row.

This example shows how to use CUDA C++ to launch a 2D grid of threads and how to calculate the 2D thread index using the `blockIdx` and `threadIdx` variables. It also demonstrates how to use the `printf` function to print messages from the kernel.

**Note: Here is the output in my case**

<img alt="Fashion-MNIST" width="280px" src="https://github.com/Kinvert/Machine-Learning/blob/master/CUDA/ChatGPT-CUDA_Lessons/002-Hello-World-Multi/output.png" />
