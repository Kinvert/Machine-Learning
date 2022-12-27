# Matrix Addition

- Introduction to matrix addition
  - Definition of matrix addition
  - Example of matrix addition
- Matrix addition in C++
  - Definition of matrix addition function
  - Example of matrix addition using a nested loop
- Naive matrix addition in CUDA
  - Definition of kernel function for matrix addition
  - Example of launching the kernel with one thread block
- Optimized matrix addition in CUDA
  - Use of shared memory to improve performance
  - Tiling to reduce global memory access
  - Example of optimized matrix addition kernel
  
## Introduction
  
We can start by discussing the introduction to matrix addition.

Matrix addition is a fundamental operation in linear algebra that involves adding corresponding elements in two matrices to create a new matrix. For example, given two matrices $A$ and $B$ with the same size, the matrix sum $C$ is calculated as follows:

$C_{i,j} = A_{i,j} + B_{i,j}$

where $C_{i,j}$ is the element in the $i$-th row and $j$-th column of matrix $C$, and $A_{i,j}$ and $B_{i,j}$ are the corresponding elements in matrices $A$ and $B$, respectively.

For example, given the following matrices $A$ and $B$:

$A = {\left\lbrack \matrix{A_{1,1} & A_{1,2} \cr A_{2,1} & A_{2,2}} \right\rbrack} = {\left\lbrack \matrix{1 & 2 \cr 3 & 4} \right\rbrack}$

$B = {\left\lbrack \matrix{B_{1,1} & B_{1,2} \cr B_{2,1} & B_{2,2}} \right\rbrack} = {\left\lbrack \matrix{5 & 6 \cr 7 & 8} \right\rbrack}$

The matrix sum $C$ is calculated as follows:

$C = {\left\lbrack \matrix{C_{1,1} & C_{1,2} \cr C_{2,1} & C_{2,2}} \right\rbrack} = {\left\lbrack \matrix{1+5 & 2+6 \cr 3+7 & 4+8} \right\rbrack}$

Which results in the following matrix $C$:

$C = {\left\lbrack \matrix{C_{1,1} & C_{1,2} \cr C_{2,1} & C_{2,2}} \right\rbrack} = {\left\lbrack \matrix{6 & 8 \cr 10 & 12} \right\rbrack}$

Matrix addition is an important operation that is used in many applications, including image processing, machine learning, and scientific computing. In this lesson, we will learn how to perform matrix addition using C++ and the CUDA programming platform.

## Matrix Addition in C++

The C++ program provided above defines a function `matrixAdd` that takes two matrices as input and returns their sum as a new matrix. The function first checks that the input matrices have the same size, and then performs matrix addition element by element using a nested loop.

The main function initializes two matrices `A` and `B`, and then calls `matrixAdd` to compute their sum, which is stored in a matrix `C`. The result is then printed to the console.

Here is a breakdown of the code:

```cpp
// Matrix type
typedef std::vector<std::vector<float>> Matrix;
```

This line defines a new type `Matrix` as a two-dimensional vector of `float` values.

```cpp
Matrix matrixAdd(const Matrix &A, const Matrix &B) {
    // Check that the matrices have the same size
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        std::cout << "Error: matrices have different sizes" << std::endl;
        exit(1);
    }

    // Create a result matrix with the same size as the input matrices
    Matrix C(A.size(), std::vector<float>(A[0].size()));

    // Perform matrix addition element by element
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < A[0].size(); j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    return C;
}
```

The function first checks that the input matrices have the same size. If they do not, it prints an error message and exits the program.

If the matrices have the same size, the function creates a new matrix `C` with the same size as the input matrices, and then performs matrix addition element by element using a nested loop. The result is then returned as a new matrix.

Finally, the main function initializes two matrices `A` and `B`, and then calls matrixAdd to compute their sum, which is stored in a matrix `C`. The result is then printed to the console.

## Naive CUDA Approach

```cpp
// Matrix type
typedef std::vector<std::vector<float>> Matrix;
```

This defines a new type `Matrix`, which is an alias for a `vector` of `vector` of `float` values. This allows us to use the more descriptive name `Matrix` instead of `vector<vector<float>>` when declaring variables of this type.

```cpp
// Kernel function for matrix addition
__global__ void matrixAddKernel(float *A, float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        C[row * numCols + col] = A[row * numCols + col] + B[row * numCols + col];
    }
}
```

This is the kernel function for our matrix addition. It's a CUDA kernel, indicated by the `__global__` modifier before the return type. The kernel function takes in four arguments: two pointers to `float` values (`A` and `B`), a pointer to `float` value (`C`), and two `int` values (`numRows` and `numCols`).

The kernel function is executed in parallel by many threads, each of which is identified by a unique set of indices (`blockIdx`, `threadIdx`). The indices are used to compute the row and column indices of the element that the thread should process. If the row and column indices are less than the number of rows and columns in the matrices, respectively, the thread performs the matrix addition by adding the element at the corresponding indices in `A` and `B` and storing the result in `C`.

This kernel function is designed to be able to process any element of the matrices, so it can handle matrices of any size. However, it is important to note that the kernel function will only be efficient for larger matrices. For small matrices, the overhead of launching the kernel may outweigh the benefits of parallel processing.

Here is the main function of the program:

```cpp
int main() {
    // Initialize two matrices
    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};

    // Flatten matrices into single arrays
    std::vector<float> A_flat;
    std::vector<float> B_flat;
    for (int i = 0; i < A.size(); i++) {
        A_flat.insert(A_flat.end(), A[i].begin(), A[i].end());
        B_flat.insert(B_flat.end(), B[i].begin(), B[i].end());
    }
```

This code initializes two matrices `A` and `B` and assigns them some values. It then flattens the matrices into single arrays `A_flat` and `B_flat` by inserting the elements of each row of the matrix into the respective array.

```cpp
// Allocate memory on the GPU
    int size = A_flat.size();
    int numRows = A.size();
    int numCols = A[0].size();
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
```

This code allocates memory on the GPU for three float pointers `d_A`, `d_B`, and `d_C`. `size` is the size of the flattened arrays, `numRows` is the number of rows in the original matrices, and `numCols` is the number of columns in the original matrices. The `cudaMalloc` function is used to allocate the memory on the GPU.

```cpp
// Copy data from host to device
    cudaMemcpy(d_A, A_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);
```

This code copies the data from the host (CPU) to the device (GPU) for the two flattened arrays `A_flat` and `B_flat`. The `cudaMemcpy` function is used to perform the copy, with the first argument being the destination on the device, the second argument being the source on the host, the third argument being the size of the data to be copied, and the fourth argument being the direction of the copy (`cudaMemcpyHostToDevice` indicates a copy from the host to the device).

```cpp
// Set block and grid sizes
int blockSize = 16;
dim3 blockDim(blockSize, blockSize, 1);
dim3 gridDim((numCols + blockSize - 1) / blockSize, (numRows + blockSize - 1) / blockSize, 1);
```

In this code snippet, we are setting the block and grid sizes for our CUDA kernel launch.

The `blockSize` variable is set to 16, which determines the number of threads in a block.

The `blockDim` variable is a `dim3` object, which represents the dimensions of a block. In this case, we set it to have a size of 16x16x1, meaning that our block will have 16x16=256 threads.

The `gridDim` variable is also a `dim3` object, which represents the dimensions of the grid of blocks. In this case, we set it to have dimensions `(numCols + blockSize - 1) / blockSize`, `(numRows + blockSize - 1) / blockSize`, and 1. This means that the grid will be divided into blocks of size `blockDim`, with `((numCols + blockSize - 1) / blockSize)` blocks in the x-dimension and `((numRows + blockSize - 1) / blockSize)` blocks in the y-dimension.

Finally, we launch the kernel function `matrixAddKernel` using the `<<<gridDim, blockDim>>>` syntax, passing in the pointers to the matrices `d_A`, `d_B`, and `d_C`, as well as the number of rows and columns in the matrices `numRows` and `numCols`.

```cpp
// Copy data from device to host
std::vector<float> C_flat(size);
cudaMemcpy(C_flat.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
```

This code block copies the data in the GPU memory location pointed to by `d_C` to the host memory location pointed to by `C_flat.data()`. `cudaMemcpy` is a function provided by the CUDA runtime API that allows data to be copied between the host and device. The fourth argument specifies the direction of the copy, in this case `cudaMemcpyDeviceToHost` indicates that the copy is from device to host. The third argument specifies the size of the data being copied, in this case `size * sizeof(float)` bytes.

```cpp
// Convert flattened array to matrix
Matrix C(numRows, std::vector<float>(numCols));
for (int i = 0; i < C.size(); i++) {
    for (int j = 0; j < C[0].size(); j++) {
        C[i][j] = C_flat[i * numCols + j];
    }
}
```

This code converts the flattened array `C_flat` back into a matrix form. The matrix is initialized with `numRows` number of rows and `numCols` number of columns. The two nested for loops iterate through each element in the matrix and assign the corresponding element in the flattened array to it. The flattened array index is computed using the row and column indices, which ensures that the elements are assigned in the correct order.

```cpp
// Print the result
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
```

This code snippet is responsible for printing the final result of the matrix addition. It first prints a string "Result:", then enters a loop that iterates through each row and column of the matrix `C`. For each element in the matrix, it prints the value followed by a space. After printing all the values in a row, it moves to the next row by printing a newline character.

```cpp
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

This is the final portion of the main program. It is responsible for freeing the memory that was allocated on the GPU for the matrices `A`, `B`, and `C`. This is important because it is good practice to free memory that is no longer needed to avoid memory leaks.

The `cudaFree` function takes in a pointer to the memory location that needs to be freed and frees it. In this case, we pass in the pointers `d_A`, `d_B`, and `d_C` which are the pointers to the memory locations of the matrices `A`, `B`, and `C` on the GPU.

Finally, the `main` function returns 0 to indicate that the program has completed successfully.

## Optimized matrix addition in CUDA

```cpp
#define BLOCK_SIZE 16

// Kernel function for matrix addition
__global__ void matrixAddKernel(float *A, float *B, float *C, int numRows, int numCols) {
```

In this code snippet, the kernel function `matrixAddKernel` is defined for performing matrix addition on two matrices `A` and `B` and storing the result in `C`. The kernel function takes in four arguments:

`float *A`: A pointer to the matrix A, which is stored in device memory.
`float *B`: A pointer to the matrix B, which is stored in device memory.
`float *C`: A pointer to the matrix C, which will store the result of the matrix addition and is also stored in device memory.
`int numRows`: The number of rows in the matrices A, B, and C.
`int numCols`: The number of columns in the matrices A, B, and C.
The `__global__` keyword indicates that this function will be called from the host (CPU) and executed on the device (GPU).

Now, let's consider a naive implementation of matrix addition using CUDA. In this implementation, we launch a kernel for each element in the output matrix C, with each thread taking care of one element. The kernel function would look something like this:

```cpp
__global__ void matrixAddKernel(float *A, float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    C[row * numCols + col] = A[row * numCols + col] + B[row * numCols + col];
}
```

Here, we launch `numRows` blocks, each with `numCols` threads. Each thread calculates the corresponding element of the output matrix `C` using the corresponding elements from the input matrices `A` and `B`.

This naive implementation has a few drawbacks:

- It requires a large number of threads, which can lead to inefficient utilization of the GPU resources.
- It has a high global memory access frequency, as each thread accesses the input matrices `A` and `B` and the output matrix `C` stored in global memory. This can lead to slow execution times.

To address these issues, the kernel function in the provided code snippet uses a technique called tiling, which involves dividing the matrices A and B into smaller sub-matrices or tiles and storing them in shared memory. This reduces the global memory access frequency, as shared memory has much lower access latencies compared to global memory.

The tiling is implemented as follows:

- The matrices `A` and `B` are tiled by storing them in shared memory arrays `sharedA` and `sharedB`, respectively.
- The tiling is performed in a way that each tile fits within a single block of threads.
- The size of each tile is determined by the `BLOCK_SIZE` macro, which is set to 16 in this case.
- The indices of the threads within a block are used to index into the shared memory arrays `sharedA` and `sharedB`.

For example, if the `BLOCK_SIZE` is set to 16, each block of threads will process a 16x16 tile of the input matrices `A` and `B`. The threads within a block can access the elements of the tile using their indices within the block.

Once the input matrices are tiled and stored in shared memory, the matrix addition can be performed using the shared memory arrays `sharedA` and `sharedB`. This reduces the global memory access frequency and improves the performance of the kernel.

Finally, the kernel function uses the shared memory arrays `sharedA` and `sharedB` to perform the matrix addition. If the current thread has a valid index within the input matrices, it writes the result to the output matrix `C`.

This approach to matrix addition using shared memory and tiling is an improvement over the naive approach of directly accessing the input matrices `A` and `B` in global memory. By reducing the number of global memory accesses and utilizing shared memory, this approach can improve the performance of the kernel. However, it should be noted that this approach may not always provide the best performance, and it may be necessary to explore other optimization techniques depending on the specific problem and hardware configuration.

```cpp
// Matrix type
typedef std::vector<std::vector<float>> Matrix;

#define BLOCK_SIZE 16

// Kernel function for matrix addition
__global__ void matrixAddKernel(float *A, float *B, float *C, int numRows, int numCols) {
    // Use shared memory to improve performance
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
```

This code defines the `matrixAddKernel` function, which is a CUDA kernel function that will be executed on the GPU. The function takes in two input matrices `A` and `B`, and an output matrix `C`. The function also takes in the number of rows and columns in the matrices.

The kernel function uses shared memory to improve performance. Shared memory is a type of memory that is shared among threads in a thread block. It is faster to access shared memory than global memory (the memory visible to all threads). The kernel function declares two shared memory arrays `sharedA` and `sharedB` for storing the input matrices.

```cpp
// Calculate the row and column indices of the element to be processed
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```

This code calculates the row and column indices of the element that the current thread will process. The `blockIdx` and `threadIdx` variables are built-in variables in CUDA that hold the block and thread indices, respectively. The `blockDim` variable holds the dimensions of the thread block.

```cpp
    // Initialize shared memory
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sharedA[i][j] = (i < numRows && j < numCols) ? A[i * numCols + j] : 0;
            sharedB[i][j] = (i < numRows && j < numCols) ? B[i * numCols + j] : 0;
        }
    }
```

This code initializes the shared memory arrays `sharedA` and `sharedB` with the values from the input matrices `A` and `B`. If the indices `i` and `j` are out of bounds (greater than or equal to the number of rows or columns), the elements are initialized to 0.

```cpp
    __syncthreads();
```

This code inserts a synchronization point that forces all threads in the block to wait until all threads have reached this point before continuing. This is necessary to ensure that the shared memory arrays are fully initialized before the matrix addition is performed.

```cpp
    // Matrix addition
    if (row < numRows && col < numCols) {
        C[row * numCols + col] = sharedA[threadIdx.y][threadIdx.x] + sharedB[threadIdx.y][threadIdx.x];
    }
}
```

This code performs the matrix addition using the shared memory arrays sharedA and sharedB. The element at position `(threadIdx.y, threadIdx.x)` in the shared memory arrays is added to the element at position(`row`, `col`) in the output matrix.

```cpp
int main() {
    // Initialize two matrices
    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};

    // Flatten matrices into single arrays
    std::vector<float> A_flat;
    std::vector<float> B_flat;
    for (int i = 0; i < A.size(); i++) {
        A_flat.insert(A_flat.end(), A[i].begin(), A[i].end());
        B_flat.insert(B_flat.end(), B[i].begin(), B[i].end());
    }
```

This code initializes two matrices `A` and `B` and flattens them into single arrays `A_flat` and `B_flat`, respectively. The matrices are flattened by iterating over each row and inserting its elements into the end of the corresponding flattened array. This is done using the `insert` function, which takes an iterator to the position in the array where the elements should be inserted, and two iterators to the first and past-the-end elements of the range of elements to be inserted. In this case, the range of elements is defined by the `begin` and `end` iterators of each row of the matrix. This results in `A_flat` and `B_flat` being filled with the elements of `A` and `B`, respectively, in row-major order.

```cpp
    // Allocate memory on the GPU
    int size = A_flat.size();
    int numRows = A.size();
    int numCols = A[0].size();
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
```

In this code snippet, the input matrices are first flattened into single arrays `A_flat` and `B_flat` using a nested loop. Then, the size of the matrices, as well as the number of rows and columns, are calculated.

Next, three arrays on the GPU are allocated using cudaMalloc. The arrays `d_A`, `d_B`, and `d_C` will be used to store the data from `A_flat`, `B_flat`, and the result of the matrix addition, respectively. The size of each array is equal to the number of elements in the input matrices, which is calculated as `size`.

This is an important step in the GPU matrix addition process because it allows the data to be transferred from the host (CPU) to the device (GPU) for processing. This is necessary because the GPU has its own memory space, which is separate from the main memory of the host. Allocating memory on the GPU and transferring data from the host to the device is a common pattern in GPU programming, and is necessary to take advantage of the parallel processing capabilities of the GPU.

```cpp
    // Copy data from host to device
    cudaMemcpy(d_A, A_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);
```

This code copies the data from the host memory to the device memory. `cudaMemcpy` is a function provided by the CUDA runtime library that allows you to copy data between host and device memory. The function takes four arguments:

- A pointer to the destination memory on the device. In this case, `d_A` and `d_B` are pointers to the device memory where the matrices `A` and `B` will be stored, respectively.

- A pointer to the source memory on the host. `A_flat.data()` and `B_flat.data()` are pointers to the beginning of the `std::vector` objects `A_flat` and `B_flat`, respectively.

- The size of the data to be copied, in bytes. `size * sizeof(float)` calculates the size of the data by multiplying the number of elements in the vectors by the size of a single element (a float).

- The direction of the copy. `cudaMemcpyHostToDevice` specifies that the data is being copied from the host to the device.

This code is necessary because the kernel function `matrixAddKernel` runs on the device, and it needs to operate on data stored in the device memory. By copying the data from the host to the device, we make it accessible to the kernel function.

```cpp
    // Set block and grid sizes
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize, 1);
    dim3 gridDim((numCols + blockSize - 1) / blockSize, (numRows + blockSize - 1) / blockSize, 1);

    // Launch kernel function
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, numRows, numCols);
```

The `blockSize` variable is defined as 16, which is the size of the shared memory arrays sharedA and sharedB.

The `blockDim` variable specifies the number of threads in each block. In this case, it is set to a 2D block of size `(16, 16, 1)`, which means each block has 16 rows and 16 columns of threads, and 1 layer.

The `gridDim` variable specifies the number of blocks in the grid. It is set to a 2D grid of size `((numCols + blockSize - 1) / blockSize, (numRows + blockSize - 1) / blockSize, 1)`, which means the grid has `((numCols + 15) / 16, (numRows + 15) / 16)` number of blocks. This is used to ensure that all elements in the input matrices are processed.

Finally, the kernel function is launched using the `<<<gridDim, blockDim>>>` syntax, with `d_A`, `d_B`, `d_C`, `numRows`, and `numCols` as the arguments. This launches the kernel function on the GPU with the specified grid and block configurations.

```cpp
    cudaDeviceSynchronize();
```

The `cudaDeviceSynchronize` function is a blocking function that waits for all previously launched kernel functions to complete before it returns. This function is useful for ensuring that the kernel has completed before continuing with the host code. In this case, it is used to ensure that the `matrixAddKernel` has completed before the next step of copying the data from the device to the host.

It is important to note that `cudaDeviceSynchronize` should be used sparingly as it introduces a performance overhead. It should only be used when it is necessary to ensure that the kernel has completed before continuing with the host code. In general, it is better to use non-blocking functions, such as `cudaMemcpyAsync`, which allow the host code to continue while the kernel is still running.

```cpp
    // Copy data from device to host
    std::vector<float> C_flat(size);
    cudaMemcpy(C_flat.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
```

The `cudaMemcpy` function is used to copy data from device memory to host memory. In this case, the data stored in the device memory at the address pointed to by `d_C` is copied to the host memory at the address pointed to by `C_flat.data()`. The size parameter specifies the number of bytes to be copied, and the `cudaMemcpyDeviceToHost` parameter indicates that the data is being copied from the device to the host.

This is useful because the kernel function `matrixAddKernel` is executed on the GPU, and the result of the matrix addition is stored in device memory. To access the result on the host side, the data needs to be copied from the device memory to the host memory.

After the data is copied, the `C_flat` vector contains the result of the matrix addition, which can be used to reconstruct the matrix `C`.

```cpp
    // Convert flattened array to matrix
    Matrix C(numRows, std::vector<float>(numCols));
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            C[i][j] = C_flat[i * numCols + j];
        }
    }
```

This code converts the flattened array `C_flat` back into a matrix `C`. It does this by looping through each element in the matrix and setting the value to the corresponding element in the flattened array. The matrix indices `i` and `j` are used to calculate the corresponding index in the flattened array, which is `i * numCols + j`. This allows the matrix to be reconstructed by iterating through each row and column of the matrix.

```cpp
    // Print the result
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
```

This code snippet converts the flattened result array `C_flat` into a matrix `C`, which is then printed to the console.

The matrix `C` is initialized with `numRows` rows and `numCols` columns, using the `std::vector` constructor `std::vector<float>(numCols)`.

Then, a nested loop iterates over the rows and columns of `C`, setting each element to the corresponding element in `C_flat`. The element at position `(i, j)` in `C` is set to the element at index `i * numCols + j` in `C_flat`.

Finally, the matrix `C` is printed to the console using another nested loop. This prints each element of `C` separated by a space, and each row on a new line.

```cpp
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

This is the end of the main function. It prints the result of the matrix addition to the console and then frees the GPU memory that was allocated earlier. Finally, it returns 0 to indicate that the program has completed successfully.

This code demonstrates how to perform matrix addition using CUDA. The kernel function uses shared memory to reduce global memory access frequency and improve performance. The input matrices are tiled and stored in shared memory, and the matrix addition is performed using the shared memory arrays. The result is then copied back to the host and printed to the console.
