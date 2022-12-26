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









