// NOTE: This one took a ton of hands on work to get it working. ChatGPT did not do a good job with this one.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Matrix type
typedef std::vector<std::vector<float>> Matrix;

// Kernel function for matrix addition
__global__ void matrixAddKernel(float *A, float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        C[row * numCols + col] = A[row * numCols + col] + B[row * numCols + col];
    }
}

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

    // Allocate memory on the GPU
    int size = A_flat.size();
    int numRows = A.size();
    int numCols = A[0].size();
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize, 1);
    dim3 gridDim((numCols + blockSize - 1) / blockSize, (numRows + blockSize - 1) / blockSize, 1);

    // Launch kernel function
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, numRows, numCols);

    // Copy data from device to host
    std::vector<float> C_flat(size);
    cudaMemcpy(C_flat.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert flattened array to matrix
    Matrix C(numRows, std::vector<float>(numCols));
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            C[i][j] = C_flat[i * numCols + j];
        }
    }

    // Print the result
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
