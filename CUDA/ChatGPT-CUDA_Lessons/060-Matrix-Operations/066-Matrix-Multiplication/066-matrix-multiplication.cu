#include <cuda_runtime.h>
#include <iostream>

// Matrix dimensions
const int rowsA = 4;
const int colsA = 3;
const int rowsB = 3;
const int colsB = 2;

// Thread block size
const int blockSize = 16;

// Device memory pointers
float *d_A, *d_B, *d_C;

// Host memory pointers
float *h_A, *h_B, *h_C;

// Kernel function to perform matrix multiplication
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int rowsA, int colsA, int rowsB, int colsB) {
  // Compute row and column indices for the current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize result for this thread
  float result = 0;

  // Perform matrix multiplication if indices are within bounds
  if (row < rowsA && col < colsB) {
    for (int k = 0; k < colsA; k++) {
      result += A[row * colsA + k] * B[k * colsB + col];
    }
    C[row * colsB + col] = result;
  }
}

int main() {
  // Allocate device memory
  cudaMalloc((void **)&d_A, rowsA * colsA * sizeof(float));
  cudaMalloc((void **)&d_B, rowsB * colsB * sizeof(float));
  cudaMalloc((void **)&d_C, rowsA * colsB * sizeof(float));

  // Allocate host memory
  h_A = (float *)malloc(rowsA * colsA * sizeof(float));
  h_B = (float *)malloc(rowsB * colsB * sizeof(float));
  h_C = (float *)malloc(rowsA * colsB * sizeof(float));

  // Initialize host matrices
  for (int i = 0; i < rowsA * colsA; i++) {
    h_A[i] = i;
  }
  for (int i = 0; i < rowsB * colsB; i++) {
    h_B[i] = i;
  }

  // Copy host matrices to device
  cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

  // Launch matrix multiplication kernel
  int threadsPerBlock = blockSize;
  int blocksPerGrid = (rowsA * colsB + threadsPerBlock - 1) / threadsPerBlock;
  matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, rowsB, colsB);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }

  // Copy result matrix from device to host
  cudaMemcpy(h_C, d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

  // Print result matrix
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      std::cout << h_C[i * colsB + j] << " ";
    }
    std::cout << std::endl;
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
