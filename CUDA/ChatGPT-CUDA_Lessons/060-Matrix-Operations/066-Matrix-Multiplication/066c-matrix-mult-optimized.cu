#include <cuda_runtime.h>
#include <iostream>

// Matrix dimensions
const int rowsA = 2;
const int colsA = 2;
const int rowsB = 2;
const int colsB = 2;

// Thread block size
const int blockSize = 16;

// Device memory pointers
float *d_A, *d_B, *d_C;

// Host memory pointers
float *h_A, *h_B, *h_C;

// Kernel function to perform matrix multiplication
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int rowsA, int colsA, int rowsB, int colsB) {
  // Declare shared memory for tile of A and B
  __shared__ float tileA[blockSize][blockSize];
  __shared__ float tileB[blockSize][blockSize];

  // Compute row and column indices for the current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize result for this thread
  float result = 0;
  
  // Initialize shared memory
  for (int i = 0; i < blockDim.y; i++) {
    for (int j = 0; j < blockDim.x; j++) {
      tileA[i][j] = (i < rowsA && j < colsA) ? A[i * colsA + j] : 0;
      tileB[i][j] = (i < rowsB && j < colsB) ? B[i * colsB + j] : 0;
    }
  }
  __syncthreads();
  
  // Multiply tiles in shared memory
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      result += tileA[threadIdx.y + i][threadIdx.x] * tileB[threadIdx.y][threadIdx.x + j];
    }
  }

  // Write result to global memory
  if (row < rowsA && col < colsB) {
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
  h_A[0] = 1; h_A[1] = 2; h_A[2] = 3; h_A[3] = 4;
  h_B[0] = 5; h_B[1] = 6; h_B[2] = 7; h_B[3] = 8;

  // Copy host matrices to device
  cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

  // Launch matrix multiplication kernel
  int threadsPerBlock = blockSize;
  dim3 blockDim(threadsPerBlock, threadsPerBlock);
  dim3 gridDim((colsB + threadsPerBlock - 1) / threadsPerBlock, (rowsA + threadsPerBlock - 1) / threadsPerBlock);

  matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, rowsB, colsB);
  
  cudaDeviceSynchronize();

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


