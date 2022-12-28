#include <iostream>

// Matrix dimensions
const int rowsA = 2;
const int colsA = 2;
const int rowsB = 2;
const int colsB = 2;

// Host memory for matrices
float A[rowsA][colsA] = {{1, 2}, {3, 4}};
float B[rowsB][colsB] = {{5, 6}, {7, 8}};
float C[rowsA][colsB];

int main() {
  // Perform matrix multiplication
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      for (int k = 0; k < colsA; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // Print result matrix
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      std::cout << C[i][j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
