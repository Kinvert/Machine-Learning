#include <iostream>
#include <vector>

// Matrix type
typedef std::vector<std::vector<float>> Matrix;

// Matrix addition function
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

int main() {
    // Initialize two matrices
    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};

    // Perform matrix addition
    Matrix C = matrixAdd(A, B);

    // Print the result
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
