#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cudnn.h"

#include "mnist.h"

#define checkCUDNN(expression)                             \
{                                                         \
    cudnnStatus_t status = (expression);                  \
    if (status != CUDNN_STATUS_SUCCESS) {                 \
        printf("Error on line %d: %s\n", __LINE__, #expression); \
        exit(1);                                           \
    }                                                     \
}

#define checkCudaErrors(expression)                        \
{                                                         \
    cudaError_t err = (expression);                       \
    if (err != cudaSuccess) {                             \
        printf("Error on line %d: %s\n", __LINE__, #expression); \
        exit(1);                                           \
    }                                                     \
}

#define checkCublasErrors(expression)                      \
{                                                         \
    cublasStatus_t status = (expression);                 \
    if (status != CUBLAS_STATUS_SUCCESS) {                \
        printf("Error on line %d: %s\n", __LINE__, #expression); \
        exit(1);                                           \
    }                                                     \
}

#define NUM_CLASSES 10
#define IMAGE_SIZE 784

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000

#define BATCH_SIZE 128

#define NUM_EPOCHS 10

#define LEARNING_RATE 0.01

// Helper function to set the dimensions of a 4D tensor
void setTensor4dDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w)
{
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
}

// Helper function to set the dimensions of a 2D tensor
void setTensor2dDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c)
{
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, 1, 1));
}

// Helper function to create a filter descriptor
void createFilterDesc(cudnnFilterDescriptor_t& filterDesc, int n, int c, int h, int w)
{
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w));
}

int main(int argc, char** argv)
{
    // Load MNIST dataset
    mnist_data* trainingData = NULL;
    unsigned int numTrainingImages = 0;
    mnist_data* testData = NULL;
    unsigned int numTestImages = 0;
    if (mnistLoad(&trainingData, &numTrainingImages, &testData, &numTestImages, "../../../zData/MNIST/train-images-idx3-ubyte.gz", "../../../zData/MNIST/train-labels-idx1-ubyte.gz", "../../../zData/MNIST/t10k-images-idx3-ubyte.gz", "../../../zData/MNIST/t10k-labels-idx1-ubyte.gz"))
    {
        printf("Error loading MNIST dataset.\n");
        return 1;
    }

    // Create CUDA objects
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    cublasHandle_t cublasHandle;
    checkCublasErrors(cublasCreate(&cublasHandle));
    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));

    // Allocate device memory for inputs and outputs
    float* dInputImages;
    checkCudaErrors(cudaMalloc((void**)&dInputImages, BATCH_SIZE * IMAGE_SIZE * sizeof(float)));
    float* dOutputLabels;
    checkCudaErrors(cudaMalloc((void**)&dOutputLabels, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    float* dOutputLogits;
    checkCudaErrors(cudaMalloc((void**)&dOutputLogits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    // Create tensor descriptors for input and output
    cudnnTensorDescriptor_t inputTensorDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
    setTensor2dDesc(inputTensorDesc, BATCH_SIZE, IMAGE_SIZE);
    cudnnTensorDescriptor_t outputTensorDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
    setTensor2dDesc(outputTensorDesc, BATCH_SIZE, NUM_CLASSES);

    // Create filter descriptor for fully connected layer weights
    cudnnFilterDescriptor_t filterDesc;
    createFilterDesc(filterDesc, NUM_CLASSES, IMAGE_SIZE, 1, 1);

    // Allocate device memory for fully connected layer weights and biases
    float* dFcWeights;
    checkCudaErrors(cudaMalloc((void**)&dFcWeights, NUM_CLASSES * IMAGE_SIZE * sizeof(float)));
    float* dFcBiases;
    checkCudaErrors(cudaMalloc((void**)&dFcBiases, NUM_CLASSES * sizeof(float)));

    // Allocate host memory for fully connected layer weights and biases
    float* hFcWeights = (float*)malloc(NUM_CLASSES * IMAGE_SIZE * sizeof(float));
    float* hFcBiases = (float*)malloc(NUM_CLASSES * sizeof(float));

    // Initialize fully connected layer weights and biases to random values
    for (int i = 0; i < NUM_CLASSES * IMAGE_SIZE; i++)
    {
        hFcWeights[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        hFcBiases[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Copy fully connected layer weights and biases to device memory
    checkCudaErrors(cudaMemcpy(dFcWeights, hFcWeights, NUM_CLASSES * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dFcBiases, hFcBiases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    // Create convolution descriptor for fully connected layer
    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Compute fully connected layer output tensor dimensions
    int fcOutputDim[4];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, 4, fcOutputDim));

    // Create tensor descriptor for fully connected layer output
    cudnnTensorDescriptor_t fcOutputTensorDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&fcOutputTensorDesc));
    setTensor4dDesc(fcOutputTensorDesc, fcOutputDim[0], fcOutputDim[1], fcOutputDim[2], fcOutputDim[3]);

    // Allocate device memory for fully connected layer output
    float* dFcOutput;
    checkCudaErrors(cudaMalloc((void**)&dFcOutput, BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    // Allocate device memory for fully connected layer gradients
    float* dFcOutputGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcOutputGrad, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    float* dFcWeightsGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcWeightsGrad, NUM_CLASSES * IMAGE_SIZE * sizeof(float)));
    float* dFcBiasesGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcBiasesGrad, NUM_CLASSES * sizeof(float)));

    // Allocate host memory for gradients
    float* hFcWeightsGrad = (float*)malloc(NUM_CLASSES * IMAGE_SIZE * sizeof(float));
    float* hFcBiasesGrad = (float*)malloc(NUM_CLASSES * sizeof(float));

    // Create operation descriptor for fully connected layer forward propagation
    cudnnConvolutionFwdAlgo_t fcFwdAlgo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputTensorDesc, filterDesc, convDesc, fcOutputTensorDesc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &fcFwdAlgo));

    // Allocate device memory for fully connected layer forward propagation workspace
    void* dFcFwdWorkspace;
    size_t fcFwdWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputTensorDesc, filterDesc, convDesc, fcOutputTensorDesc, fcFwdAlgo, &fcFwdWorkspaceSize));
    checkCudaErrors(cudaMalloc((void**)&dFcFwdWorkspace, fcFwdWorkspaceSize));

    // Create operation descriptor for fully connected layer backward data propagation
    cudnnConvolutionBwdDataAlgo_t fcBwdDataAlgo;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filterDesc, fcOutputTensorDesc, convDesc, inputTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &fcBwdDataAlgo));

    // Allocate device memory for fully connected layer backward data propagation workspace
    void* dFcBwdDataWorkspace;
    size_t fcBwdDataWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, fcOutputTensorDesc, convDesc, inputTensorDesc, fcBwdDataAlgo, &fcBwdDataWorkspaceSize));
    checkCudaErrors(cudaMalloc((void**)&dFcBwdDataWorkspace, fcBwdDataWorkspaceSize));

    // Create operation descriptor for fully connected layer backward filter propagation
    cudnnConvolutionBwdFilterAlgo_t fcBwdFilterAlgo;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, inputTensorDesc, fcOutputTensorDesc, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &fcBwdFilterAlgo));

    // Allocate device memory for fully connected layer backward filter propagation workspace
    void* dFcBwdFilterWorkspace;
    size_t fcBwdFilterWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputTensorDesc, fcOutputTensorDesc, convDesc, filterDesc, fcBwdFilterAlgo, &fcBwdFilterWorkspaceSize));
    checkCudaErrors(cudaMalloc((void**)&dFcBwdFilterWorkspace, fcBwdFilterWorkspaceSize));

    // Create softmax algorithm descriptor
    cudnnSoftmaxAlgorithm_t softmaxAlgorithm;
    checkCUDNN(cudnnGetSoftmaxAlgorithm(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &softmaxAlgorithm));

    // Create softmax mode descriptor
    cudnnSoftmaxMode_t softmaxMode;
    checkCUDNN(cudnnGetSoftmaxMode(cudnnHandle, &softmaxMode));

    // Create cross-entropy loss descriptor
    cudnnCrossEntropyLossDescriptor_t lossDesc;
    checkCUDNN(cudnnCreateCrossEntropyLossDescriptor(&lossDesc));

    // Train network
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        for (int batch = 0; batch < NUM_TRAINING_IMAGES / BATCH_SIZE; batch++)
        {
            // Copy input images and output labels to device memory
            checkCudaErrors(cudaMemcpyAsync(dInputImages, trainingData[batch * BATCH_SIZE].image, BATCH_SIZE * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            checkCudaErrors(cudaMemcpyAsync(dOutputLabels, trainingData[batch * BATCH_SIZE].label, BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Forward propagate through fully connected layer
            checkCUDNN(cudnnConvolutionForward(cudnnHandle, &LEARNING_RATE, inputTensorDesc, dInputImages, filterDesc, dFcWeights, convDesc, fcFwdAlgo, dFcFwdWorkspace, fcFwdWorkspaceSize, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput));

            // Add biases to fully connected layer output
            checkCublasErrors(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, &LEARNING_RATE, dFcBiases, NUM_CLASSES, &LEARNING_RATE, dFcOutput, NUM_CLASSES, dFcOutput, NUM_CLASSES));

            // Perform softmax on fully connected layer output
            checkCUDNN(cudnnSoftmaxForward(cudnnHandle, softmaxAlgorithm, softmaxMode, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput, &LEARNING_RATE, fcOutputTensorDesc, dOutputLogits));

            // Compute cross-entropy loss
            float loss;
            checkCUDNN(cudnnCrossEntropyLoss(cudnnHandle, lossDesc, fcOutputTensorDesc, dOutputLogits, outputTensorDesc, dOutputLabels, &loss));

            // Backward propagate through fully connected layer
            checkCUDNN(cudnnSoftmaxBackward(cudnnHandle, softmaxAlgorithm, softmaxMode, &LEARNING_RATE, fcOutputTensorDesc, dOutputLogits, outputTensorDesc, dOutputLabels, &LEARNING_RATE, fcOutputTensorDesc, dFcOutputGrad));
            checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &LEARNING_RATE, fcOutputTensorDesc, dFcOutputGrad, &LEARNING_RATE, outputTensorDesc, dFcBiasesGrad));
            checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &LEARNING_RATE, inputTensorDesc, dInputImages, fcOutputTensorDesc, dFcOutputGrad, convDesc, fcBwdFilterAlgo, dFcBwdFilterWorkspace, fcBwdFilterWorkspaceSize, &LEARNING_RATE, filterDesc, dFcWeightsGrad));
            checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &LEARNING_RATE, filterDesc, dFcWeights, fcOutputTensorDesc, dFcOutputGrad, convDesc, fcBwdDataAlgo, dFcBwdDataWorkspace, fcBwdDataWorkspaceSize, &LEARNING_RATE, inputTensorDesc, dInputImages));

            // Update fully connected layer weights and biases
            checkCublasErrors(cublasSaxpy(cublasHandle, NUM_CLASSES * IMAGE_SIZE, &LEARNING_RATE, dFcWeightsGrad, 1, dFcWeights, 1));
            checkCublasErrors(cublasSaxpy(cublasHandle, NUM_CLASSES, &LEARNING_RATE, dFcBiasesGrad, 1, dFcBiases, 1));
        }
    }

    // Copy fully connected layer weights and biases back to host memory
    checkCudaErrors(cudaMemcpy(hFcWeights, dFcWeights, NUM_CLASSES * IMAGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hFcBiases, dFcBiases, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

    // Test network
    for (int i = 0; i < numTestImages; i++)
    {
        // Copy input image to device memory
        checkCudaErrors(cudaMemcpyAsync(dInputImages, testData[i].image, IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Forward propagate through fully connected layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &LEARNING_RATE, inputTensorDesc, dInputImages, filterDesc, dFcWeights, convDesc, fcFwdAlgo, dFcFwdWorkspace, fcFwdWorkspaceSize, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput));

        // Add biases to fully connected layer output
        checkCublasErrors(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, &LEARNING_RATE, dFcBiases, NUM_CLASSES, &LEARNING_RATE, dFcOutput, NUM_CLASSES, dFcOutput, NUM_CLASSES));

        // Perform softmax on fully connected layer output
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, softmaxAlgorithm, softmaxMode, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput, &LEARNING_RATE, fcOutputTensorDesc, dOutputLogits));

        // Find index of maximum value in output logits
        int maxIndex = 0;
        float maxValue = dOutputLogits[0];
        for (int j = 1; j < NUM_CLASSES; j++)
        {
            if (dOutputLogits[j] > maxValue)
            {
                maxIndex = j;
                maxValue = dOutputLogits[j];
            }
        }

        // Check if maximum value is correct
        if (maxIndex != testData[i].label)
        {
            printf("Test image %d: Incorrect classification\n", i);
        }
    }

    // Destroy resources
    checkCudaErrors(cudaFree(dInputImages));
    checkCudaErrors(cudaFree(dOutputLabels));
    checkCudaErrors(cudaFree(dFcOutput));
    checkCudaErrors(cudaFree(dFcOutputGrad));
    checkCudaErrors(cudaFree(dFcWeights));
    checkCudaErrors(cudaFree(dFcWeightsGrad));
    checkCudaErrors(cudaFree(dFcBiases));
    checkCudaErrors(cudaFree(dFcBiasesGrad));
    checkCudaErrors(cudaFree(dFcFwdWorkspace));
    checkCudaErrors(cudaFree(dFcBwdDataWorkspace));
    checkCudaErrors(cudaFree(dFcBwdFilterWorkspace));
    checkCudaErrors(cudaFree(dOutputLogits));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCublasErrors(cublasDestroy(cublasHandle));
    checkCUDNN(cudnnDestroy(c






