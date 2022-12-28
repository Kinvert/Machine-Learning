#include <cudnn.h>
#include <cublas_v2.h>
//#include <cstdio.h>
#include <stdio.h>
//#include <cstdlib.h>
#include <stdlib.h>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include "cuda_error.h"
#include <cuda_runtime.h>
//#include "cudnn_error.h"
//#include "cublas_error.h"

#define IMAGE_SIZE 28 * 28
#define NUM_CLASSES 10
#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define BATCH_SIZE 256
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.01

struct MNISTData
{
    float image[IMAGE_SIZE];
    int label;
};

void checkCudaErrors(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCUDNN(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasErrors(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS Error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

void readMNISTData(std::string fileName, int numImages, MNISTData* data)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    int magicNumber, imageSize;
    file.read((char*)&magicNumber, 4);
    magicNumber = ((magicNumber & 0xff) << 24) | ((magicNumber & 0xff00) << 8) | ((magicNumber & 0xff0000) >> 8) | ((magicNumber & 0xff000000) >> 24);
    file.read((char*)&numImages, 4);
    numImages = ((numImages & 0xff) << 24) | ((numImages & 0xff00) << 8) | ((numImages & 0xff0000) >> 8) | ((numImages & 0xff000000) >> 24);
    file.read((char*)&imageSize, 4);
    imageSize = ((imageSize & 0xff) << 24) | ((imageSize & 0xff00) << 8) | ((imageSize & 0xff0000) >> 8) | ((imageSize & 0xff000000) >> 24);
    for (int i = 0; i < numImages; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);
        data[i].label = label;
        file.read((char*)&data[i].image, imageSize);
    }
}

void readMNISTLabels(std::string fileName, int numLabels, MNISTData* data)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    int magicNumber;
    file.read((char*)&magicNumber, 4);
    magicNumber = ((magicNumber & 0xff) << 24) | ((magicNumber & 0xff00) << 8) | ((magicNumber & 0xff0000) >> 8) | ((magicNumber & 0xff000000) >> 24);
    file.read((char*)&numLabels, 4);
    numLabels = ((numLabels & 0xff) << 24) | ((numLabels & 0xff00) << 8) | ((numLabels & 0xff0000) >> 8) | ((numLabels & 0xff000000) >> 24);
    for (int i = 0; i < numLabels; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);
        data[i].label = label;
    }
}

int main()
{
    // Read training data
    MNISTData* trainingData = new MNISTData[NUM_TRAINING_IMAGES];
    readMNISTData("../../../zData/MNIST/train-images-idx3-ubyte.gz", NUM_TRAINING_IMAGES, trainingData);
    readMNISTLabels("../../../zData/MNIST/train-labels-idx1-ubyte.gz", NUM_TRAINING_IMAGES, trainingData);

    // Read test data
    MNISTData* testData = new MNISTData[NUM_TEST_IMAGES];
    readMNISTData("../../../zData/MNIST/t10k-images-idx3-ubyte.gz", NUM_TEST_IMAGES, testData);
    readMNISTLabels("../../../zData/MNIST/t10k-labels-idx1-ubyte.gz", NUM_TEST_IMAGES, testData);

    // Initialize CUDA and CUDNN
    checkCudaErrors(cudaSetDevice(0));
    checkCudnnErrors(cudnnCreate(&cudnnHandle));

    // Create input tensor descriptor
    checkCudnnErrors(cudnnCreateTensorDescriptor(&inputTensorDesc));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, 1, IMAGE_SIZE, 1));

    // Create output tensor descriptor
    checkCudnnErrors(cudnnCreateTensorDescriptor(&outputTensorDesc));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, 1, NUM_CLASSES, 1));

    // Create fully connected layer filter descriptor
    checkCudnnErrors(cudnnCreateFilterDescriptor(&filterDesc));
    checkCudnnErrors(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_CLASSES, 1, IMAGE_SIZE, 1));

    // Create fully connected layer output tensor descriptor
    checkCudnnErrors(cudnnCreateTensorDescriptor(&fcOutputTensorDesc));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(fcOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1));

    // Create convolution descriptor
    checkCudnnErrors(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Create fully connected layer forward convolution algorithm
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputTensorDesc, filterDesc, convDesc, fcOutputTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fcFwdAlgo));
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputTensorDesc, filterDesc, convDesc, fcOutputTensorDesc, fcFwdAlgo, &fcFwdWorkspaceSize));

    // Create fully connected layer backward data convolution algorithm
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filterDesc, fcOutputTensorDesc, convDesc, inputTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &fcBwdDataAlgo));
    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, fcOutputTensorDesc, convDesc, inputTensorDesc, fcBwdDataAlgo, &fcBwdDataWorkspaceSize));

    // Create fully connected layer backward filter convolution algorithm
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, inputTensorDesc, fcOutputTensorDesc, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &fcBwdFilterAlgo));
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputTensorDesc, fcOutputTensorDesc, convDesc, filterDesc, fcBwdFilterAlgo, &fcBwdFilterWorkspaceSize));

    // Create softmax algorithm
    checkCudnnErrors(cudnnGetSoftmaxForwardAlgorithm(cudnnHandle, softmaxMode, outputTensorDesc, &softmaxAlgorithm));

    // Allocate device memory
    float* dInputImages;
    checkCudaErrors(cudaMalloc((void**)&dInputImages, IMAGE_SIZE * BATCH_SIZE * sizeof(float)));
    float* dOutputLabels;
    checkCudaErrors(cudaMalloc((void**)&dOutputLabels, NUM_CLASSES * BATCH_SIZE * sizeof(float)));
    float* dFcWeights;
    checkCudaErrors(cudaMalloc((void**)&dFcWeights, NUM_CLASSES * IMAGE_SIZE * sizeof(float)));
    float* dFcWeightsGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcWeightsGrad, NUM_CLASSES * IMAGE_SIZE * sizeof(float)));
    float* dFcBiases;
    checkCudaErrors(cudaMalloc((void**)&dFcBiases, NUM_CLASSES * sizeof(float)));
    float* dFcBiasesGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcBiasesGrad, NUM_CLASSES * sizeof(float)));
    float* dFcOutput;
    checkCudaErrors(cudaMalloc((void**)&dFcOutput, NUM_CLASSES * BATCH_SIZE * sizeof(float)));
    float* dFcOutputGrad;
    checkCudaErrors(cudaMalloc((void**)&dFcOutputGrad, NUM_CLASSES * BATCH_SIZE * sizeof(float)));
    void* dFcFwdWorkspace;
    checkCudaErrors(cudaMalloc(&dFcFwdWorkspace, fcFwdWorkspaceSize));
    void* dFcBwdDataWorkspace;
    checkCudaErrors(cudaMalloc(&dFcBwdDataWorkspace, fcBwdDataWorkspaceSize));
    void* dFcBwdFilterWorkspace;
    checkCudaErrors(cudaMalloc(&dFcBwdFilterWorkspace, fcBwdFilterWorkspaceSize));
    float* dSoftmaxOutput;
    checkCudaErrors(cudaMalloc((void**)&dSoftmaxOutput, NUM_CLASSES * BATCH_SIZE * sizeof(float)));
    float* dSoftmaxOutputGrad;
    checkCudaErrors(cudaMalloc((void**)&dSoftmaxOutputGrad, NUM_CLASSES * BATCH_SIZE * sizeof(float)));

    // Initialize fully connected layer weights and biases
    std::srand(std::time(0));
    for (int i = 0; i < NUM_CLASSES * IMAGE_SIZE; i++)
    {
        dFcWeights[i] = ((float)std::rand()) / RAND_MAX;
    }
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        dFcBiases[i] = ((float)std::rand()) / RAND_MAX;
    }

    // Initialize CUBLAS
    checkCublasErrors(cublasCreate(&cublasHandle));

    // Train model
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        for (int i = 0; i < NUM_TRAINING_IMAGES / BATCH_SIZE; i++)
        {
            // Copy input images and output labels to device
            checkCudaErrors(cudaMemcpy(dInputImages, trainingData[i * BATCH_SIZE].image, IMAGE_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dOutputLabels, trainingData[i * BATCH_SIZE].label, NUM_CLASSES * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Forward pass
            checkCublasErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, IMAGE_SIZE, &LEARNING_RATE, dFcWeights, NUM_CLASSES, dInputImages, IMAGE_SIZE, &LEARNING_RATE, dFcOutput, NUM_CLASSES));
            checkCublasErrors(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, &LEARNING_RATE, dFcBiases, 1, &LEARNING_RATE, dFcOutput, NUM_CLASSES, dFcOutput, NUM_CLASSES));
            checkCudnnErrors(cudnnSoftmaxForward(cudnnHandle, softmaxAlgorithm, softmaxMode, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput, outputTensorDesc, dSoftmaxOutput));

            // Backward pass
            checkCudnnErrors(cudnnSoftmaxBackward(cudnnHandle, softmaxAlgorithm, softmaxMode, outputTensorDesc, dSoftmaxOutput, outputTensorDesc, dSoftmaxOutputGrad, fcOutputTensorDesc, dFcOutput, fcOutputTensorDesc, dFcOutputGrad));
            checkCublasErrors(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, &LEARNING_RATE, dFcOutputGrad, NUM_CLASSES, &LEARNING_RATE, dFcBiasesGrad, 1, dFcBiasesGrad, 1));
            checkCublasErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, BATCH_SIZE, NUM_CLASSES, NUM_CLASSES, &LEARNING_RATE, dFcOutputGrad, NUM_CLASSES, dFcWeights, NUM_CLASSES, &LEARNING_RATE, dFcBwdFilterWorkspace, BATCH_SIZE));
            checkCudnnErrors(cudnnConvolutionBackwardFilter(cudnnHandle, &LEARNING_RATE, inputTensorDesc, dInputImages, fcOutputTensorDesc, dFcOutputGrad, convDesc, fcBwdFilterAlgo, dFcBwdFilterWorkspace, fcBwdFilterWorkspaceSize, &LEARNING_RATE, filterDesc, dFcWeightsGrad));
            checkCublasErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, &LEARNING_RATE, dFcWeights, NUM_CLASSES, dFcOutputGrad, NUM_CLASSES, &LEARNING_RATE, dFcBwdDataWorkspace, IMAGE_SIZE));
            checkCudnnErrors(cudnnConvolutionBackwardData(cudnnHandle, &LEARNING_RATE, filterDesc, dFcWeights, fcOutputTensorDesc, dFcOutputGrad, convDesc, fcBwdDataAlgo, dFcBwdDataWorkspace, fcBwdDataWorkspaceSize, &LEARNING_RATE, inputTensorDesc, dInputImages));

            // Update weights and biases
            checkCublasErrors(cublasSaxpy(cublasHandle, NUM_CLASSES, &LEARNING_RATE, dFcBiasesGrad, 1, dFcBiases, 1));
            checkCublasErrors(cublasSaxpy(cublasHandle, NUM_CLASSES * IMAGE_SIZE, &LEARNING_RATE, dFcWeightsGrad, 1, dFcWeights, 1));
        }
    }

    // Test model
    int numCorrect = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        // Copy input image to device
        checkCudaErrors(cudaMemcpy(dInputImages, testData[i].image, IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward pass
        checkCublasErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, NUM_CLASSES, 1, IMAGE_SIZE, &LEARNING_RATE, dFcWeights, NUM_CLASSES, dInputImages, IMAGE_SIZE, &LEARNING_RATE, dFcOutput, NUM_CLASSES));
        checkCublasErrors(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, 1, &LEARNING_RATE, dFcBiases, 1, &LEARNING_RATE, dFcOutput, NUM_CLASSES, dFcOutput, NUM_CLASSES));
        checkCudnnErrors(cudnnSoftmaxForward(cudnnHandle, softmaxAlgorithm, softmaxMode, &LEARNING_RATE, fcOutputTensorDesc, dFcOutput, outputTensorDesc, dSoftmaxOutput));

        // Get index of highest probability
        float maxProb = 0.0f;
        int maxIndex = 0;
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            float prob;
            checkCudaErrors(cudaMemcpy(&prob, dSoftmaxOutput + j, sizeof(float), cudaMemcpyDeviceToHost));
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = j;
            }
        }

        // Check if index is correct
        if (testData[i].label[maxIndex] == 1.0f)
        {
            numCorrect++;
        }
    }

    // Print accuracy
    std::cout << "Accuracy: " << ((float)numCorrect / NUM_TEST_IMAGES) * 100 << "%" << std::endl;

    // Cleanup
    checkCudnnErrors(cudnnDestroy(cudnnHandle));
    checkCublasErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(cudaFree(dInputImages));
    checkCudaErrors(cudaFree(dOutputLabels));
    checkCudaErrors(cudaFree(dFcWeights));
    checkCudaErrors(cudaFree(dFcWeightsGrad));
    checkCudaErrors(cudaFree(dFcBiases));
    checkCudaErrors(cudaFree(dFcBiasesGrad));
    checkCudaErrors(cudaFree(dFcOutput));
    checkCudaErrors(cudaFree(dFcOutputGrad));
    checkCudaErrors(cudaFree(dFcFwdWorkspace));
    checkCudaErrors(cudaFree(dFcBwdDataWorkspace));
    checkCudaErrors(cudaFree(dFcBwdFilterWorkspace));
    checkCudaErrors(cudaFree(dSoftmaxOutput));
    checkCudaErrors(cudaFree(dSoftmaxOutputGrad));

    return 0;
}








