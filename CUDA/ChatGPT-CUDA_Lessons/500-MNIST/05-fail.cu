#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdint.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE IMAGE_WIDTH * IMAGE_HEIGHT
#define NUM_CLASSES 10
#define NUM_CHANNELS 1
#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f

const int HIDDEN_SIZE = 128;

struct TrainingData
{
    float image[IMAGE_SIZE];
    float label[NUM_CLASSES];
};

struct TestData
{
    float image[IMAGE_SIZE];
    float label[NUM_CLASSES];
};

int32_t readInt(std::ifstream& stream)
{
    int32_t value;
    stream.read((char*)&value, 4);
    return __builtin_bswap32(value);
}

// Load MNIST training data
int loadTrainingData(TrainingData* trainingData)
{
    // Open the training images file
    std::ifstream file("../../../zData/MNIST/train-images-idx3-ubyte", std::ios::binary);

    // Read the magic number and number of images
    int32_t magic = readInt(file);
    int32_t numImages = readInt(file);

    // Read the image data
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            uint8_t pixel;
            file.read((char*)&pixel, 1);
            hTrainingData[i * IMAGE_SIZE + j] = pixel / 255.0f;
    }
    
    // Open training labels file
    std::ifstream file("../../../zData/MNIST/train-labels-idx1-ubyte", std::ios::binary);
    if (!file.is_open())
    {
        std::printf("Error opening file\n");
        return 1;
    }

    // Read header
    int32_t magicNumber;
    int32_t numLabels;
    file.read((char*)&magicNumber, 4);
    file.read((char*)&numLabels, 4);
    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    // Read data
    for (int i = 0; i < numLabels; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);
        trainingData[i].label[(int)label] = 1.0f;
    }

    file.close();
    
    // Read the image data
    float* hTrainingData = new float[numImages * IMAGE_SIZE];
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            uint8_t pixel;
            file.read((char*)&pixel, 1);
            hTrainingData[i * IMAGE_SIZE + j] = pixel / 255.0f;
        }
    }
    
    // Close the file
    file.close();

    // Open training labels file
    std::ifstream file("../../../zData/MNIST/train-labels-idx1-ubyte", std::ios::binary);
    if (!file.is_open())
    {
        std::printf("Error opening file\n");
        return 1;
    }

    // Read header
    int32_t magicNumber = readInt(file);
    int32_t numLabels = readInt(file);

    // Read data
    for (int i = 0; i < numLabels; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);
        trainingData[i].label[(int)label] = 1.0f;
    }

    file.close();

    // Copy data to device
    float* dTrainingData;
    cudaMalloc((void**)&dTrainingData, numImages * IMAGE_
    
    // Allocate device memory for training data
    checkCudaErrors(cudaMalloc((void**)&dTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float)));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(dTrainingData, hTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Free host memory
    delete[] hTrainingData;

    // Create device memory for labels
    float* dLabels;
    checkCudaErrors(cudaMalloc((void**)&dLabels, NUM_IMAGES * NUM_CLASSES * sizeof(float)));

    // Copy labels to device
    checkCudaErrors(cudaMemcpy(dLabels, trainingData, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDNN tensors
    cudnnTensorDescriptor_t inputTensor;
    cudnnTensorDescriptor_t outputTensor;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input
    
        cudaMalloc((void**)&dTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float));
    cudaMemcpy(dTrainingData, hTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Copy labels to device
    float* dTrainingLabels;
    cudaMalloc((void**)&dTrainingLabels, NUM_IMAGES * NUM_CLASSES * sizeof(float));
    cudaMemcpy(dTrainingLabels, trainingData, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // Create input and output tensors
    cudnnTensorDescriptor_t inputTensorDesc;
    cudnnTensorDescriptor_t outputTensorDesc;
    cudnnCreateTensorDescriptor(&inputTensorDesc);
    cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
    cudnnCreateTensorDescriptor(&outputTensorDesc);
    cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1);

    // Create filter and bias tensors
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_CLASSES, NUM_CH

// Allocate device memory for labels
float* dLabels;
cudaMalloc((void**)&dLabels, NUM_TRAINING_IMAGES * NUM_CLASSES * sizeof(float));

// Copy labels to device
cudaMemcpy(dLabels, hLabels, NUM_TRAINING_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

// Define layer sizes
const int inputSize = IMAGE_SIZE;
const int hiddenSize = 100;
const int outputSize = NUM_CLASSES;

// Allocate device memory for weights and biases
float* dWeights1, dBiases1, dWeights2, dBiases2;
cudaMalloc((void)&dWeights1, inputSize * hiddenSize * sizeof(float));
cudaMalloc((void*)&dBiases1, hiddenSize * sizeof(float));
cudaMalloc((void**)&dWeights2, hiddenSize * outputSize * sizeof(float));
cudaMalloc((void**)&dBiases2, outputSize * sizeof(float));

// Initialize weights and biases with random values
initializeWeights(dWeights1, inputSize * hiddenSize);
initializeWeights(dBiases1, hiddenSize);
initializeWeights(dWeights2, hiddenSize * outputSize);
initializeWeights(dBiases2, outputSize);

// Allocate device memory for layer inputs and outputs
float* dInput, dHidden, dOutput;
cudaMalloc((void)&dInput, BATCH_SIZE * inputSize * sizeof(float));
cudaMalloc((void**)&dHidden, BATCH_SIZE * hiddenSize * sizeof(float));
cudaMalloc((void**)&dOutput, BATCH_SIZE * outputSize * sizeof(float));

// Allocate memory on device
cudaMalloc((void**)&dTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float));
cudaMalloc((void**)&dTrainingLabels, NUM_IMAGES * NUM_CLASSES * sizeof(float));

// Copy data to device
cudaMemcpy(dTrainingData, hTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(dTrainingLabels, hTrainingLabels, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

// Free host memory
delete[] hTrainingData;
delete[] hTrainingLabels;

// Set up input tensor descriptor
cudnnTensorDescriptor_t inputTensorDesc;
cudnnCreateTensorDescriptor(&inputTensorDesc);
cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT);

// Set up output tensor descriptor
cudnnTensorDescriptor_t outputTensorDesc;
cudnnCreateTensorDescriptor(&outputTensorDesc);
cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1);

// Set up convolution layer
cudnnConvolutionDescriptor_t convDesc;
cudnnCreateConvolutionDescriptor(&convDesc);
cudnnSetConvolution2dDescriptor(convDesc,
    
// Copy data to device
float* dTrainingData;
cudaMalloc((void**)&dTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float));
cudaMemcpy(dTrainingData, hTrainingData, NUM_IMAGES * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

// Create device arrays for labels
float* dLabels;
cudaMalloc((void**)&dLabels, NUM_IMAGES * NUM_CLASSES * sizeof(float));
cudaMemcpy(dLabels, trainingData, NUM_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

// Create device arrays for weights and biases
float* dWeights1;
float* dBias1;
float* dWeights2;
float* dBias2;
cudaMalloc((void**)&dWeights1, IMAGE_SIZE * HIDDEN_SIZE * sizeof(float));
cudaMalloc((void**)&dBias1, HIDDEN_SIZE * sizeof(float));
cudaMalloc((void**)&dWeights2, HIDDEN_SIZE * NUM_CLASSES * sizeof(float));
cudaMalloc((void**)&dBias2, NUM_CLASSES * sizeof(float));

// Initialize weights and biases to random values
srand(time(NULL));
for (int i = 0; i < IMAGE_SIZE * HIDDEN_SIZE; i++)
{
dWeights1[i] = (float)rand() / RAND_MAX;
}
for (int i = 0; i < HIDDEN_SIZE; i++)
{
dBias1[i] = (float)rand() / RAND_MAX;
}
for

// Allocate device memory for labels
float* dTrainingLabels;
cudaMalloc((void**)&dTrainingLabels, NUM_TRAINING_IMAGES * NUM_CLASSES * sizeof(float));

// Copy labels to device
cudaMemcpy(dTrainingLabels, hTrainingLabels, NUM_TRAINING_IMAGES * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

// Create cudnnTensorDescriptor_t for data
cudnnTensorDescriptor_t dataDesc;
cudnnCreateTensorDescriptor(&dataDesc);
cudnnSetTensor4dDescriptor(dataDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);

// Create cudnnTensorDescriptor_t for labels
cudnnTensorDescriptor_t labelDesc;
cudnnCreateTensorDescriptor(&labelDesc);
cudnnSetTensor4dDescriptor(labelDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1);

// Create cudnnFilterDescriptor_t for weights
cudnnFilterDescriptor_t weightDesc;
cudnnCreateFilterDescriptor(&weightDesc);
cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_HIDDEN, NUM_CHANNELS, KERNEL_WIDTH, KERNEL_HEIGHT

// NOTE - At this point I give up on this one too. This sort of thing happens a lot. I used to be able to argue it in to finally working. But on Dec 28 2022 they seem to be limiting hourly questions.

#include <cudnn.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE IMAGE_WIDTH * IMAGE_HEIGHT
#define NUM_CLASSES 10
#define NUM_CHANNELS 1
#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f

struct TrainingData
{
float image[IMAGE_SIZE];
float label[NUM_CLASSES];
};

struct TestData
{
float image[IMAGE_SIZE];
float label[NUM_CLASSES];
};

int32_t readInt(std::ifstream& stream)
{
int32_t value;
stream.read((char*)&value, 4);
return __builtin_bswap32(value);
}

// Load MNIST training data
int loadTrainingData(TrainingData* trainingData)
{
// Open the training images file
std::ifstream file("../../../zData/MNIST/train-images-idx3-ubyte", std::ios::binary);

// Read the magic number and number of images
int32_t magic = readInt(file);
int32_t numImages = readInt(file);

// Read the image data
float* hTrainingData = new float[numImages * IMAGE_SIZE];
for (int i = 0; i < numImages; i++)
{
    for (

