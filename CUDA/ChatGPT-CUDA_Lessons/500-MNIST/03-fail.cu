#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

cudnnSoftmaxMode_t softmaxMode = CUDNN_SOFTMAX_ACCURATE;
cudnnSoftmaxAlgorithm_t softmaxAlgorithm = CUDNN_SOFTMAX_ALGORITHM_ACCURATE;


// Load MNIST training data
int loadTrainingData(TrainingData* trainingData)
{
    // Open the training images file
    std::ifstream file("../../../zData/MNIST/train-images-idx3-ubyte", std::ios::binary);

    // Read the magic number and number of images
    int32_t magic = readInt(file);
    int32_t numImages = readInt(file);

    // Read the image data
    float* hTrainingData = new float[NUM_IMAGES * IMAGE_SIZE];
    for (int i = 0; i < NUM_IMAGES; i++)
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
    file.read((char*)&magicNumber, 4);
    file.read((char*)&numImages, 4);
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);

    // Read data
    for (int i = 0; i < numImages; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);
        trainingData[i].label[(int)label] = 1.0f;
    }

    file.close();

    return 0;
}

// Load MNIST test data
int loadTestData(TestData* testData)
{
    // Open test images file
    std::FILE* file = std::fopen("../../../zData/MNIST/t10k-images-idx3-ubyte", "rb");
    if (file == NULL)
    {
        std::printf("Error opening file\n");
        return 1;
    }

    // Read header
    int magicNumber;
    int numImages;
    int imageRows;
    int imageCols;
    std::fread((char*)&magicNumber, 4, 1, file);
    std::fread((char*)&numImages, 4, 1, file);
    std::fread((char*)&imageRows, 4, 1, file);
    std::fread((char*)&imageCols, 4, 1, file);
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    imageRows = __builtin_bswap32(imageRows);
    imageCols = __builtin_bswap32(imageCols);

    // Read data
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < IMAGE_SIZE; j++)
        {
            unsigned char pixel;
            std::fread((char*)&pixel, 1, 1, file);
            testData[i].image[j] = (float)pixel / 255.0f;
        }
    }

    std::fclose(file);

    // Open test labels file
    file = std::fopen("../../../zData/MNIST/t10k-labels-idx1-ubyte", "rb");
    if (file == NULL)
    {
        std::printf("Error opening file\n");
        return 1;
    }

    // Read header
    std::fread((char*)&magicNumber, 4, 1, file);
    std::fread((char*)&numImages, 4, 1, file);
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);

    // Read data
    for (int i = 0; i < numImages; i++)
    {
        unsigned char label;
        std::fread((char*)&label, 1, 1, file);
        testData[i].label[(int)label] = 1.0f;
    }

    std::fclose(file);

    return 0;
}

// Check for CUDA errors
void checkCudaErrors(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::printf("CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

// Check for CUDNN errors
void checkCUDNN(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::printf("CUDNN error: %s\n", cudnnGetErrorString(status));
        std::exit(1);
    }
}

int main(int argc, char** argv)
{
    // Seed random number generator
    std::srand((unsigned int)time(NULL));

    // Initialize CUDA
    checkCudaErrors(cudaSetDevice(0));

    // Allocate host memory
    TrainingData* hTrainingData = (TrainingData*)std::malloc(sizeof(TrainingData) * 60000);
    TestData* hTestData = (TestData*)std::malloc(sizeof(TestData) * 10000);
    float* hFcWeights = (float*)std::malloc(sizeof(float) * IMAGE_SIZE * NUM_CLASSES);
    float* hFcBiases = (float*)std::malloc(sizeof(float) * NUM_CLASSES);
    float* hFcOutput = (float*)std::malloc(sizeof(float) * BATCH_SIZE * NUM_CLASSES);
    float* hFcOutputGrad = (float*)std::malloc(sizeof(float) * BATCH_SIZE * NUM_CLASSES);
    float* hSoftmaxOutput = (float*)std::malloc(sizeof(float) * BATCH_SIZE * NUM_CLASSES);
    float* hSoftmaxOutputGrad = (float*)std::malloc(sizeof(float) * BATCH_SIZE * NUM_CLASSES);
    float* hSoftmaxLoss = (float*)std::malloc(sizeof(float) * BATCH_SIZE);

    // Allocate device memory
    TrainingData* dTrainingData;
    TestData* dTestData;
    float* dFcWeights;
    float* dFcBiases;
    float* dFcOutput;
    float* dFcOutputGrad;
    float* dSoftmaxOutput;
    float* dSoftmaxOutputGrad;
    float* dSoftmaxLoss;
    checkCudaErrors(cudaMalloc((void**)&dTrainingData, sizeof(TrainingData) * 60000));
    checkCudaErrors(cudaMalloc((void**)&dTestData, sizeof(TestData) * 10000));
    checkCudaErrors(cudaMalloc((void**)&dFcWeights, sizeof(float) * IMAGE_SIZE * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dFcBiases, sizeof(float) * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dFcOutput, sizeof(float) * BATCH_SIZE * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dFcOutputGrad, sizeof(float) * BATCH_SIZE * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dSoftmaxOutput, sizeof(float) * BATCH_SIZE * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dSoftmaxOutputGrad, sizeof(float) * BATCH_SIZE * NUM_CLASSES));
    checkCudaErrors(cudaMalloc((void**)&dSoftmaxLoss, sizeof(float) * BATCH_SIZE));

    // Initialize CUDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    // Create CUDNN tensors
    cudnnTensorDescriptor_t inputTensor;
    cudnnTensorDescriptor_t outputTensor;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1));

    // Create CUDNN filter descriptor
    cudnnFilterDescriptor_t filterDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_CLASSES, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH));

    // Create CUDNN convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Create CUDNN convolution operation
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, inputTensor, filterDesc, convDesc, outputTensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convFwdAlgo));
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filterDesc, outputTensor, convDesc, inputTensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convBwdDataAlgo));
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, inputTensor, outputTensor, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convBwdFilterAlgo));
    size_t convFwdWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputTensor, filterDesc, convDesc, outputTensor, convFwdAlgo, &convFwdWorkspaceSize));
    size_t convBwdDataWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filterDesc, outputTensor, convDesc, inputTensor, convBwdDataAlgo, &convBwdDataWorkspaceSize));
    size_t convBwdFilterWorkspaceSize;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, inputTensor, outputTensor, convDesc, filterDesc, convBwdFilterAlgo, &convBwdFilterWorkspaceSize));

    // Allocate device memory for convolution workspace
    void* dConvFwdWorkspace;
    void* dConvBwdDataWorkspace;
    void* dConvBwdFilterWorkspace;
    checkCudaErrors(cudaMalloc((void**)&dConvFwdWorkspace, convFwdWorkspaceSize));
    checkCudaErrors(cudaMalloc((void**)&dConvBwdDataWorkspace, convBwdDataWorkspaceSize));
    checkCudaErrors(cudaMalloc((void**)&dConvBwdFilterWorkspace, convBwdFilterWorkspaceSize));

    // Train network
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        for (int batch = 0; batch < NUM_BATCHES; batch++)
        {
            // Select training data for batch
            int offset = batch * BATCH_SIZE;
            TrainingData* batchTrainingData = dTrainingData + offset;

            // Forward pass
            checkCudaErrors(cudaMemcpy(dFcInput, batchTrainingData, sizeof(TrainingData) * BATCH_SIZE, cudaMemcpyDeviceToDevice));
            checkCUDNN(cudnnConvolutionForward(cudnn, (void*)&alpha, inputTensor, dFcInput, filterDesc, dFcWeights, convDesc, convFwdAlgo, dConvFwdWorkspace, (void*)&beta, outputTensor, dFcOutput));
            checkCUDNN(cudnnAddTensor(cudnn, (void*)&alpha, outputTensor, dFcBiases, (void*)&alpha, outputTensor, dFcOutput));
            checkCUDNN(cudnnSoftmaxForward(cudnn, softmaxAlgorithm, softmaxMode, (void*)&alpha, outputTensor, dFcOutput, (void*)&beta, outputTensor, dSoftmaxOutput));

            // Compute loss
            checkCUDNN(cudnnSoftmaxCrossEntropyLoss(cudnn, (void*)&alpha, outputTensor, batchTrainingData->label, (void*)&beta, outputTensor, dSoftmaxLoss));

            // Backward pass
            checkCUDNN(cudnnSoftmaxBackward(cudnn, softmaxAlgorithm, softmaxMode, (void*)&alpha, outputTensor, dSoftmaxOutput, outputTensor, dSoftmaxOutputGrad, (void*)&beta, outputTensor, dFcOutputGrad));
            checkCUDNN(cudnnConvolutionBackwardFilter(cudnn, (void*)&alpha, inputTensor, dFcInput, outputTensor, dFcOutputGrad, convDesc, convBwdFilterAlgo, dConvBwdFilterWorkspace, (void*)&beta, filterDesc, dFcWeights));
            checkCUDNN(cudnnConvolutionBackwardData(cudnn, (void*)&alpha, filterDesc, dFcWeights, outputTensor, dFcOutputGrad, convDesc, convBwdDataAlgo, dConvBwdDataWorkspace, (void*)&beta, inputTensor, dFcInput));

            // Update weights and biases
            float learningRate = 0.001f;
            checkCudaErrors(cublasSaxpy(cublas, NUM_CLASSES, (void*)&learningRate, dFcOutputGrad, 1, dFcBiases, 1));
            checkCudaErrors(cublasSgemv(cublas, CUBLAS_OP_T, IMAGE_SIZE, NUM_CLASSES, (void*)&learningRate, dFcInput, IMAGE_SIZE, dFcOutputGrad, 1, (void*)&alpha, dFcWeights, 1));
        }
    }

    // Test network
    int numCorrect = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        // Forward pass
        checkCudaErrors(cudaMemcpy(dFcInput, dTestData + i, sizeof(TestData), cudaMemcpyDeviceToDevice));
        checkCUDNN(cudnnConvolutionForward(cudnn, (void*)&alpha, inputTensor, dFcInput, filterDesc, dFcWeights, convDesc, convFwdAlgo, dConvFwdWorkspace, (void*)&beta, outputTensor, dFcOutput));
        checkCUDNN(cudnnAddTensor(cudnn, (void*)&alpha, outputTensor, dFcBiases, (void*)&alpha, outputTensor, dFcOutput));
        checkCUDNN(cudnnSoftmaxForward(cudnn, softmaxAlgorithm, softmaxMode, (void*)&alpha, outputTensor, dFcOutput, (void*)&beta, outputTensor, dSoftmaxOutput));

        // Select class with maximum probability
        int classIdx;
        float classProb;
        checkCudaErrors(cudaMemcpy(&classProb, dSoftmaxOutput, sizeof(float), cudaMemcpyDeviceToHost));
        classIdx = 0;
        for (int j = 1; j < NUM_CLASSES; j++)
        {
            float prob;
            checkCudaErrors(cudaMemcpy(&prob, dSoftmaxOutput + j, sizeof(float), cudaMemcpyDeviceToHost));
            if (prob > classProb)
            {
                classProb = prob;
                classIdx = j;
            }
        }

        // Update accuracy
        if (classIdx == hTestLabels[i])
        {
            numCorrect++;
        }
    }
    float accuracy = (float)numCorrect / NUM_TEST_IMAGES;
    printf("Accuracy: %f\n", accuracy);

    // Clean up
    checkCudaErrors(cudaFree(dTrainingData));
    checkCudaErrors(cudaFree(dTestData));
    checkCudaErrors(cudaFree(dFcWeights));
    checkCudaErrors(cudaFree(dFcBiases));
    checkCudaErrors(cudaFree(dFcInput));
    checkCudaErrors(cudaFree(dFcOutput));
    checkCudaErrors(cudaFree(dFcOutputGrad));
    checkCudaErrors(cudaFree(dSoftmaxOutput));
    checkCudaErrors(cudaFree(dSoftmaxLoss));
    checkCudaErrors(cudaFree(dSoftmaxOutputGrad));
    checkCudaErrors(cudaFree(dConvFwdWorkspace));
    checkCudaErrors(cudaFree(dConvBwdDataWorkspace));
    checkCudaErrors(cudaFree(dConvBwdFilterWorkspace));
    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCudaErrors(cublasDestroy(cublas));
    checkCUDNN(cudnnDestroy(cudnn));

    delete[] hTrainingData;
    delete[] hTestData;
    delete[] hTestLabels;
    delete[] hFcWeights;
    delete[] hFcBiases;
    
    return 0;
}
   



