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
            file.read

