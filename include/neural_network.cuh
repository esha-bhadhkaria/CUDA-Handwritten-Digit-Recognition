#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

void matrixMultiply(const float* A, const float* B, float* C, int M, int N, int K);
void relu(const float* input, float* output, int size);

void forwardPass(
    const float* input,
    const float* W1,
    const float* b1,
    const float* W2,
    const float* b2,
    float* hiddenOutput,
    float* probabilities,
    int batchSize
);

void calculateOutputGradient(
    const float* probabilities,
    const int* labels,
    float* gradients,
    int batchSize
);

void updateParameters(float* parameters, const float* gradients, float learningRate, int size);

void calculateGradients(
    const float* input,
    const float* hidden,
    const float* W2,
    const float* outputGradients,
    float* dW1,
    float* db1,
    float* dW2,
    float* db2,
    int batchSize
);

float calculateLoss(const float* probabilities, const int* labels, int batchSize);

#endif
