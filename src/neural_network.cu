#include <cuda_runtime.h>
#include "neural_network.cuh"

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void reluKernel(const float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = (input[index] > 0.0f) ? input[index] : 0.0f;
    }
}

void matrixMultiply(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void relu(const float* input, float* output, int size) {

    float *d_input, *d_output;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    reluKernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

__global__ void biasReluKernel(
    float* hidden,
    const float* bias,
    int batchSize,
    int size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * size;

    if (index < total) {
        int neuron = index % size;
        float value = hidden[index] + bias[neuron];
        hidden[index] = (value > 0.0f) ? value : 0.0f;
    }
}

__global__ void biasKernel(
    float* output,
    const float* bias,
    int batchSize,
    int size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * size;

    if (index < total) {
        int neuron = index % size;
        output[index] += bias[neuron];
    }
}

__global__ void softmaxKernel(
    const float* input,
    float* output,
    int batchSize,
    int size
) {
    int row = blockIdx.x;

    if (row < batchSize) {
        float maxValue = input[row * size];

        for (int i = 1; i < size; i++) {
            float value = input[row * size + i];

            if (value > maxValue) {
                maxValue = value;
            }
        }

        float sum = 0.0f;

        for (int i = 0; i < size; i++) {
            float value = expf(
                input[row * size + i] - maxValue
            );

            output[row * size + i] = value;
            sum += value;
        }

        for (int i = 0; i < size; i++) {
            output[row * size + i] /= sum;
        }
    }
}

void forwardPass(
    const float* input,
    const float* W1,
    const float* b1,
    const float* W2,
    const float* b2,
    float* hiddenOutput,
    float* probabilities,
    int batchSize
) {
    float *d_input, *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_hidden, *d_output, *d_probabilities;

    size_t inputBytes =
        batchSize * INPUT_SIZE * sizeof(float);

    size_t w1Bytes =
        INPUT_SIZE * HIDDEN_SIZE * sizeof(float);

    size_t b1Bytes =
        HIDDEN_SIZE * sizeof(float);

    size_t hiddenBytes =
        batchSize * HIDDEN_SIZE * sizeof(float);

    size_t w2Bytes =
        HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);

    size_t b2Bytes =
        OUTPUT_SIZE * sizeof(float);

    size_t outputBytes =
        batchSize * OUTPUT_SIZE * sizeof(float);

    cudaMalloc((void**)&d_input, inputBytes);
    cudaMalloc((void**)&d_W1, w1Bytes);
    cudaMalloc((void**)&d_b1, b1Bytes);
    cudaMalloc((void**)&d_W2, w2Bytes);
    cudaMalloc((void**)&d_b2, b2Bytes);
    cudaMalloc((void**)&d_hidden, hiddenBytes);
    cudaMalloc((void**)&d_output, outputBytes);
    cudaMalloc((void**)&d_probabilities, outputBytes);

    cudaMemcpy(d_input, input, inputBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, w1Bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, b1Bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, w2Bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, b2Bytes,
               cudaMemcpyHostToDevice);

    matrixMultiplyKernel<<<
        dim3((HIDDEN_SIZE + 15) / 16,
             (batchSize + 15) / 16),
        dim3(16, 16)
    >>>(
        d_input, d_W1, d_hidden,
        batchSize, HIDDEN_SIZE, INPUT_SIZE
    );

    int hiddenTotal = batchSize * HIDDEN_SIZE;

    biasReluKernel<<<
        (hiddenTotal + 255) / 256, 256
    >>>(
        d_hidden, d_b1, batchSize, HIDDEN_SIZE
    );

    matrixMultiplyKernel<<<
        dim3((OUTPUT_SIZE + 15) / 16,
             (batchSize + 15) / 16),
        dim3(16, 16)
    >>>(
        d_hidden, d_W2, d_output,
        batchSize, OUTPUT_SIZE, HIDDEN_SIZE
    );

    int outputTotal = batchSize * OUTPUT_SIZE;

    biasKernel<<<
        (outputTotal + 255) / 256, 256
    >>>(
        d_output, d_b2, batchSize, OUTPUT_SIZE
    );

    softmaxKernel<<<batchSize, 1>>>(
        d_output, d_probabilities,
        batchSize, OUTPUT_SIZE
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        probabilities, d_probabilities, outputBytes,
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        hiddenOutput, d_hidden, hiddenBytes,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_probabilities);
}

#define LEARNING_OUTPUT_SIZE 10
__global__ void outputGradientKernel(
    const float* probabilities,
    const int* labels,
    float* gradients,
    int batchSize
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * LEARNING_OUTPUT_SIZE;

    if (index < total) {
        int imageIndex = index / LEARNING_OUTPUT_SIZE;
        int digitIndex = index % LEARNING_OUTPUT_SIZE;

        float target =
            (labels[imageIndex] == digitIndex) ? 1.0f : 0.0f;

        gradients[index] = probabilities[index] - target;
    }
}

__global__ void updateParametersKernel(
    float* parameters,
    const float* gradients,
    float learningRate,
    int size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        parameters[index] -=
            learningRate * gradients[index];
    }
}

void calculateOutputGradient(
    const float* probabilities,
    const int* labels,
    float* gradients,
    int batchSize
) {
    int total = batchSize * LEARNING_OUTPUT_SIZE;
    size_t probabilityBytes = total * sizeof(float);
    size_t labelBytes = batchSize * sizeof(int);
    float* d_probabilities;
    float* d_gradients;
    int* d_labels;

    cudaMalloc((void**)&d_probabilities, probabilityBytes);
    cudaMalloc((void**)&d_gradients, probabilityBytes);
    cudaMalloc((void**)&d_labels, labelBytes);

    cudaMemcpy(d_probabilities, probabilities,
               probabilityBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_labels, labels,
               labelBytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    outputGradientKernel<<<blocks, threads>>>(
        d_probabilities, d_labels,
        d_gradients, batchSize
    );

    cudaDeviceSynchronize();

    cudaMemcpy(gradients, d_gradients,
               probabilityBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_probabilities);
    cudaFree(d_gradients);
    cudaFree(d_labels);
}

void updateParameters(
    float* parameters,
    const float* gradients,
    float learningRate,
    int size
) {
    size_t bytes = size * sizeof(float);
    float *d_parameters, *d_gradients;

    cudaMalloc((void**)&d_parameters, bytes);
    cudaMalloc((void**)&d_gradients, bytes);

    cudaMemcpy(d_parameters, parameters,
               bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_gradients, gradients,
               bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    updateParametersKernel<<<blocks, threads>>>(
        d_parameters, d_gradients,
        learningRate, size
    );

    cudaDeviceSynchronize();

    cudaMemcpy(parameters, d_parameters,
               bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_parameters);
    cudaFree(d_gradients);
}

__global__ void weightGradientKernel(
    const float* hidden,
    const float* outputGrad,
    float* dW2,
    int batchSize
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * OUTPUT_SIZE;

    if (index < total) {
        int h = index / OUTPUT_SIZE;
        int o = index % OUTPUT_SIZE;

        float sum = 0.0f;

        for (int b = 0; b < batchSize; b++) {
            sum += hidden[b * HIDDEN_SIZE + h] *
                   outputGrad[b * OUTPUT_SIZE + o];
        }

        dW2[index] = sum / batchSize;
    }
}

__global__ void hiddenGradientKernel(
    const float* outputGrad,
    const float* W2,
    const float* hidden,
    float* hiddenGrad,
    int batchSize
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * HIDDEN_SIZE;

    if (index < total) {
        int b = index / HIDDEN_SIZE;
        int h = index % HIDDEN_SIZE;
        float sum = 0.0f;

        for (int o = 0; o < OUTPUT_SIZE; o++) {
            sum += outputGrad[b * OUTPUT_SIZE + o] *
                   W2[h * OUTPUT_SIZE + o];
        }

        hiddenGrad[index] =
            (hidden[index] > 0.0f) ? sum : 0.0f;
    }
}

__global__ void inputWeightGradientKernel(
    const float* input,
    const float* hiddenGrad,
    float* dW1,
    int batchSize
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = INPUT_SIZE * HIDDEN_SIZE;

    if (index < total) {
        int i = index / HIDDEN_SIZE;
        int h = index % HIDDEN_SIZE;

        float sum = 0.0f;

        for (int b = 0; b < batchSize; b++) {
            sum += input[b * INPUT_SIZE + i] *
                   hiddenGrad[b * HIDDEN_SIZE + h];
        }

        dW1[index] = sum / batchSize;
    }
}

__global__ void hiddenBiasGradientKernel(
    const float* hiddenGrad,
    float* db1,
    int batchSize
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < HIDDEN_SIZE) {
        float sum = 0.0f;

        for (int b = 0; b < batchSize; b++) {
            sum += hiddenGrad[b * HIDDEN_SIZE + h];
        }

        db1[h] = sum / batchSize;
    }
}

__global__ void outputBiasGradientKernel(
    const float* outputGrad,
    float* db2,
    int batchSize
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o < OUTPUT_SIZE) {
        float sum = 0.0f;

        for (int b = 0; b < batchSize; b++) {
            sum += outputGrad[b * OUTPUT_SIZE + o];
        }

        db2[o] = sum / batchSize;
    }
}

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
) {
    float *d_input, *d_hidden, *d_W2;
    float *d_outputGrad, *d_hiddenGrad;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;

    size_t inputBytes =
        batchSize * INPUT_SIZE * sizeof(float);
    size_t hiddenBytes =
        batchSize * HIDDEN_SIZE * sizeof(float);
    size_t outputBytes =
        batchSize * OUTPUT_SIZE * sizeof(float);

    cudaMalloc((void**)&d_input, inputBytes);
    cudaMalloc((void**)&d_hidden, hiddenBytes);
    cudaMalloc((void**)&d_W2,
               HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_outputGrad, outputBytes);
    cudaMalloc((void**)&d_hiddenGrad, hiddenBytes);

    cudaMalloc((void**)&d_dW1,
               INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_db1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_dW2,
               HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_db2, OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(d_input, input, inputBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden, hiddenBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2,
               HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputGrad, outputGradients, outputBytes,
               cudaMemcpyHostToDevice);

    int threads = 256;
    int weight2Total = HIDDEN_SIZE * OUTPUT_SIZE;
    weightGradientKernel<<<
        (weight2Total + threads - 1) / threads, threads
    >>>(d_hidden, d_outputGrad, d_dW2, batchSize);

    int hiddenTotal = batchSize * HIDDEN_SIZE;
    hiddenGradientKernel<<<
        (hiddenTotal + threads - 1) / threads, threads
    >>>(d_outputGrad, d_W2, d_hidden, d_hiddenGrad, batchSize);

    int weight1Total = INPUT_SIZE * HIDDEN_SIZE;
    inputWeightGradientKernel<<<
        (weight1Total + threads - 1) / threads, threads
    >>>(d_input, d_hiddenGrad, d_dW1, batchSize);

    hiddenBiasGradientKernel<<<
        (HIDDEN_SIZE + threads - 1) / threads, threads
    >>>(d_hiddenGrad, d_db1, batchSize);

    outputBiasGradientKernel<<<
        (OUTPUT_SIZE + threads - 1) / threads, threads
    >>>(d_outputGrad, d_db2, batchSize);

    cudaDeviceSynchronize();

    cudaMemcpy(dW1, d_dW1,
               weight1Total * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(db1, d_db1,
               HIDDEN_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dW2, d_dW2,
               weight2Total * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(db2, d_db2,
               OUTPUT_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_W2);
    cudaFree(d_outputGrad);
    cudaFree(d_hiddenGrad);
    cudaFree(d_dW1);
    cudaFree(d_db1);
    cudaFree(d_dW2);
    cudaFree(d_db2);
}

#include <cmath>

__global__ void crossEntropyKernel(
    const float* probabilities,
    const int* labels,
    float* losses,
    int batchSize
) {
    int image = blockIdx.x * blockDim.x + threadIdx.x;

    if (image < batchSize) {
        int correctLabel = labels[image];

        float probability =
            probabilities[image * OUTPUT_SIZE + correctLabel];

        if (probability < 1e-7f) {
            probability = 1e-7f;
        }

        losses[image] = -logf(probability);
    }
}

float calculateLoss(
    const float* probabilities,
    const int* labels,
    int batchSize
) {
    float *d_probabilities, *d_losses;
    int* d_labels;

    size_t probabilityBytes =
        batchSize * OUTPUT_SIZE * sizeof(float);

    size_t labelBytes = batchSize * sizeof(int);
    size_t lossBytes = batchSize * sizeof(float);

    cudaMalloc((void**)&d_probabilities, probabilityBytes);
    cudaMalloc((void**)&d_labels, labelBytes);
    cudaMalloc((void**)&d_losses, lossBytes);

    cudaMemcpy(d_probabilities, probabilities,
               probabilityBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_labels, labels,
               labelBytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batchSize + threads - 1) / threads;

    crossEntropyKernel<<<blocks, threads>>>(
        d_probabilities, d_labels, d_losses, batchSize
    );

    cudaDeviceSynchronize();
    float* losses = new float[batchSize];

    cudaMemcpy(losses, d_losses, lossBytes,
               cudaMemcpyDeviceToHost);

    float totalLoss = 0.0f;

    for (int i = 0; i < batchSize; i++) {
        totalLoss += losses[i];
    }

    float averageLoss = totalLoss / batchSize;
    delete[] losses;
    cudaFree(d_probabilities);
    cudaFree(d_labels);
    cudaFree(d_losses);
    return averageLoss;
}
