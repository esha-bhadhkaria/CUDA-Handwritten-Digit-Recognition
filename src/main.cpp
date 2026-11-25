#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "data_loader.h"
#include "neural_network.cuh"
#include <fstream>

bool loadModel(
    std::vector<float>& W1,
    std::vector<float>& b1,
    std::vector<float>& W2,
    std::vector<float>& b2
) {
    std::ifstream modelFile(
        "results/model_weights.bin",
        std::ios::binary
    );

    if (!modelFile) {
        std::cout << "Failed to load model weights!\n";
        return false;
    }

    modelFile.read(
        reinterpret_cast<char*>(W1.data()),
        W1.size() * sizeof(float)
    );

    modelFile.read(
        reinterpret_cast<char*>(b1.data()),
        b1.size() * sizeof(float)
    );

    modelFile.read(
        reinterpret_cast<char*>(W2.data()),
        W2.size() * sizeof(float)
    );

    modelFile.read(
        reinterpret_cast<char*>(b2.data()),
        b2.size() * sizeof(float)
    );

    if (!modelFile) {
        std::cout << "Error reading model weights!\n";
        return false;
    }

    std::cout << "Model weights loaded successfully!\n";
    return true;
}

int main() {

    const int TRAIN_SIZE = 60000;
    const int TEST_SIZE = 10000;
    const int INPUT_SIZE = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;
    const int BATCH_SIZE = 32;
    const int TRAIN_SAMPLES = 60000;
    const int EPOCHS = 5;
    const float LEARNING_RATE = 0.01f;

    std::vector<float> trainImages(
        TRAIN_SIZE * INPUT_SIZE
    );
    std::vector<int> trainLabels(TRAIN_SIZE);

    std::vector<float> testImages(
        TEST_SIZE * INPUT_SIZE
    );
    std::vector<int> testLabels(TEST_SIZE);

    if (!loadData("dataset/X_train.bin",
                  trainImages.data(),
                  TRAIN_SIZE * INPUT_SIZE) ||
        !loadLabels("dataset/y_train.bin",
                    trainLabels.data(), TRAIN_SIZE) ||
        !loadData("dataset/X_test.bin",
                  testImages.data(),
                  TEST_SIZE * INPUT_SIZE) ||
        !loadLabels("dataset/y_test.bin",
                    testLabels.data(), TEST_SIZE)) {
        std::cerr << "Dataset loading failed!\n";
        return 1;
    }

    // Initialize weights
    std::mt19937 generator(42);
    std::normal_distribution<float> dist1(
        0.0f, 0.05f
    );

    std::normal_distribution<float> dist2(
        0.0f, 0.125f
    );

    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);

    std::vector<float> b1(HIDDEN_SIZE, 0.0f);
    std::vector<float> b2(OUTPUT_SIZE, 0.0f);

    for (float& w : W1) {
        w = dist1(generator);
    }

    for (float& w : W2) {
        w = dist2(generator);
    }

    std::cout << "Starting Training.\n";

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        float totalLoss = 0.0f;
        int batches = 0;

        for (int start = 0;
             start + BATCH_SIZE <= TRAIN_SAMPLES;
             start += BATCH_SIZE) {

            // Prepare current batch
            std::vector<float> input(
                BATCH_SIZE * INPUT_SIZE
            );

            std::vector<int> labels(BATCH_SIZE);

            for (int i = 0; i < BATCH_SIZE; i++) {

                int index = start + i;

                std::copy(
                    trainImages.begin() + index * INPUT_SIZE,
                    trainImages.begin() + (index + 1) * INPUT_SIZE,
                    input.begin() + i * INPUT_SIZE
                );

                labels[i] = trainLabels[index];
            }

            // Forward pass
            std::vector<float> hidden(
                BATCH_SIZE * HIDDEN_SIZE
            );

            std::vector<float> probabilities(
                BATCH_SIZE * OUTPUT_SIZE
            );

            forwardPass(
                input.data(),
                W1.data(), b1.data(),
                W2.data(), b2.data(),
                hidden.data(),
                probabilities.data(),
                BATCH_SIZE
            );

            // Calculate loss
            float loss = calculateLoss(
                probabilities.data(),
                labels.data(),
                BATCH_SIZE
            );

            totalLoss += loss;
            batches++;

            // Calculate output gradients
            std::vector<float> outputGradients(
                BATCH_SIZE * OUTPUT_SIZE
            );

            calculateOutputGradient(
                probabilities.data(),
                labels.data(),
                outputGradients.data(),
                BATCH_SIZE
            );

            std::vector<float> dW1(
                INPUT_SIZE * HIDDEN_SIZE
            );

            std::vector<float> db1(HIDDEN_SIZE);

            std::vector<float> dW2(
                HIDDEN_SIZE * OUTPUT_SIZE
            );

            std::vector<float> db2(OUTPUT_SIZE);

            // Backpropagation
            calculateGradients(
                input.data(),
                hidden.data(),
                W2.data(),
                outputGradients.data(),
                dW1.data(),
                db1.data(),
                dW2.data(),
                db2.data(),
                BATCH_SIZE
            );

            // Update parameters
            updateParameters(
                W1.data(), dW1.data(),
                LEARNING_RATE, W1.size()
            );

            updateParameters(
                b1.data(), db1.data(),
                LEARNING_RATE, b1.size()
            );

            updateParameters(
                W2.data(), dW2.data(),
                LEARNING_RATE, W2.size()
            );

            updateParameters(
                b2.data(), db2.data(),
                LEARNING_RATE, b2.size()
            );

            if (batches % 20 == 0) {
                std::cout
                    << "Epoch " << epoch + 1
                    << " | Batch " << batches
                    << " | Loss: " << loss
                    << "\n";
            }
        }

        std::cout << "\nEpoch " << epoch + 1
                  << " Average Loss: "
                  << totalLoss / batches
                  << "\n";
    }


    // Save trained model weights
    std::cout << "\nSaving Model Weights.\n";

    std::ofstream modelFile(
        "results/model_weights.bin",
        std::ios::binary
    );

    if (!modelFile) {
        std::cerr << "Failed to save model weights.\n";
        return 1;
    }

    modelFile.write(
        reinterpret_cast<const char*>(W1.data()),
        W1.size() * sizeof(float)
    );

    modelFile.write(
        reinterpret_cast<const char*>(b1.data()),
        b1.size() * sizeof(float)
    );

    modelFile.write(
        reinterpret_cast<const char*>(W2.data()),
        W2.size() * sizeof(float)
    );

    modelFile.write(
        reinterpret_cast<const char*>(b2.data()),
        b2.size() * sizeof(float)
    );

    modelFile.close();

    if (!modelFile) {
        std::cerr << "Error writing model weights.\n";
        return 1;
    }

    std::cout << "Model weights saved successfully.\n";

    // Load saved model weights
    if (!loadModel(W1, b1, W2, b2)) {
        return 1;
    }

    // Predict one test image
    std::vector<float> input(INPUT_SIZE);

    std::copy(
        testImages.begin(),
        testImages.begin() + INPUT_SIZE,
        input.begin()
    );

    std::vector<float> hidden(HIDDEN_SIZE);
    std::vector<float> probabilities(OUTPUT_SIZE);

    forwardPass(
        input.data(),
        W1.data(), b1.data(),
        W2.data(), b2.data(),
        hidden.data(),
        probabilities.data(),
        1
    );

    int predicted = 0;

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (probabilities[i] > probabilities[predicted]) {
            predicted = i;
        }
    }

    std::cout << "\nPREDICTION\n";
    std::cout << "Actual digit: " << testLabels[0] << "\n";
    std::cout << "Predicted digit: " << predicted << "\n";

    
    // Evaluate on training data
    std::cout << "\nEvaluating Training Data.\n";

    int trainCorrect = 0;
    const int EVAL_BATCH_SIZE = 32;

    for (int start = 0;
         start + EVAL_BATCH_SIZE <= TRAIN_SAMPLES;
         start += EVAL_BATCH_SIZE) {

        std::vector<float> input(
            EVAL_BATCH_SIZE * INPUT_SIZE
        );

        for (int i = 0; i < EVAL_BATCH_SIZE; i++) {
            std::copy(
                trainImages.begin() + (start + i) * INPUT_SIZE,
                trainImages.begin() + (start + i + 1) * INPUT_SIZE,
                input.begin() + i * INPUT_SIZE
            );
        }

        std::vector<float> hidden(
            EVAL_BATCH_SIZE * HIDDEN_SIZE
        );

        std::vector<float> probabilities(
            EVAL_BATCH_SIZE * OUTPUT_SIZE
        );

        forwardPass(
            input.data(),
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            hidden.data(),
            probabilities.data(),
            EVAL_BATCH_SIZE
        );

        for (int i = 0; i < EVAL_BATCH_SIZE; i++) {

            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (probabilities[i * OUTPUT_SIZE + j] >
                    probabilities[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }

            if (predicted == trainLabels[start + i]) {
                trainCorrect++;
            }
        }
    }

    float trainAccuracy = 100.0f * trainCorrect / TRAIN_SAMPLES;

    // Evaluate on test data
    std::cout << "\nEvaluating Test Data...\n";

    int correct = 0;
    const int TEST_BATCH_SIZE = 32;

    for (int start = 0;
         start + TEST_BATCH_SIZE <= TEST_SIZE;
         start += TEST_BATCH_SIZE) {

        std::vector<float> input(
            TEST_BATCH_SIZE * INPUT_SIZE
        );

        for (int i = 0; i < TEST_BATCH_SIZE; i++) {
            std::copy(
                testImages.begin() + (start + i) * INPUT_SIZE,
                testImages.begin() + (start + i + 1) * INPUT_SIZE,
                input.begin() + i * INPUT_SIZE
            );
        }

        std::vector<float> hidden(
            TEST_BATCH_SIZE * HIDDEN_SIZE
        );

        std::vector<float> probabilities(
            TEST_BATCH_SIZE * OUTPUT_SIZE
        );

        forwardPass(
            input.data(),
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            hidden.data(),
            probabilities.data(),
            TEST_BATCH_SIZE
        );

        for (int i = 0; i < TEST_BATCH_SIZE; i++) {

            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (probabilities[i * OUTPUT_SIZE + j] >
                    probabilities[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }

            if (predicted == testLabels[start + i]) {
                correct++;
            }
        }
    }

    int evaluatedTestSamples =
        TEST_SIZE - TEST_SIZE % TEST_BATCH_SIZE;

    float testAccuracy =
        100.0f * correct / evaluatedTestSamples;

    std::cout << "\nRESULTS\n";

    std::cout << "Training samples: "
              << TRAIN_SAMPLES << "\n";

    std::cout << "Training correct predictions: "
              << trainCorrect << "\n";

    std::cout << "Training Accuracy: "
              << trainAccuracy << "%\n";

    std::cout << "\nTest samples: "
              << evaluatedTestSamples << "\n";

    std::cout << "Test correct predictions: "
              << correct << "\n";

    std::cout << "Test Accuracy: "
              << testAccuracy << "%\n";

    std::cout << "Accuracy Difference: "
              << trainAccuracy - testAccuracy << "%\n";

    return 0;
}
