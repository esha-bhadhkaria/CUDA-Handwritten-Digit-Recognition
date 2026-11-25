# GPU-Accelerated Handwritten Digit Recognition Using Custom CUDA Kernels

## 1. Project Overview

This project implements a handwritten digit recognition system using a Multi-Layer Perceptron (MLP) neural network accelerated with custom NVIDIA CUDA kernels.

It uses the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9, each having a resolution of 28 × 28 pixels.

The main objective is to implement neural network operations using custom CUDA kernels.

## 2. Development Environment and Technologies

**Development and execution environment:** Google Colab with an NVIDIA GPU runtime.

The project was developed using the following technologies:

- C++17 – Host-side implementation.
- CUDA C++ – Custom GPU kernel implementation.
- Python – Dataset downloading and preprocessing.
- NumPy – Dataset processing.
- NVIDIA GPU – Parallel execution of CUDA kernels.

Google Colab was used to compile and execute the project using NVIDIA's CUDA compiler (`nvcc`).

## 3. Neural Network Architecture

The project uses a fully connected neural network with the following architecture:

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 784 | None |
| Hidden | 128 | ReLU |
| Output | 10 | Softmax |

Each input image is flattened into 784 pixel values. The output layer produces probabilities for the ten possible digits, and the digit with the highest probability is selected as the prediction.

## 4. CUDA Implementation

The following operations are implemented using custom CUDA kernels:

- Matrix multiplication
- Bias addition
- ReLU activation
- Softmax
- Gradient calculations for backpropagation
- Weight and bias updates
- Cross-entropy loss calculation

The C++ host code manages memory allocation, CPU-GPU data transfers and kernel execution.

## 5. Dataset

The project uses the MNIST dataset:

- 60,000 training images
- 10,000 testing images
- 28 × 28 grayscale images
- 10 digit classes (0–9)

The dataset is downloaded and preprocessed using `src/dataset.py`. Images are normalized to values between 0 and 1 and saved as binary files.

The following files are generated inside the `dataset/` directory:

```text
dataset/
├── X_train.bin
├── y_train.bin
├── X_test.bin
└── y_test.bin
```

## 6. Project Structure

```text
Handwritten-Digit-Recognition/
│
├── include/
│   ├── data_loader.h
│   └── neural_network.cuh
│
├── src/
│   ├── data_loader.cpp
│   ├── dataset.py
│   ├── main.cpp
│   └── neural_network.cu
│
├── Makefile
├── README.md
├── dataset/       (generated)
└── results/       (generated)
```

## 7. Requirements

Install NumPy if it is not already available:

```bash
pip install numpy
```

Verify CUDA availability:

```bash
nvcc --version
nvidia-smi
```

## 8. How to Run

### Step 1: Generate the dataset

```bash
python src/dataset.py
```

This downloads, preprocesses and generates the MNIST binary files inside `dataset/`.

### Step 2: Create the results directory

```bash
mkdir -p results
```

### Step 3: Compile the project

```bash
make
```

The Makefile uses `nvcc` to compile the C++ and CUDA source files and generates the `digit_recognition` executable.

### Step 4: Run the program

```bash
./digit_recognition
```

The program trains the neural network for five epochs, saves the trained model weights, performs a sample prediction and evaluates training and testing accuracy.

The trained model weights are saved at:

```text
results/model_weights.bin
```

## 9. Training Configuration

| Parameter | Value |
|---|---:|
| Training samples | 60,000 |
| Testing samples | 10,000 |
| Input neurons | 784 |
| Hidden neurons | 128 |
| Output neurons | 10 |
| Batch size | 32 |
| Epochs | 5 |
| Learning rate | 0.01 |
| Random seed | 42 |

## 10. Results

The model was successfully trained for five epochs.

### Training Loss

| Epoch | Average Loss |
|---|---:|
| 1 | 0.631176 |
| 2 | 0.333290 |
| 3 | 0.284868 |
| 4 | 0.254880 |
| 5 | 0.232223 |

### Accuracy

| Metric | Result |
|---|---:|
| Training accuracy | 93.7317% |
| Testing accuracy | 93.6899% |
| Accuracy difference | 0.0418 percentage points |

The testing evaluation processed 9,984 images in complete batches of 32.

### Sample Prediction

```text
Actual digit: 7
Predicted digit: 7
```

## 11. Challenges and Learning Outcomes

- Implementing neural network operations using custom CUDA kernels helped develop an understanding of GPU parallelism.
- Managing memory allocation and data transfers between CPU and GPU was an important part of the implementation.
- Implementing backpropagation provided practical experience with gradient calculations and parameter updates.
- Evaluating training and testing accuracy helped assess the model's performance on unseen data.

## 12. Conclusion

This project demonstrates a GPU-accelerated handwritten digit recognition system implemented using C++ and custom CUDA kernels.

The model achieved approximately 93.69% testing accuracy on the MNIST dataset. The project provided practical experience in CUDA programming, GPU memory management, neural network training and parallel computation.