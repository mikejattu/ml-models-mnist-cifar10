# Machine Learning Models for MNIST and CIFAR-10

This repository contains implementations of machine learning models for the MNIST and CIFAR-10 datasets using PyTorch. The models are:

1. **Logistic Regression for MNIST**
2. **Fully-connected Neural Network (FNN) for CIFAR-10**
3. **Hyperparameter Search using Adam Optimizer**

## Table of Contents
- [Logistic Regression for MNIST](#logistic-regression-for-mnist)
- [Fully-connected Neural Network for CIFAR-10](#fully-connected-neural-network-for-cifar-10)
- [Hyperparameter Search](#hyperparameter-search)
- [Requirements](#requirements)
- [Running the Code](#running-the-code)
- [File Structure](#file-structure)

## Logistic Regression for MNIST

### Task:
Implemented a logistic regression model using PyTorch and train it on the MNIST dataset. The dataset is split such that the last 12,000 samples of the training set are used as a validation set. The model is trained using stochastic gradient descent (SGD) and cross-entropy loss.

### Features:
- Used SGD optimizer.
- Implemented L1 or L2 regularization for model improvement.
- Evaluated the model on the validation set every few epochs to prevent overfitting.

### Performance:
Achieving an accuracy between 92-94% on both the test and validation sets is expected with proper model tuning.

You can find the implementation in the `logistic_regression` function in `A1_submission.py`.

## Fully-connected Neural Network for CIFAR-10

### Task:
Implemented a fully-connected neural network for CIFAR-10 classification. The network uses cross-entropy loss and is trained on the CIFAR-10 dataset, consisting of RGB images of size 32x32x3.

### Network Architecture:
- **Input Layer**: Flattened 32x32x3 images.
- **Hidden Layers**: A series of layers with ReLU, Tanh, and Softmax activations.
- **Output Layer**: Produces class probabilities for the 10 CIFAR-10 classes.

The network should be trained with a dataset split of 40,000 for training and 10,000 for validation.

### Implementation:
The forward pass and loss computation are implemented in the `FNN` class in `A1_submission.py`. You can find the exact architecture in the project details above.

## Hyperparameter Search

### Task:
Performed a hyperparameter search for the Logistic Regression and Fully-connected Neural Network models using the Adam optimizer.

### Method:
- Conducted grid search for hyperparameter optimization.
- Useed the validation accuracy to evaluate the best hyperparameter configuration.

The function `tune_hyper_parameter` in `A1_submission.py` handles the hyperparameter tuning.


## Sample Dataset Images

### MNIST
![MNIST Samples](MNIST%20Sample%20Digits.png)

### CIFAR-10
![CIFAR-10 Samples](CIFAR%2010%20Image%20Samples.png)
## Requirements

To run the code, you need to have Python (version >= 3.6) installed and the following packages:

```bash
python3 -m pip install numpy torch torchvision tqdm paramparse
```
## Running the Code
1. Clone the repository:
```bash
git clone https://github.com/mikejattu/ml-models-mnist-cifar10.git
cd ml-models-mnist-cifar10
```
2. Install the required dependencies:
```bash
python3 -m pip install numpy torch torchvision tqdm paramparse
```
3. Run the code
   For Logistic Regression on MNIST:
```bash
python A1_main.py
```
  For the Fully-connected Neural Network on CIFAR-10:
```bash
python FNN_main.py
```

## File Structure
```
ml-models-mnist-cifar10/
│
├── A1_main.py               # Main entry point for running the logistic regression model on MNIST
├── FNN_main.py              # Main entry point for running the fully-connected neural network on CIFAR-10
├── A1_submission.py         # Contains implementations of models, hyperparameter tuning, and additional functions
│
└── README.md                # Project description, instructions, and details
```
