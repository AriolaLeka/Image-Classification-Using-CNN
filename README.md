# Image Classification Using ConvNets

This repository contains an image classification project using Convolutional Neural Networks (ConvNets) to accurately classify images from the CIFAR-10 dataset. The goal is to correctly identify objects belonging to ten different classes: plane, car, bird, cat, deer, dog, frog, horse, ship, or truck.

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Bonus](#bonus)
- [Code](#usage)

## About

In this project, we implemented a Convolutional Neural Network (ConvNet) to perform image classification on the CIFAR-10 dataset. The network is trained to accurately categorize images into one of the ten predefined classes. Our goal is to achieve a high level of accuracy by applying various techniques such as data normalization, dropout regularization, and hyperparameter tuning.

## Dataset

The CIFAR-10 dataset consists of 60,000 images divided into a training set of 50,000 images and a test set of 10,000 images. Each image has dimensions 32×32 with three color channels: red, green, and blue. The images belong to ten classes, making it a challenging benchmark for image classification algorithms.

## Model Architecture

We designed a Convolutional Neural Network with the following architecture:

- Convolutional Layer 1: 32 filters, 3 × 3
- Convolutional Layer 2: 32 filters, 3 × 3
- Max-pooling Layer 1: 2 × 2 windows
- Convolutional Layer 3: 64 filters, 3 × 3
- Convolutional Layer 4: 64 filters, 3 × 3
- Max-pooling Layer 2: 2 × 2 windows
- Fully Connected Layer 1: 512 units
- Output Layer

## Training

Our training pipeline involves the following steps:

1. Data normalization using `transforms.Normalize`.
2. Splitting the original training set into a validation set and an effective training set.
3. Training the model using stochastic gradient descent (SGD) optimizer with specified hyperparameters.
4. Monitoring training loss, accuracy, and validation accuracy during training.
5. Adding dropout layers after each max-pooling layer for regularization.

## Results

We achieved the following results:

- Final validation accuracy: Above 75%
- Test accuracy: Above 80% (Bonus)

We plotted the evolution of loss and accuracy for both the training and validation sets to visualize the training process.

## Bonus

For the bonus section, we aimed to achieve a test accuracy above 80% by experimenting with additional techniques and hyperparameter adjustments. We documented the changes we made and provided insights into their impact on model performance.

## Usage

To explore and reproduce the image classification results, follow these steps:

1. Clone the repository: `git clone https://github.com/AriolaLeka/Image-Classification-Using-CNN.git`
2. Navigate to the project directory: `cd Image-Classification-Using-CNN`
3. Open the Python script: `python image classification.py`
4. The script will load the dataset, implement the model, and train it.
