# vlg-kaggle-pixelplay-25
This is the repository for the kaggle challenge for IIT-R freshers (pixel-play'25)
Project Overview
This repository contains a comprehensive implementation of an image classification pipeline using ConvNeXt-Tiny, a state-of-the-art deep learning model. The model is fine-tuned for classifying images into predefined categories. The repository includes preprocessing steps, training scripts, validation mechanisms, and inference capabilities for generating predictions on unseen data.
Key Features
Dataset Handling:
Custom handling of imbalanced datasets using weighted random sampling.
Automatic augmentation to increase the number of samples per class.
Support for structured folder datasets with subfolders representing classes.
Model:
Fine-tuning of ConvNeXt-Tiny, a modern convolutional neural network architecture optimized for image classification.
Addition of a custom classification head with Batch Normalization and Dropout for better generalization.
Label smoothing for robustness against noisy labels.
Optimization:
Training with AdamW optimizer and cosine annealing learning rate scheduler.
Early stopping to prevent overfitting based on validation loss.
Inference:
Generates predictions on test datasets and exports results in a CSV format.
Model Architecture
Stem (Patchify Stem):

Initial 4x4 convolution layer with a stride of 4 to reduce spatial resolution.
Produces feature maps of size 56x56.
Stages:

Stage 1:
Input resolution: 56x56.
Number of ConvNeXt Blocks: 3.
Feature dimension: 96.
Stage 2:
Input resolution: 28x28.
Number of ConvNeXt Blocks: 3.
Feature dimension: 192.
Stage 3:
Input resolution: 14x14.
Number of ConvNeXt Blocks: 9.
Feature dimension: 384.
Stage 4:
Input resolution: 7x7.
Number of ConvNeXt Blocks: 3.
Feature dimension: 768.
Classification Head:

A global average pooling layer to reduce spatial dimensions to a single vector.
Custom fully connected layers:
Linear (768 → 512) with Batch Normalization, Dropout (0.6), and ReLU.
Linear (512 → number of classes) with Dropout (0.6).
Important Notes
Dataset Handling:
The training dataset should be structured as:
Copy code
train/
├── class_1/
├── class_2/
├── ...
└── class_n/
Synthetic Augmentations:
Data augmentations applied include Random Resized Crop, Horizontal Flip, and Random Rotation.
Weighted random sampling is used to address class imbalance.
Performance Monitoring:
Early stopping is implemented with a patience of 3 epochs to minimize overfitting.
Training and validation accuracy/loss metrics are logged for each epoch.
Future Directions
Advanced Augmentations:
Incorporate CutMix and MixUp augmentations for synthetic data generation.
K-Fold Cross-Validation:
Use k-fold CV for better model robustness and validation.
Model Ensemble:
Combine predictions from multiple models to improve generalization.
Explainability:
Use Grad-CAM to visualize feature importance in decision-making.
Results
Test Accuracy: 59.1%
Observations:
The model generalizes well to common patterns but struggles with rare classes.
Increasing dataset diversity and synthetic augmentations could further improve performance.
Contact
For questions or collaboration, contact Abhiraj Bharangar at abhiraj_b@me.iitr.ac.in.

