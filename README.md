Overview
This repository contains the implementation and insights for the VLG Recruitment Challenge 2024, which involved classifying animals into 40 distinct categories. The dataset posed challenges such as class imbalance, noise, and distribution shifts between the training and test datasets. The solution employed advanced deep learning techniques and ConvNeXt-Tiny, a state-of-the-art architecture, to tackle these issues.

Steps of Implementation
1. Dataset Preparation
Dataset Structure:
Training set: 40 subfolders (one per class) with varying numbers of images.
Test set: 3,000 unlabeled images.
Challenges:
Class imbalance: Classes had between 150–250 images.
Noise: Redundant or mislabeled samples.
Solutions:
Weighted random sampling for class balancing.
Synthetic data augmentation to equalize class representation.
2. Preprocessing
Images were resized to 224×224 to match the ConvNeXt input size.
Normalized using ImageNet statistics:
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
Applied augmentations:
Random horizontal flips.
Random rotations (up to 30 degrees).
3. Model Selection
ConvNeXt-Tiny was chosen for its balance of accuracy and efficiency.
Key architectural features:
Large kernels (7x7) for broader spatial context.
Depthwise convolutions for efficient feature extraction.
Layer normalization replacing batch normalization.
4. Custom Model Head
The default classification head was replaced with a custom head

5. Training
Loss Function: Cross-entropy with label smoothing (factor=0.2) to soften target probabilities.
Optimizer: AdamW with weight decay (1e-3).
Scheduler: Cosine annealing for smooth learning rate adjustments.
Early Stopping: Halted training after 3 consecutive epochs without validation loss improvement.

7. Validation
15% of training data was used as a validation set.
Validation accuracy peaked at 96%, indicating good fit on the training data.

9. Inference
Predictions on the test set were saved to a CSV file.
Grad-CAM was used for model explainability, highlighting the focus on species-specific features.
Performance
Training Accuracy: 97.8%
Validation Accuracy: 96%
Test Accuracy: ~59% on both public and private leaderboards.
Observations:
A significant drop in test accuracy suggests potential dataset shift. Several test set classes were not present in the training set.
Challenges
Overfitting:

Despite regularization (dropout, weight decay), the model overfitted to the training data.
Key insight: Some test set classes were missing in the training data, limiting generalization.
Dataset Noise:

Redundancies and mislabeled samples affected model robustness.
Hardware Constraints:

Training was constrained by memory, requiring smaller batch sizes (32).
Future Directions
Zero-Shot Learning:

Use CLIP (OpenAI) for handling unseen test classes.
Expected improvement: ~20% test accuracy boost.
Data Augmentation:

Advanced synthetic augmentation (e.g., CutMix, GAN-based techniques).
Ensemble Learning:

Combine predictions from multiple architectures.
Hyperparameter Tuning:

Use Optuna for fine-tuning learning rates, dropout, and weight decay.
Here is a brief step by step summary for implementation of code->

1. Dataset Preparation
Define Paths: Paths for training and test datasets are set.
Transformations:
Base Transformations:
Resize images to 224×224.
Apply augmentations like random horizontal flips and rotations.
Normalize images with ImageNet mean and standard deviation.
Custom Dataset Class:
A custom ImageFolder is implemented to:
Balance classes by augmenting underrepresented classes.
Handle transformations during data loading.


2. Data Loading
Class Balancing:

Classes are balanced using:
Synthetic duplication for underrepresented classes.
Weighted random sampling based on class frequencies.
Validation Split:

15% of the balanced training dataset is split into a validation set.
DataLoaders:

DataLoaders are created for training and validation sets:
Batch size: 32.
Weighted sampling for the training set.

3. Model Definition
Base Model:
A pretrained ConvNeXt-Small model is loaded.
Custom Classification Head:
The default classification head is replaced with:
Dropout layers for regularization.
Fully connected layers with batch normalization and ReLU activation.
Final layer outputs predictions for 40 classes.

4. Training Setup
Device Selection:
Use GPU if available; otherwise, fallback to CPU.
Loss Function:
Cross-Entropy Loss with label smoothing (factor=0.2) to handle noisy labels.
Optimizer:
AdamW optimizer with weight decay (1e-3).
Scheduler:
Cosine annealing learning rate scheduler to adjust the learning rate smoothly.
Mixed Precision Training:
AMP (Automatic Mixed Precision) is used for faster computation.


5. Training Loop
Initialize Metrics:
Track training loss, accuracy, and validation performance.
Train Phase:
Iterate over training batches:
Forward pass with AMP.
Compute loss.
Backpropagation and weight updates using the scaled gradient.
Validation Phase:
Evaluate the model on the validation set:
Compute validation loss and accuracy.
Save the best model based on validation loss.
Early Stopping:
Stop training if validation loss does not improve for 3 consecutive epochs.


6. Test Predictions
Inference Mode:

Set the model to evaluation mode to disable gradient computation.
Prediction Loop:
For each test image:

Apply transformations and pass through the model.

Extract the predicted class.

Save Predictions:

Store results in a CSV file with columns Id and Category.
Summary of Key Steps

Data Preparation: Load, augment, and balance the dataset.

Model Definition: Use ConvNeXt-Small with a custom classification head.

Training:
Loss: Cross-entropy with label smoothing.

Optimizer: AdamW.

Scheduler: Cosine annealing.

Validation: Monitor validation performance for early stopping.

Inference: Predict classes for the test set and save to a CSV file.





