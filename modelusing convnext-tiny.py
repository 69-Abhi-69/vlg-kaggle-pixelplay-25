import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm
import torch.cuda.amp as amp
import csv
from PIL import Image
import time

# Paths to the dataset (update paths as needed)
train_dir = r"C:\Users\Abhi\Downloads\vlg-recruitment-24-challenge\vlg-dataset\train"
test_dir = r"C:\Users\Abhi\Downloads\vlg-recruitment-24-challenge\vlg-dataset\test"

# Base transformation applied to all images
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ConvNeXt-Small input
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
    transforms.RandomRotation(30),  # Random rotation of images for data augmentation
    transforms.ToTensor(),  # Convert PIL image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Custom dataset class to handle augmentations and class balancing
class CustomImageFolderWithAugmentation(ImageFolder):
    def __init__(self, root, transform=None, max_images_per_class=1000):
        """
        Custom class that extends ImageFolder to handle class balancing.
        Args:
        - root: Path to dataset directory
        - transform: Transform to apply on images
        - max_images_per_class: Maximum number of images per class to balance the dataset
        """
        super().__init__(root, transform)
        self.max_images_per_class = max_images_per_class  # Max images per class for balancing
        self.class_indices = self.get_class_indices()  # Get the indices for each class

    def get_class_indices(self):
        """
        Group indices by class.
        Returns a dictionary with class index as keys and list of image indices as values.
        """
        class_indices = {cls: [] for cls in range(len(self.classes))}  # Initialize dictionary for class indices
        for idx, (_, target) in enumerate(self.samples):
            class_indices[target].append(idx)  # Append the sample index to the corresponding class
        return class_indices

    def get_subset_indices(self):
        """
        Create a balanced subset by adding more images up to max_images_per_class for each class.
        """
        subset_indices = []  # List to store selected subset indices
        for class_idx, indices in self.class_indices.items():
            total_images = len(indices)
            if total_images < self.max_images_per_class:
                subset_indices.extend(indices)  # Add all images if class size is less than max
                augment_count = self.max_images_per_class - total_images  # Number of augmentations needed
                subset_indices.extend(indices[:augment_count])  # Duplicate samples to balance the class
            else:
                subset_indices.extend(indices[:self.max_images_per_class])  # Trim to max if class size exceeds max
        return subset_indices

    def __getitem__(self, index):
        """
        Override the __getitem__ method to apply transformations to the image before returning.
        """
        image, target = super().__getitem__(index)  # Get the image and target from the parent class
        if isinstance(image, Image.Image):  # If the image is a PIL Image
            image = base_transform(image)  # Apply the base transformation
        return image, target

# Training loop for model training and validation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, patience=3):
    """
    Trains and evaluates the model over a number of epochs with early stopping.
    """
    model.train()  # Set the model to training mode
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(num_epochs):  # Loop over epochs
        epoch_start_time = time.time()  # Track epoch duration
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Training phase
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):  # Loop over training batches
            images, labels = images.to(device), labels.to(device)  # Move data to the device (GPU/CPU)

            optimizer.zero_grad()  # Zero the gradients
            with amp.autocast():  # Automatic Mixed Precision (AMP) for faster training
                outputs = model(images)  # Forward pass through the model
                loss = criterion(outputs, labels)  # Calculate loss

            scaler.scale(loss).backward()  # Scaled backpropagation for stability
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()  # Update the scaler

            # Update training metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            train_correct += (predicted == labels).sum().item()  # Count correct predictions
            train_total += labels.size(0)  # Count total samples

        # Calculate training metrics
        train_loss /= len(train_loader)  # Average training loss
        train_accuracy = 100 * train_correct / train_total  # Training accuracy

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in val_loader:  # Loop over validation batches
                images, labels = images.to(device), labels.to(device)  # Move data to device
                outputs = model(images)  # Forward pass through the model
                loss = criterion(outputs, labels)  # Calculate loss
                val_loss += loss.item()  # Update validation loss
                _, predicted = outputs.max(1)  # Get predicted class
                val_correct += (predicted == labels).sum().item()  # Count correct predictions
                val_total += labels.size(0)  # Count total samples

        # Calculate validation metrics
        val_loss /= len(val_loader)  # Average validation loss
        val_accuracy = 100 * val_correct / val_total  # Validation accuracy

        # Print results for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping and saving the best model
        if val_loss < best_val_loss:  # If the validation loss improves
            best_val_loss = val_loss  # Update best validation loss
            epochs_without_improvement = 0  # Reset early stopping counter
            torch.save(model.state_dict(), "convnext_small_custom_head_best.pth")  # Save the model
            print("Saved best model.")
        else:
            epochs_without_improvement += 1  # Increment the early stopping counter
            if epochs_without_improvement >= patience:  # If no improvement for 'patience' epochs
                print("Early stopping triggered.")
                break

        scheduler.step()  # Update the learning rate based on the scheduler
        print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f}s")  # Print epoch duration

# Generate predictions on the test dataset and save to a CSV file
def generate_predictions(model, test_dir, transform, output_csv):
    """
    Generate predictions for the test dataset and save the results to a CSV file.
    """
    model.eval()  # Set model to evaluation mode
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]  # List test images
    predictions = []  # List to store predictions

    for img_name in test_images:  # Loop through each test image
        img_path = os.path.join(test_dir, img_name)  # Get full path to the image
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
        image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(image)  # Forward pass through the model
            predicted_class = torch.argmax(outputs, dim=1).item()  # Get the predicted class

        # Append predictions to the list
        predictions.append((img_name, original_train_dataset.classes[predicted_class]))

    # Save predictions to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)  # Initialize CSV writer
        writer.writerow(["Id", "Category"])  # Write CSV header
        writer.writerows(predictions)  # Write predictions as rows

if __name__ == '__main__':
    # Load and preprocess the dataset
    original_train_dataset = CustomImageFolderWithAugmentation(root=train_dir, transform=base_transform)
    subset_indices = original_train_dataset.get_subset_indices()  # Get the balanced subset indices
    train_dataset_augmented = Subset(original_train_dataset, subset_indices)

    # Split dataset into training and validation sets
    val_size = int(0.15 * len(train_dataset_augmented))  # 15% for validation
    train_size = len(train_dataset_augmented) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_augmented, [train_size, val_size])

    # Weighted sampler for class balancing (using inverse class frequencies)
    class_weights = [1.0 / max(1, len(original_train_dataset.get_class_indices()[cls])) for cls in range(len(original_train_dataset.classes))]
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # Model definition (ConvNeXt-Small with a custom classification head)
    class ConvNeXtWithCustomHead(nn.Module):
        def __init__(self, num_classes):
            super(ConvNeXtWithCustomHead, self).__init__()
            self.model = models.convnext_small(pretrained=True)  # Load pre-trained ConvNeXt-Small
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Sequential(  # Modify classifier head
                nn.Dropout(0.6),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.model(x)

    # Initialize the model, device, and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtWithCustomHead(num_classes=len(original_train_dataset.classes)).to(device)

    # Hyperparameters and optimizer setup
    num_epochs = 20
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Cross-entropy loss with label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)  # AdamW optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # Learning rate scheduler
    scaler = amp.GradScaler()  # AMP gradient scaler for mixed precision training

    # Train and evaluate the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs)

    # Generate predictions for the test set and save to CSV
    generate_predictions(model, test_dir, base_transform, "predictions.csv")
