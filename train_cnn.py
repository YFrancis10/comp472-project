"""
Train and evaluate CNN (VGG11) models
Includes depth and kernel size experiments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from models.cnn import VGG11, VGG11Shallow, VGG11LargeKernel


def load_data():
    """Load preprocessed raw images and convert to PyTorch tensors."""
    print("Loading preprocessed data...")
    X_train = np.load('data/processed/train_images.npy')
    y_train = np.load('data/processed/train_labels.npy')
    X_test = np.load('data/processed/test_images.npy')
    y_test = np.load('data/processed/test_labels.npy')

    # Convert from (N, 3, 32, 32) to torch tensors
    # Data is already in correct format from preprocessing
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    """
    Train the CNN model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            acc = 100 * correct / total
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluate model accuracy.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to evaluate on

    Returns:
        Accuracy score
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def save_model(model, filename):
    """Save trained model."""
    os.makedirs('saved_models', exist_ok=True)
    filepath = os.path.join('saved_models', filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def main():
    """Main training pipeline."""
    # Set device
    device = torch.device('cpu')  # CPU only as specified
    print(f"Using device: {device}\n")

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Create data loaders
    batch_size = 32  # Smaller batch size for CNN
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training parameters
    learning_rate = 0.01
    momentum = 0.9
    epochs = 100

    results = []

    # 1. Train base VGG11
    print("="*60)
    print("Training VGG11 (Base)")
    print("="*60)
    model = VGG11().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model = train_model(model, train_loader, criterion, optimizer, device, epochs)

    train_acc = evaluate_model(model, train_loader, device)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\nVGG11 Base - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    save_model(model, 'vgg11_base.pth')
    results.append(('VGG11 Base', train_acc, test_acc))

    # 2. Train shallower VGG11
    print("\n" + "="*60)
    print("Training VGG11 Shallow (Fewer Conv Layers)")
    print("="*60)
    model_shallow = VGG11Shallow().to(device)
    optimizer_shallow = optim.SGD(model_shallow.parameters(), lr=learning_rate, momentum=momentum)

    model_shallow = train_model(model_shallow, train_loader, criterion, optimizer_shallow, device, epochs)

    train_acc_shallow = evaluate_model(model_shallow, train_loader, device)
    test_acc_shallow = evaluate_model(model_shallow, test_loader, device)
    print(f"\nVGG11 Shallow - Train Accuracy: {train_acc_shallow:.4f}, Test Accuracy: {test_acc_shallow:.4f}")
    save_model(model_shallow, 'vgg11_shallow.pth')
    results.append(('VGG11 Shallow', train_acc_shallow, test_acc_shallow))

    # 3. Train VGG11 with larger kernels
    print("\n" + "="*60)
    print("Training VGG11 Large Kernel (5x5)")
    print("="*60)
    model_large = VGG11LargeKernel().to(device)
    optimizer_large = optim.SGD(model_large.parameters(), lr=learning_rate, momentum=momentum)

    model_large = train_model(model_large, train_loader, criterion, optimizer_large, device, epochs)

    train_acc_large = evaluate_model(model_large, train_loader, device)
    test_acc_large = evaluate_model(model_large, test_loader, device)
    print(f"\nVGG11 Large Kernel - Train Accuracy: {train_acc_large:.4f}, Test Accuracy: {test_acc_large:.4f}")
    save_model(model_large, 'vgg11_large_kernel.pth')
    results.append(('VGG11 Large Kernel (5x5)', train_acc_large, test_acc_large))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - CNN Models")
    print("="*60)
    print(f"{'Model':<35} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 65)
    for name, train_acc, test_acc in results:
        print(f"{name:<35} {train_acc:<15.4f} {test_acc:<15.4f}")
    print("="*65)


if __name__ == '__main__':
    main()
