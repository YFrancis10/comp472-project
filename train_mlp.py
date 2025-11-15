"""
Train and evaluate MLP models
Includes depth and layer size experiments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from models.mlp import MLP, MLPDeep, MLPWide, MLPNarrow


def load_data():
    """Load preprocessed PCA features and convert to PyTorch tensors."""
    print("Loading preprocessed data...")
    X_train = np.load('data/processed/train_features_50.npy')
    y_train = np.load('data/processed/train_labels.npy')
    X_test = np.load('data/processed/test_features_50.npy')
    y_test = np.load('data/processed/test_labels.npy')

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """
    Train the MLP model.

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
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training parameters
    learning_rate = 0.01
    momentum = 0.9
    epochs = 50

    results = []

    # 1. Train base MLP
    print("="*60)
    print("Training Base MLP (50→512→512→10)")
    print("="*60)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model = train_model(model, train_loader, criterion, optimizer, device, epochs)

    train_acc = evaluate_model(model, train_loader, device)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\nBase MLP - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    save_model(model, 'mlp_base.pth')
    results.append(('Base MLP (50→512→512→10)', train_acc, test_acc))

    # 2. Train deeper MLP
    print("\n" + "="*60)
    print("Training Deep MLP (5 layers)")
    print("="*60)
    model_deep = MLPDeep().to(device)
    optimizer_deep = optim.SGD(model_deep.parameters(), lr=learning_rate, momentum=momentum)

    model_deep = train_model(model_deep, train_loader, criterion, optimizer_deep, device, epochs)

    train_acc_deep = evaluate_model(model_deep, train_loader, device)
    test_acc_deep = evaluate_model(model_deep, test_loader, device)
    print(f"\nDeep MLP - Train Accuracy: {train_acc_deep:.4f}, Test Accuracy: {test_acc_deep:.4f}")
    save_model(model_deep, 'mlp_deep.pth')
    results.append(('Deep MLP (5 layers)', train_acc_deep, test_acc_deep))

    # 3. Train wider MLP
    print("\n" + "="*60)
    print("Training Wide MLP (1024 hidden units)")
    print("="*60)
    model_wide = MLPWide().to(device)
    optimizer_wide = optim.SGD(model_wide.parameters(), lr=learning_rate, momentum=momentum)

    model_wide = train_model(model_wide, train_loader, criterion, optimizer_wide, device, epochs)

    train_acc_wide = evaluate_model(model_wide, train_loader, device)
    test_acc_wide = evaluate_model(model_wide, test_loader, device)
    print(f"\nWide MLP - Train Accuracy: {train_acc_wide:.4f}, Test Accuracy: {test_acc_wide:.4f}")
    save_model(model_wide, 'mlp_wide.pth')
    results.append(('Wide MLP (1024 units)', train_acc_wide, test_acc_wide))

    # 4. Train narrower MLP
    print("\n" + "="*60)
    print("Training Narrow MLP (128 hidden units)")
    print("="*60)
    model_narrow = MLPNarrow().to(device)
    optimizer_narrow = optim.SGD(model_narrow.parameters(), lr=learning_rate, momentum=momentum)

    model_narrow = train_model(model_narrow, train_loader, criterion, optimizer_narrow, device, epochs)

    train_acc_narrow = evaluate_model(model_narrow, train_loader, device)
    test_acc_narrow = evaluate_model(model_narrow, test_loader, device)
    print(f"\nNarrow MLP - Train Accuracy: {train_acc_narrow:.4f}, Test Accuracy: {test_acc_narrow:.4f}")
    save_model(model_narrow, 'mlp_narrow.pth')
    results.append(('Narrow MLP (128 units)', train_acc_narrow, test_acc_narrow))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - MLP Models")
    print("="*60)
    print(f"{'Model':<35} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 65)
    for name, train_acc, test_acc in results:
        print(f"{name:<35} {train_acc:<15.4f} {test_acc:<15.4f}")
    print("="*65)


if __name__ == '__main__':
    main()
