"""
Comprehensive evaluation script for all models
Generates confusion matrices and calculates all metrics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

from models.naive_bayes import GaussianNaiveBayes
from models.decision_tree import DecisionTree
from models.mlp import MLP, MLPDeep, MLPWide, MLPNarrow
from models.cnn import VGG11, VGG11Shallow, VGG11LargeKernel


# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_pkl_model(filepath):
    """Load a pickle model."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_pytorch_model(model_class, filepath, device):
    """Load a PyTorch model."""
    model = model_class().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model


def get_predictions_sklearn(model, X):
    """Get predictions from scikit-learn style model."""
    return model.predict(X)


def get_predictions_pytorch(model, data_loader, device):
    """Get predictions from PyTorch model."""
    predictions = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    return np.array(predictions)


def calculate_metrics(y_true, y_pred):
    """
    Calculate all required metrics.

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Generate and save confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: Output filename
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {filepath}")


def evaluate_model(model, X, y, model_name, model_type='sklearn'):
    """
    Evaluate a single model and generate all metrics.

    Args:
        model: Trained model
        X: Test features
        y: Test labels
        model_name: Name for display
        model_type: 'sklearn' or 'pytorch'

    Returns:
        Dictionary with metrics and predictions
    """
    print(f"\nEvaluating {model_name}...")

    if model_type == 'sklearn':
        y_pred = get_predictions_sklearn(model, X)
    elif model_type == 'pytorch':
        y_pred = get_predictions_pytorch(model, X, device='cpu')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    metrics = calculate_metrics(y, y_pred)

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    # Generate confusion matrix
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(y, y_pred, f"Confusion Matrix - {model_name}", filename)

    return {
        'name': model_name,
        'predictions': y_pred,
        'metrics': metrics
    }


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    X_features = np.load('data/processed/test_features_50.npy')
    y_test = np.load('data/processed/test_labels.npy')

    X_images = np.load('data/processed/test_images.npy')
    X_images_tensor = torch.FloatTensor(X_images)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loader for PyTorch models
    test_dataset = TensorDataset(X_images_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cpu')

    results = []

    # Evaluate Naive Bayes models
    if os.path.exists('saved_models/naive_bayes_custom.pkl'):
        model = load_pkl_model('saved_models/naive_bayes_custom.pkl')
        result = evaluate_model(model, X_features, y_test,
                              "Custom Naive Bayes", "sklearn")
        results.append(result)

    if os.path.exists('saved_models/naive_bayes_sklearn.pkl'):
        model = load_pkl_model('saved_models/naive_bayes_sklearn.pkl')
        result = evaluate_model(model, X_features, y_test,
                              "Scikit-learn Naive Bayes", "sklearn")
        results.append(result)

    # Evaluate Decision Tree models
    if os.path.exists('saved_models/decision_tree_custom.pkl'):
        model = load_pkl_model('saved_models/decision_tree_custom.pkl')
        result = evaluate_model(model, X_features, y_test,
                              "Custom Decision Tree", "sklearn")
        results.append(result)

    if os.path.exists('saved_models/decision_tree_sklearn.pkl'):
        model = load_pkl_model('saved_models/decision_tree_sklearn.pkl')
        result = evaluate_model(model, X_features, y_test,
                              "Scikit-learn Decision Tree", "sklearn")
        results.append(result)

    # Evaluate MLP models
    mlp_models = [
        ('mlp_base.pth', MLP, "MLP Base"),
        ('mlp_deep.pth', MLPDeep, "MLP Deep"),
        ('mlp_wide.pth', MLPWide, "MLP Wide"),
        ('mlp_narrow.pth', MLPNarrow, "MLP Narrow")
    ]

    for filename, model_class, name in mlp_models:
        filepath = os.path.join('saved_models', filename)
        if os.path.exists(filepath):
            # For MLP, create a data loader with features
            features_dataset = TensorDataset(
                torch.FloatTensor(X_features),
                y_test_tensor
            )
            features_loader = DataLoader(features_dataset, batch_size=32, shuffle=False)

            model = load_pytorch_model(model_class, filepath, device)
            result = evaluate_model(model, features_loader, y_test,
                                  name, "pytorch")
            results.append(result)

    # Evaluate CNN models
    cnn_models = [
        ('vgg11_base.pth', VGG11, "VGG11 Base"),
        ('vgg11_shallow.pth', VGG11Shallow, "VGG11 Shallow"),
        ('vgg11_large_kernel.pth', VGG11LargeKernel, "VGG11 Large Kernel")
    ]

    for filename, model_class, name in cnn_models:
        filepath = os.path.join('saved_models', filename)
        if os.path.exists(filepath):
            model = load_pytorch_model(model_class, filepath, device)
            result = evaluate_model(model, test_loader, y_test,
                                  name, "pytorch")
            results.append(result)

    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - ALL MODELS")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 83)

    for result in results:
        m = result['metrics']
        print(f"{result['name']:<35} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f}")

    print("="*83)

    # Save results to file
    with open('results/metrics_summary.txt', 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 83 + "\n")
        for result in results:
            m = result['metrics']
            f.write(f"{result['name']:<35} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
                   f"{m['recall']:<12.4f} {m['f1']:<12.4f}\n")

    print("\nResults saved to results/metrics_summary.txt")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
