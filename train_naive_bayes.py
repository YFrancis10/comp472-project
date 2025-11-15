"""
Train and evaluate Gaussian Naive Bayes models
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
import os
from models.naive_bayes import GaussianNaiveBayes


def load_data():
    """Load preprocessed PCA features."""
    print("Loading preprocessed data...")
    X_train = np.load('data/processed/train_features_50.npy')
    y_train = np.load('data/processed/train_labels.npy')
    X_test = np.load('data/processed/test_features_50.npy')
    y_test = np.load('data/processed/test_labels.npy')

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def train_custom_nb(X_train, y_train):
    """Train custom Gaussian Naive Bayes."""
    print("\n" + "="*60)
    print("Training Custom Gaussian Naive Bayes")
    print("="*60)

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    print("Training complete!")
    return model


def train_sklearn_nb(X_train, y_train):
    """Train scikit-learn Gaussian Naive Bayes."""
    print("\n" + "="*60)
    print("Training Scikit-learn Gaussian Naive Bayes")
    print("="*60)

    model = GaussianNB()
    model.fit(X_train, y_train)

    print("Training complete!")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance."""
    print(f"\nEvaluating {model_name}...")

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return train_acc, test_acc


def save_model(model, filename):
    """Save trained model."""
    os.makedirs('saved_models', exist_ok=True)
    filepath = os.path.join('saved_models', filename)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def main():
    """Main training pipeline."""
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train custom Naive Bayes
    custom_nb = train_custom_nb(X_train, y_train)
    custom_train_acc, custom_test_acc = evaluate_model(
        custom_nb, X_train, y_train, X_test, y_test, "Custom Naive Bayes"
    )
    save_model(custom_nb, 'naive_bayes_custom.pkl')

    # Train scikit-learn Naive Bayes
    sklearn_nb = train_sklearn_nb(X_train, y_train)
    sklearn_train_acc, sklearn_test_acc = evaluate_model(
        sklearn_nb, X_train, y_train, X_test, y_test, "Scikit-learn Naive Bayes"
    )
    save_model(sklearn_nb, 'naive_bayes_sklearn.pkl')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 60)
    print(f"{'Custom Naive Bayes':<30} {custom_train_acc:<15.4f} {custom_test_acc:<15.4f}")
    print(f"{'Scikit-learn Naive Bayes':<30} {sklearn_train_acc:<15.4f} {sklearn_test_acc:<15.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
