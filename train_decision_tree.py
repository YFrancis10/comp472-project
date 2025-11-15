"""
Train and evaluate Decision Tree models
Includes custom implementation and depth experiments
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
from models.decision_tree import DecisionTree


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


def train_custom_dt(X_train, y_train, max_depth=50):
    """Train custom Decision Tree."""
    print(f"\nTraining Custom Decision Tree (max_depth={max_depth})...")

    model = DecisionTree(max_depth=max_depth)
    model.fit(X_train, y_train)

    print("Training complete!")
    return model


def train_sklearn_dt(X_train, y_train, max_depth=50):
    """Train scikit-learn Decision Tree."""
    print(f"\nTraining Scikit-learn Decision Tree (max_depth={max_depth})...")

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Training complete!")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return train_acc, test_acc


def save_model(model, filename):
    """Save trained model."""
    os.makedirs('saved_models', exist_ok=True)
    filepath = os.path.join('saved_models', filename)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def depth_experiments(X_train, y_train, X_test, y_test):
    """Run experiments with varying tree depths."""
    print("\n" + "="*60)
    print("DEPTH EXPERIMENTS")
    print("="*60)

    depths = [5, 10, 20, 30, 40, 50]
    results = []

    for depth in depths:
        # Train custom model
        model = train_custom_dt(X_train, y_train, max_depth=depth)
        train_acc, test_acc = evaluate_model(model, X_train, y_train, X_test, y_test)

        results.append({
            'depth': depth,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': model
        })

        print(f"Depth {depth}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    # Save best model (based on test accuracy)
    best_result = max(results, key=lambda x: x['test_acc'])
    save_model(best_result['model'],
              f'decision_tree_depth_{best_result["depth"]}.pkl')

    return results


def main():
    """Main training pipeline."""
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train custom Decision Tree (depth=50)
    print("\n" + "="*60)
    print("Training Custom Decision Tree (max_depth=50)")
    print("="*60)
    custom_dt = train_custom_dt(X_train, y_train, max_depth=50)
    custom_train_acc, custom_test_acc = evaluate_model(
        custom_dt, X_train, y_train, X_test, y_test
    )
    print(f"Training Accuracy: {custom_train_acc:.4f}")
    print(f"Test Accuracy: {custom_test_acc:.4f}")
    save_model(custom_dt, 'decision_tree_custom.pkl')

    # Train scikit-learn Decision Tree (depth=50)
    print("\n" + "="*60)
    print("Training Scikit-learn Decision Tree (max_depth=50)")
    print("="*60)
    sklearn_dt = train_sklearn_dt(X_train, y_train, max_depth=50)
    sklearn_train_acc, sklearn_test_acc = evaluate_model(
        sklearn_dt, X_train, y_train, X_test, y_test
    )
    print(f"Training Accuracy: {sklearn_train_acc:.4f}")
    print(f"Test Accuracy: {sklearn_test_acc:.4f}")
    save_model(sklearn_dt, 'decision_tree_sklearn.pkl')

    # Depth experiments
    depth_results = depth_experiments(X_train, y_train, X_test, y_test)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Decision Trees")
    print("="*60)
    print(f"{'Model':<40} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 70)
    print(f"{'Custom DT (depth=50)':<40} {custom_train_acc:<15.4f} {custom_test_acc:<15.4f}")
    print(f"{'Scikit-learn DT (depth=50)':<40} {sklearn_train_acc:<15.4f} {sklearn_test_acc:<15.4f}")
    print()
    print("Depth Experiments:")
    for result in depth_results:
        print(f"{'  Depth=' + str(result['depth']):<40} {result['train_acc']:<15.4f} {result['test_acc']:<15.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
