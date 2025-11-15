"""
CIFAR-10 Data Preprocessing
Loads CIFAR-10, extracts subset (500 train, 100 test per class),
and extracts features using ResNet-18 + PCA.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os


def load_cifar10_subset(train_per_class=500, test_per_class=100):
    """
    Load CIFAR-10 and extract first N images per class.

    Args:
        train_per_class: Number of training images per class (default: 500)
        test_per_class: Number of test images per class (default: 100)

    Returns:
        train_images, train_labels, test_images, test_labels
    """
    print("Loading CIFAR-10 dataset...")

    # Load full CIFAR-10 dataset
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

    # Extract subset - first N images per class
    train_images, train_labels = extract_subset(trainset, train_per_class)
    test_images, test_labels = extract_subset(testset, test_per_class)

    print(f"Extracted {len(train_images)} training images")
    print(f"Extracted {len(test_images)} test images")

    return train_images, train_labels, test_images, test_labels


def extract_subset(dataset, images_per_class):
    """
    Extract first N images from each of the 10 classes.

    Args:
        dataset: PyTorch dataset
        images_per_class: Number of images to extract per class

    Returns:
        images (numpy array), labels (numpy array)
    """
    class_counts = {i: 0 for i in range(10)}
    images = []
    labels = []

    for img, label in dataset:
        if class_counts[label] < images_per_class:
            images.append(img)
            labels.append(label)
            class_counts[label] += 1

        # Stop when we have enough images from all classes
        if all(count >= images_per_class for count in class_counts.values()):
            break

    # Convert to numpy arrays
    images = torch.stack(images).numpy()  # Shape: (N, 3, 32, 32)
    labels = np.array(labels)

    return images, labels


def extract_resnet_features(images):
    """
    Extract 512-dimensional features using pre-trained ResNet-18.

    Args:
        images: numpy array of shape (N, 3, 32, 32)

    Returns:
        features: numpy array of shape (N, 512)
    """
    print("Loading pre-trained ResNet-18...")

    # Load pre-trained ResNet-18
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # Remove the final classification layer (avgpool gives 512 features)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Preprocessing for ResNet-18: resize to 224x224 and normalize
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    print("Extracting features...")
    features = []

    # Process in batches to avoid memory issues
    batch_size = 64
    num_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))

            # Get batch and convert to torch tensor
            batch = torch.from_numpy(images[start_idx:end_idx])

            # Apply preprocessing
            batch = torch.stack([preprocess(img) for img in batch])

            # Extract features
            batch_features = model(batch)
            batch_features = batch_features.squeeze()  # Remove spatial dimensions

            features.append(batch_features.numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(images)} images")

    features = np.vstack(features)
    print(f"Extracted features shape: {features.shape}")

    return features


def apply_pca(train_features, test_features, n_components=50):
    """
    Apply PCA to reduce feature dimensionality from 512 to 50.

    Args:
        train_features: Training features (N_train, 512)
        test_features: Test features (N_test, 512)
        n_components: Target dimensionality (default: 50)

    Returns:
        train_pca, test_pca, pca_model
    """
    print(f"Applying PCA to reduce from {train_features.shape[1]} to {n_components} dimensions...")

    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)

    variance_explained = np.sum(pca.explained_variance_ratio_)
    print(f"PCA variance explained: {variance_explained:.4f}")
    print(f"Reduced features shape: {train_pca.shape}")

    return train_pca, test_pca, pca


def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("CIFAR-10 Preprocessing Pipeline")
    print("="*60)

    # Step 1: Load CIFAR-10 subset
    train_images, train_labels, test_images, test_labels = load_cifar10_subset()

    # Step 2: Extract ResNet-18 features
    print("\nExtracting training set features...")
    train_features = extract_resnet_features(train_images)

    print("\nExtracting test set features...")
    test_features = extract_resnet_features(test_images)

    # Step 3: Apply PCA
    train_pca, test_pca, pca_model = apply_pca(train_features, test_features)

    # Step 4: Save all preprocessed data
    print("\nSaving preprocessed data...")
    os.makedirs('data/processed', exist_ok=True)

    # Save raw images (for CNN)
    np.save('data/processed/train_images.npy', train_images)
    np.save('data/processed/test_images.npy', test_images)

    # Save labels
    np.save('data/processed/train_labels.npy', train_labels)
    np.save('data/processed/test_labels.npy', test_labels)

    # Save ResNet features (512-dim)
    np.save('data/processed/train_features_512.npy', train_features)
    np.save('data/processed/test_features_512.npy', test_features)

    # Save PCA features (50-dim) - for Naive Bayes, Decision Tree, MLP
    np.save('data/processed/train_features_50.npy', train_pca)
    np.save('data/processed/test_features_50.npy', test_pca)

    # Save PCA model
    with open('data/processed/pca_model.pkl', 'wb') as f:
        pickle.dump(pca_model, f)

    print("\nPreprocessing complete!")
    print("\nSaved files:")
    print("  - train_images.npy, test_images.npy (for CNN)")
    print("  - train_labels.npy, test_labels.npy")
    print("  - train_features_512.npy, test_features_512.npy (ResNet features)")
    print("  - train_features_50.npy, test_features_50.npy (PCA features)")
    print("  - pca_model.pkl")

    print("\nDataset summary:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Classes: {len(np.unique(train_labels))}")
    print(f"  Class distribution (train): {np.bincount(train_labels)}")
    print(f"  Class distribution (test): {np.bincount(test_labels)}")


if __name__ == '__main__':
    main()
