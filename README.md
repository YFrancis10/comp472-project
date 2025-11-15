## Project Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── preprocess_data.py         # Data loading and feature extraction
├── models/                    # Model implementations
│   ├── naive_bayes.py        # Custom Gaussian Naive Bayes
│   ├── decision_tree.py      # Custom Decision Tree with Gini
│   ├── mlp.py                # MLP variants (base, deep, wide, narrow)
│   └── cnn.py                # VGG11 variants
├── train_naive_bayes.py      # Train Naive Bayes models
├── train_decision_tree.py    # Train Decision Tree models + experiments
├── train_mlp.py              # Train MLP models + experiments
├── train_cnn.py              # Train CNN models + experiments
├── evaluate.py               # Comprehensive evaluation script
├── data/                     # Dataset and preprocessed features
│   ├── cifar-10-batches-py/  # Downloaded CIFAR-10 data
│   └── processed/            # Preprocessed features
├── saved_models/             # Trained model files
└── results/                  # Confusion matrices and metrics
```

Note: Saved models vg11_base.pth and vgg11_large_kernel.pth are not included in the github since files exceed 100MB

## Setup and Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Data Preprocessing

This step loads CIFAR-10, extracts 500 training and 100 test images per class, and performs feature extraction.

```bash
python preprocess_data.py
```

**What this does:**
- Downloads CIFAR-10 dataset (if not present)
- Extracts subset: 500 train + 100 test images per class
- Uses pre-trained ResNet-18 to extract 512-dim features
- Applies PCA to reduce features to 50 dimensions
- Saves:
  - Raw images (for CNN)
  - 512-dim ResNet features
  - 50-dim PCA features (for Naive Bayes, Decision Tree, MLP)

**Expected output:**
- Training samples: 5000 (500 per class × 10 classes)
- Test samples: 1000 (100 per class × 10 classes)
- Files saved in `data/processed/`

### Step 2: Train Models

Train each model type individually:

#### Naive Bayes

```bash
python train_naive_bayes.py
```

Trains both custom (NumPy-only) and scikit-learn implementations.

#### Decision Tree

```bash
python train_decision_tree.py
```

Trains:
- Custom Decision Tree (max_depth=50) using Gini coefficient
- Scikit-learn Decision Tree for comparison
- Depth experiments (5, 10, 20, 30, 40, 50)

#### Multi-Layer Perceptron

```bash
python train_mlp.py
```

Trains:
- Base MLP: 50→512→512→10
- Deep MLP: 5 layers
- Wide MLP: 1024 hidden units
- Narrow MLP: 128 hidden units

All use SGD optimizer (momentum=0.9) and CrossEntropyLoss.
Training runs for 50 epochs.

#### Convolutional Neural Network (VGG11)

```bash
python train_cnn.py
```

Trains:
- VGG11 Base (as specified)
- VGG11 Shallow (fewer conv layers)
- VGG11 Large Kernel (5×5 kernels)

All use SGD optimizer (momentum=0.9) and CrossEntropyLoss.
Training runs for 100 epochs.

### Step 3: Evaluate All Models

After training all models, run comprehensive evaluation:

```bash
python evaluate.py
```

**Generates:**
- Confusion matrices for all models (saved as PNG images in `results/`)
- Metrics table (accuracy, precision, recall, F1-score)
- Summary file: `results/metrics_summary.txt`

## Trained Models

All trained models are saved in `saved_models/`:

- `naive_bayes_custom.pkl` - Custom Naive Bayes
- `naive_bayes_sklearn.pkl` - Scikit-learn Naive Bayes
- `decision_tree_custom.pkl` - Custom Decision Tree
- `decision_tree_sklearn.pkl` - Scikit-learn Decision Tree
- `decision_tree_depth_*.pkl` - Best model from depth experiments
- `mlp_base.pth` - Base MLP
- `mlp_deep.pth` - Deep MLP
- `mlp_wide.pth` - Wide MLP
- `mlp_narrow.pth` - Narrow MLP
- `vgg11_base.pth` - VGG11 Base (Not included in github since file size exceeds 100MB)
- `vgg11_shallow.pth` - VGG11 Shallow
- `vgg11_large_kernel.pth` - VGG11 Large Kernel (Not included in github since file size exceeds 100MB)

## Model Details

### Gaussian Naive Bayes
- **Input:** 50-dim PCA features
- **Implementation:** Custom (NumPy only) + scikit-learn comparison
- **Key assumption:** Features follow Gaussian distribution within each class

### Decision Tree
- **Input:** 50-dim PCA features
- **Implementation:** Custom (NumPy only) with Gini impurity
- **Max depth:** 50 (default), with experiments at 5, 10, 20, 30, 40, 50

### Multi-Layer Perceptron (MLP)
- **Input:** 50-dim PCA features
- **Architecture:** Linear(50,512) - ReLU - Linear(512,512) - BatchNorm - ReLU - Linear(512,10)
- **Loss:** CrossEntropyLoss
- **Optimizer:** SGD (lr=0.01, momentum=0.9)
- **Variants:** Deep (5 layers), Wide (1024 units), Narrow (128 units)

### VGG11 CNN
- **Input:** 32×32×3 RGB images (raw)
- **Architecture:** 8 conv layers + 3 FC layers
- **Loss:** CrossEntropyLoss
- **Optimizer:** SGD (lr=0.01, momentum=0.9)
- **Variants:** Shallow (fewer conv layers), Large Kernel (5×5)

## Evaluation Metrics

For each model, the following metrics are calculated:

1. **Accuracy:** Overall classification accuracy
2. **Precision:** Weighted average precision across all classes
3. **Recall:** Weighted average recall across all classes
4. **F1-Score:** Weighted average F1-score
5. **Confusion Matrix:** 10×10 matrix showing per-class performance

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

