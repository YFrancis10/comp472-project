"""
Decision Tree Implementation using Gini coefficient
Uses only NumPy for custom implementation
"""

import numpy as np


class Node:
    """Represents a node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class value if leaf node


class DecisionTree:
    """
    Custom Decision Tree classifier using Gini coefficient.
    Uses only NumPy for implementation.
    """

    def __init__(self, max_depth=50, min_samples_split=2):
        """
        Initialize Decision Tree.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        """
        Calculate Gini impurity.

        Gini = 1 - sum(p_i^2) where p_i is proportion of class i

        Args:
            y: Class labels

        Returns:
            Gini impurity value
        """
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def _split(self, X, y, feature, threshold):
        """
        Split dataset based on feature and threshold.

        Args:
            X: Features
            y: Labels
            feature: Feature index to split on
            threshold: Threshold value

        Returns:
            Left and right splits (X_left, y_left, X_right, y_right)
        """
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        """
        Find the best split for a node.

        Args:
            X: Features
            y: Labels

        Returns:
            best_feature, best_threshold
        """
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        # Try each feature
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            # Try each unique value as threshold
            for threshold in thresholds:
                # Split data
                X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)

                # Skip if split doesn't separate data
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate weighted Gini impurity
                n_left, n_right = len(y_left), len(y_right)
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)

                weighted_gini = (n_left / n_samples) * gini_left + \
                               (n_right / n_samples) * gini_right

                # Update best split if better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Args:
            X: Features
            y: Labels
            depth: Current depth in tree

        Returns:
            Node representing root of (sub)tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Create leaf node with majority class
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # Find best split
        feature, threshold = self._best_split(X, y)

        # If no valid split found, create leaf
        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # Split data
        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)

        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=feature, threshold=threshold,
                   left=left_child, right=right_child)

    def fit(self, X, y):
        """
        Train the decision tree.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        """
        Traverse tree to predict a single sample.

        Args:
            x: Single sample features
            node: Current node

        Returns:
            Predicted class
        """
        # If leaf node, return its value
        if node.value is not None:
            return node.value

        # Otherwise, traverse left or right based on feature value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def score(self, X, y):
        """
        Calculate accuracy on test set.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
