"""
Gaussian Naive Bayes Implementation
Uses only NumPy for custom implementation
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Custom Gaussian Naive Bayes classifier using only NumPy.

    Assumes features follow a Gaussian (normal) distribution within each class.
    Uses maximum likelihood estimation to learn parameters.
    """

    def __init__(self):
        self.classes = None
        self.class_priors = None  # P(class)
        self.means = None  # Mean of each feature for each class
        self.variances = None  # Variance of each feature for each class

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameters
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # Get all samples belonging to class c

            self.means[idx, :] = np.mean(X_c, axis=0)
            self.variances[idx, :] = np.var(X_c, axis=0)
            self.class_priors[idx] = X_c.shape[0] / n_samples

    def _gaussian_pdf(self, x, mean, var):
        """
        Calculate Gaussian probability density function.

        P(x | mean, var) = (1 / sqrt(2*pi*var)) * exp(-(x - mean)^2 / (2*var))

        Args:
            x: Feature value(s)
            mean: Mean of the distribution
            var: Variance of the distribution

        Returns:
            Probability density
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-9
        var = var + eps

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def _predict_sample(self, x):
        """
        Predict class for a single sample.

        Args:
            x: Single sample features, shape (n_features,)

        Returns:
            Predicted class label
        """
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            # Start with log of class prior
            prior = np.log(self.class_priors[idx])

            # Calculate likelihood using Gaussian PDF for each feature
            # Use log probabilities to avoid underflow
            likelihood = np.sum(np.log(self._gaussian_pdf(
                x, self.means[idx, :], self.variances[idx, :]
            )))

            # Posterior = prior * likelihood (in log space: log_prior + log_likelihood)
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        predictions = [self._predict_sample(x) for x in X]
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
