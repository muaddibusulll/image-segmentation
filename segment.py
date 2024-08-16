import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize the weights, means, and covariances
        np.random.seed(42)
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])

        log_likelihood = 0
        
        for _ in range(self.max_iter):
            # E-step: Calculate the responsibilities
            responsibilities = np.zeros((n_samples, self.n_components))
            
            for i in range(self.n_components):
                rv = multivariate_normal(mean=self.means_[i], cov=self.covariances_[i])
                responsibilities[:, i] = self.weights_[i] * rv.pdf(X)
                
            total_responsibility = responsibilities.sum(axis=1)[:, np.newaxis]
            responsibilities /= total_responsibility

            # M-step: Update the weights, means, and covariances
            Nk = responsibilities.sum(axis=0)
            self.weights_ = Nk / n_samples
            self.means_ = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
            
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for i in range(self.n_components):
                X_centered = X - self.means_[i]
                self.covariances_[i] = (responsibilities[:, i][:, np.newaxis] * X_centered).T @ X_centered / Nk[i]

            # Check for convergence
            new_log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            if np.abs(new_log_likelihood - log_likelihood) <= self.tol:
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            rv = multivariate_normal(mean=self.means_[i], cov=self.covariances_[i])
            responsibilities[:, i] = self.weights_[i] * rv.pdf(X)
        
        return responsibilities.argmax(axis=1)

# Load CIFAR-10 dataset
cifar10 = fetch_openml('CIFAR_10_small')
X = cifar10['data']
y = cifar10['target']

# Reshape a single image (e.g., first image) into a 2D array of pixels
img = X[0].reshape(32, 32, 3)

# Reshape the image to have pixels as rows and RGB as columns
pixels = img.reshape(-1, 3)

# Define the number of clusters (e.g., 3)
n_clusters = 3

# Fit a Gaussian Mixture Model using the custom EM algorithm
gmm = GaussianMixtureModel(n_components=n_clusters)
gmm.fit(pixels)

# Predict the cluster for each pixel
segmented_img = gmm.predict(pixels)

# Reshape the labels back to the original image shape
segmented_img = segmented_img.reshape(32, 32)

# Plot the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title(f"Segmented Image with {n_clusters} Clusters")
plt.imshow(segmented_img, cmap='viridis')

plt.show()
