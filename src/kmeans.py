import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

class kmeans:
    def __init__(self, k=2, max_iters=10, color_space="YCbCr", channels=[1]):
        self.k = k
        self.max_iters = max_iters
        self.color_space = color_space
        self.channels = channels

    def initialize_centroids(self, data, k):
        # np.random.seed(50)
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    def assign_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, data, labels, centroids, k):
        return np.array([
                data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(k)
            ])

    def kmeans_core(self, data, k, max_iters):
        centroids = self.initialize_centroids(data, k)
        for _ in range(max_iters):
            labels = self.assign_clusters(data, centroids)
            new_centroids = self.update_centroids(data, labels, centroids, k)

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels, centroids

    def clustering(self, image):
        # Convert color space
        if self.color_space == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "Lab":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        elif self.color_space == "YCbCr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Select channels
        h, w = image.shape[:2]
        img_data = image[:, :, self.channels]
        pixels = img_data.reshape(-1, len(self.channels)).astype(np.float32)

        labels, centroids = self.kmeans_core(pixels, self.k, self.max_iters)

        label_counts = np.bincount(labels)
        dominant_label = np.argmax(label_counts)
        
        # Assign colors
        output = np.zeros((h * w, 3), dtype=np.uint8)
        for i in range(self.k):
            if i == dominant_label:
                output[labels == i] = [0, 0, 0]  # Black for background
            else:
                color = np.random.randint(50, 255, size=3)
                output[labels == i] = color

        segmented_image = output.reshape((h, w, 3))
        return segmented_image
