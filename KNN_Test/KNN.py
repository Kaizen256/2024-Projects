import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_clusters_per_class=1, 
                           n_informative=2, n_redundant=0, n_repeated=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the KNN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the class for the test set
y_pred = knn.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=100, edgecolor='k', label='Training data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='s', s=100, edgecolor='k', label='Predicted test data')
plt.title(f'KNN Classification (k={k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()