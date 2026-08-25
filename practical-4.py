"""Practical 4: Classification Algorithm — Implementation and Evaluation
Objective: Implement a classification algorithm on an appropriate dataset and evaluate it.
Explanation: Classification predicts a discrete label. Here we use Decision Tree and K-Nearest Neighbors (KNN) on the Iris dataset, and compare them using accuracy, precision, recall, F1, and a confusion matrix."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = load_iris()
X, y = data.data, data.target # pyright: ignore[reportAttributeAccessIssue]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# ---- Decision Tree Classifier ----
dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, target_names=data.target_names)) # pyright: ignore[reportAttributeAccessIssue]

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names, filled=True) # pyright: ignore[reportAttributeAccessIssue]
plt.savefig('decision_tree.png')
plt.close()

# ---- KNN Classifier ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=data.target_names)) # pyright: ignore[reportAttributeAccessIssue]

print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))