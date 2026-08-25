"""Practical 3: Train an ML Model and Evaluate Using Train/Test Split
Objective: Split data into training/testing sets, train a model, evaluate performance, and interpret results.
Explanation: The dataset is split (commonly 70:30 or 80:20) into training data (to fit the model) and testing data (to check generalization on unseen data). Common evaluation metrics: accuracy, precision, recall, F1-score (classification) or MSE/R² (regression)."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data # pyright: ignore[reportAttributeAccessIssue]
y = iris.target # pyright: ignore[reportAttributeAccessIssue]

# Split into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names)) # pyright: ignore[reportAttributeAccessIssue]

"""Interpretation: Accuracy tells overall correctness; the confusion matrix shows per-class errors; precision/recall/F1 show how well each class is predicted, useful when classes are imbalanced."""