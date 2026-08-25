import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and prepare data
data = load_iris()
X, y = data.data, data.target # pyright: ignore[reportAttributeAccessIssue]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---- Define ANN architecture ----
# hidden_layer_sizes=(16, 8) -> two hidden layers with 16 and 8 neurons
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# ---- Train ----
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names)) # pyright: ignore[reportAttributeAccessIssue]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot training loss curve (this is where backpropagation's effect is visible)
plt.figure()
plt.plot(model.loss_curve_)
plt.xlabel('Iteration (epoch)')
plt.ylabel('Loss')
plt.title('ANN Training Loss Curve (via Backpropagation)')
plt.savefig('ann_training.png')
plt.close()