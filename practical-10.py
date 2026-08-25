import numpy as np

# ---- Part A: Backpropagation demo from scratch (single hidden layer, XOR problem) ----
np.random.seed(42)

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
input_neurons, hidden_neurons, output_neurons = 2, 4, 1
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

lr = 0.5
epochs = 10000

for epoch in range(epochs):
    # ---- Forward propagation ----
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # ---- Compute error ----
    error = y - final_output

    # ---- Backpropagation ----
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- Update weights (gradient descent) ----
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal predictions after training (XOR):")
print(final_output.round(3))

# ---- Part B: End-to-end ML application ----
# Full workflow: load -> preprocess -> split -> train -> evaluate -> predict on new sample
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names) # pyright: ignore[reportAttributeAccessIssue]
y = data.target # pyright: ignore[reportAttributeAccessIssue]

# 2. Preprocess: check for missing values, scale features
print("Missing values:", X.isnull().sum().sum())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nEnd-to-end Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names)) # pyright: ignore[reportAttributeAccessIssue]

# 6. Save model + scaler for deployment
joblib.dump(model, 'final_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 7. Predict on a new unseen sample (simulate deployment)
new_sample = X.iloc[[0]]  # example new input
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print("\nPrediction for new sample:", data.target_names[prediction[0]]) # pyright: ignore[reportAttributeAccessIssue]