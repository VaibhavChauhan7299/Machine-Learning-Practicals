"""Practical 5: Regression Algorithm — Implementation and Evaluation
Objective: Implement a regression algorithm and evaluate the regression model.
Explanation: Regression predicts a continuous value. We use Linear Regression on the California housing dataset. Evaluation metrics: MAE, MSE, RMSE, and R² score (goodness of fit)."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data # pyright: ignore[reportAttributeAccessIssue]
y = data.target # pyright: ignore[reportAttributeAccessIssue]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

print("\nModel Coefficients:")
for name, coef in zip(data.feature_names, model.coef_): # pyright: ignore[reportAttributeAccessIssue]
    print(f"  {name}: {coef:.4f}")
print("Intercept:", model.intercept_)

"""Interpretation: R² close to 1 means the model explains most of the variance in house prices; RMSE gives the average prediction error in the same units as the target."""