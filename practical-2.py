"""Practical 2: Feature Selection and Dimensionality Reduction
Objective: Identify relevant features and demonstrate feature subset selection / dimensionality reduction.
Explanation:
Feature selection picks the most relevant original features (e.g., using correlation, SelectKBest with chi2/ANOVA, or feature importance from a tree model).
Dimensionality reduction (e.g., PCA) transforms features into a smaller set of new components that retain most of the variance/information. Both reduce overfitting, training time, and noise."""

import pandas as pd
import numpy as np
from typing import Any, cast
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = cast(Any, load_breast_cancer())
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

print("Original number of features:", X.shape[1])

# ---- 1. Feature Selection using SelectKBest (ANOVA F-test) ----
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nTop 10 selected features (SelectKBest):\n", list(selected_features))

# ---- 2. Feature Importance using Random Forest ----
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features by Random Forest importance:\n", importances.head(10))

# ---- 3. Dimensionality Reduction using PCA ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print("\nOriginal shape:", X.shape)
print("Shape after PCA:", X_pca.shape)
print("\nExplained variance ratio per component:\n", pca.explained_variance_ratio_)
print("Total variance captured by 5 components: {:.2f}%".format(pca.explained_variance_ratio_.sum()*100))