import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_iris()
X = data.data  # pyright: ignore[reportAttributeAccessIssue] # unlabeled -- we ignore data.target during clustering

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Elbow method to choose k ----
inertias = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('elbow_method.png')
plt.close()

# ---- Fit K-Means with chosen k=3 ----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df = pd.DataFrame(X, columns=data.feature_names) # pyright: ignore[reportAttributeAccessIssue]
df['Cluster'] = clusters
df['ActualSpecies'] = data.target  # pyright: ignore[reportAttributeAccessIssue] # only used afterward, for analysis/validation

print("Cluster centers (scaled space):\n", kmeans.cluster_centers_)
print("\nCluster sizes:\n", df['Cluster'].value_counts())

# Compare clusters to actual species (just for analysis)
print("\nCross-tab of Cluster vs Actual Species:\n", pd.crosstab(df['Cluster'], df['ActualSpecies']))

sil_score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", sil_score)