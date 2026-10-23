# Import KMeans from sklearn.cluster

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare the customer data using a python list
customer_data = [
    [1,000, 2],
    [1,500, 3],
    [2,000, 2],
    [5,000, 6],
    [6,000, 7],
    [5,500, 5],
    [12,000, 12],
    [15,000, 14],
    [13,000, 11],
    [3,000, 4]
]
# Use StandardScaler to scale the spending and visits features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Create a KMeans model using n_clusters=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model using the customer dataset
kmeans.fit(scaled_data)

# Display the cluster assigned to each customer
labels = kmeans.labels_
for i, label in enumerate(labels):
    print(f"Customer {i+1} belongs to cluster {label}")
print("\n" "="*30 + "\n")

# Display the total number of customers in each cluster
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

for cluster, count in cluster_counts.items():
    print(f"cluster {cluster}: {count} customers")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare the customer data using a python list
customer_data = ( [1000, 2], [1500, 3], [2000, 2], [5000, 6], [6000, 7], [5500, 5], [12000, 12], [15000, 14], [13000, 11], [3000, 4])

# Use StandardScaler to scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Create and train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled_data)

# --- MATPLOTLIB VISUALIZATION ---

# Create the figure
plt.figure(figsize=(8, 6))

# Scatter plot: Color points by the cluster label using a colormap (cmap)
scatter = plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=labels,
    cmap='viridis',
    s=100,
    edgecolor='k',
    alpha=0.8
)
# Plot the cluster centers (centroids)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='x',
    s=200,
    label='Centroids'
)

# Add labels, title, grid, and legend
plt.title('Customer Segments (KMeans Clustering)', fontsize=14, fontweight='bold')
plt.xlabel('Scaled Spending Amount', fontsize=12)
plt.ylabel('Scaled Frequency of Visits', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Generate a legend for the cluster groups
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper left")
plt.gca().add_artist(legend1)
plt.legend(loc="lower right")

# Save or show the plot
plt.tight_layout()
plt.savefig('customer_clusters.png')
plt.show()



