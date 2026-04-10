import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# 1. Generate Synthetic Crime Data (Baghdad Coordinates)
np.random.seed(42)
num_crimes = 300
latitudes = np.random.normal(loc=33.3152, scale=0.05, size=num_crimes)
longitudes = np.random.normal(loc=44.3661, scale=0.05, size=num_crimes)
data = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})

# 2. Apply AI Model (K-Means Clustering)
num_hotspots = 5
kmeans = KMeans(n_clusters=num_hotspots, random_state=42, n_init=10)
data['Hotspot_Cluster'] = kmeans.fit_predict(data[['Latitude', 'Longitude']])
cluster_centers = kmeans.cluster_centers_

# 3. Visualize the Results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Hotspot_Cluster', palette='viridis', data=data, s=50, alpha=0.7)
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='red', s=200, marker='X', label='Hotspot Centers')

plt.title('Crime Hotspot Prediction using K-Means Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.savefig('hotspots_result.png')
plt.show()

print("AI Model Execution Complete.")
