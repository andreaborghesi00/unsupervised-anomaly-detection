# Anomaly Detection in Mixed Medical Data

This is the final project for the Unsupervised Learning course at Unimib.
This is a group project developed along with my uni husband @MirkoMorello.

## Summary

### Objective

Explore and implement unsupervised anomaly detection methods on a high-dimensional medical dataset composed of mixed-type features (numerical and categorical).

### Approach

The dataset contained 7,200 observations and 23 features (reduced to 21 after cleaning), including both numerical and one-hot encoded categorical features.

Given the mixed nature of the data, we computed pairwise dissimilarities using **Gower's distance**, combining Euclidean (for numerical) and Hamming (for categorical) distances. We also implemented a fast custom version of Gower's distance.

Dimensionality reduction via **PCA** and **t-SNE** was used for qualitative inspection and validation.

We explored three main families of anomaly detection methods:

---

### Proximity-Based Methods

**Assumption**: Anomalies are distant from or isolated compared to their neighbors.

* **k-Nearest Neighbors (k-NN)**: Used the 5th nearest neighbor distance as the anomaly score.
  ⟶ Detected \~9.6% anomalies, performed well.

* **Local Outlier Factor (LOF)**: Struggled due to high dimensionality (curse of dimensionality).
  ⟶ Many false positives near dense regions.

* **Connectivity Outlier Factor (COF)**: Graph-based LOF variant, more robust in high dimensions.
  ⟶ Detected \~14.3% anomalies, decent results.

* **DBSCAN**: Clustered dense regions, marked low-density points as noise.
  ⟶ Detected \~5.9% anomalies with good qualitative results.

---

### Prototype-Based Methods

**Assumption**: Anomalies are far from their cluster’s centroid.

* **K-Means++**: Required reimplementation to support Gower distance and preserve categorical information.
  ⟶ Used modified centroid update rules for booleans.

* **Cluster count selection**: Used a second-derivative-based elbow method across multiple runs to determine optimal `k ≈ 5–6`.

* **Anomaly Score**: Distance from point to cluster centroid.
  ⟶ Detected \~4.2% anomalies. Results were solid and interpretable.

---

### Reconstruction-Based Methods

**Assumption**: Normal points are well-reconstructed; anomalies yield high reconstruction error.

* **PCA**: Reconstructed data with 6 components.
  ⟶ Only \~1.1% anomalies detected. Many were false positives.

* **Autoencoder**: 1 hidden layer, trained to reconstruct the input.
  ⟶ Detected \~3.9% anomalies. Performed comparably to k-NN.

---

### Ensembles

Combined top-performing methods (k-NN, K-Means, Autoencoder, DBSCAN) using:

* **AND**: Strictest ensemble. Detected 2.3% anomalies.
* **OR**: Most inclusive. Detected 11% anomalies.
* **Weighted Sum**: Normalized scores from k-NN, K-Means, and Autoencoder.
  ⟶ Detected \~5.6% anomalies with the best balance between precision and recall.

---

### Coherence

Measured agreement between methods using Adjusted Rand Index.
Highest agreement between K-Means, k-NN, and Autoencoder — validating our ensemble choice.

---

See the report for visualizations, further analysis, and detailed methodology.