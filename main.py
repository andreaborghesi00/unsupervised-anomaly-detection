# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import threading
import queue
import concurrent.futures
import gc
from sklearn.neighbors import NearestNeighbors
from pythresh.thresholds.dsn import DSN
import gower
from torchsummary import summary


from itertools import combinations

# %%

# Load the data
df = pd.read_csv('datasets/data.csv', sep=';', index_col='Row')
df.head(3)


# %% [markdown]
# ----
# # <center> Utility and Visualization functions

# %%
def PCA_tSNE_visualization(data2visualize, NCOMP, LABELS, PAL='viridis', title_addition='', legend=None, show_title=False):

  '''
  INPUT
  data2visualize    - data matrix to visualize
  NCOMP             - no. of components to decompose the dataset during PCA
  LABELS            - labels given by the clustering solution
  PAL               - palette of colours to distinguish between clusters
  '''

  '''
  OUTPUT
  Two figures: one using PCA and one using tSNE
  '''


  # PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components=NCOMP)
  pca_result = pca.fit_transform(data2visualize)
  print('PCA: explained variation per principal component: {}'.format(pca.explained_variance_ratio_.round(2)))

  # tSNE
  from sklearn.manifold import TSNE
  print('\nApplying tSNE...')
  np.random.seed(100)
  tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=300)
  tsne_results = tsne.fit_transform(data2visualize)


  # Plots
  fig1000 = plt.figure(figsize=(10,5))
  if show_title: fig1000.suptitle(f'Reduced dataset - {title_addition}', fontsize=16)


  # Plot 1: 2D image of the entire dataset
  ax1 = fig1000.add_subplot(121)
  sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], ax=ax1, hue=LABELS, palette=PAL)
  ax1.set_xlabel('Dimension 1', fontsize=10)
  ax1.set_ylabel('Dimension 2', fontsize=10)
  ax1.title.set_text('PCA')
  if legend is not None:
    ax1.legend(legend)
  plt.grid()

  ax2= fig1000.add_subplot(122)
  sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], ax=ax2, hue=LABELS, palette=PAL)
  ax2.set_xlabel('Dimension 1', fontsize=10)
  ax2.set_ylabel('Dimension 2', fontsize=10)
  ax2.title.set_text('tSNE')
  if legend is not None:
    ax2.legend(legend)
  plt.grid()
  plt.show()


# %%
def plot_float_comb_dimensions(df, labels, palette, legend=None):
    """
    Plot the combination of all float columns of the dataset with the specified labels.

    Parameters:
    df (pandas DataFrame): The input dataset.
    labels (array-like): The labels of the dataset.
    palette (list): The colors to use for the labels.
    """
    float_cols = df.select_dtypes(exclude=['bool']).columns
    df_float = df[float_cols]

    float_comb = combinations(range(df_float.shape[1]), 2)
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot(111)
    for i, (feat1, feat2) in enumerate(float_comb):
        ax = fig.add_subplot(3, 5, i+1)
        sns.scatterplot(x = df_float[float_cols[feat1]], y = df_float[float_cols[feat2]], hue=labels, palette=palette)
        ax.set_xlabel(f'Dimension {feat1}')
        ax.set_ylabel(f'Dimension {feat2}')
        if legend is not None:
            ax.legend(legend)
    plt.show()
    


# %%
def plot_kj_dimension(df, labels, feat1, feat2, palette):
    """
    Plot the k-th dimension against the j-th dimension of the dataset with the specified labels.

    Parameters:
    df (pandas DataFrame): The input dataset.
    labels (array-like): The labels of the dataset.
    k (int): The dimension to plot.
    palette (list): The colors to use for the labels.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.scatterplot(x = df[df.columns[feat1]], y = df[df.columns[feat2]], hue=labels, palette=palette)
    ax.set_xlabel(f'Dimension {feat1}')
    ax.set_ylabel(f'Dimension {feat2}')
    ax.legend()
    plt.show()



# %%
def iqr_bound(scores):
    """
    Calculate the upper bound for outliers using the interquartile range (IQR) method.

    Parameters:
    scores (array-like): An array-like object containing the scores.

    Returns:
    float: The upper bound for outliers.

    """
    q3 = np.quantile(scores, 0.75)
    q1 = np.quantile(scores, 0.25)
    iqr = q3 - q1

    return q3 + 1.5*iqr

def two_stage_iqr_bound(scores):
    """
    Calculate the two-stage IQR bound for a given array of scores.

    Parameters:
    scores (array-like): An array of scores.

    Returns:
    float: The two-stage IQR bound.

    """
    t1_bound = iqr_bound(scores)
    t1_survivors = np.where(scores <= t1_bound)[0]

    t2_bound = iqr_bound(scores[t1_survivors])

    return t2_bound


# %% [markdown]
# # Data cleaning

# %%
# change the data type of the columns to float
df = df.replace(',', '.', regex=True)
df.drop(['Unnamed: 22', 'Unnamed: 23'], axis=1, inplace=True)
df.head(3)

# %%
# convert 0-1 columns to boolean
int_cols = df.select_dtypes(include=['int64']).columns
# convert to bool if the column has only 0 and 1
for col in int_cols:
    if df[col].nunique() == 2:
        df[col] = df[col].astype(bool)
df.dtypes

# %%
bool_cols = df.select_dtypes(include=['bool']).columns
df.head(3)


# %%
# take columns that are not boolean
float_cols = df.select_dtypes(exclude=['bool']).columns
df[float_cols] = df[float_cols].astype(float)
df.dtypes

# %%
df.describe()


# %%
# check for missing values, none are found
df.isnull().sum()

# %%
# scaling of float columns
scaler = StandardScaler()
df[float_cols] = scaler.fit_transform(df[float_cols])

# %%
# as expected the mean is (numerically) 0 and the standard deviation is (numerically) 1
df.describe()

# %%
# PCA & tSNE floats visualization
PCA_tSNE_visualization(df[float_cols], 2, np.ones(df.shape[0]), 'viridis')

# %%
# PCA & tSNE bools visualization
PCA_tSNE_visualization(df[bool_cols], 2, np.ones(df.shape[0]), 'viridis')

# %%
# the whole thing
PCA_tSNE_visualization(df, 2, np.ones(df.shape[0]), 'viridis')


# %% [markdown]
# ----
# # <center>Anomaly detection
# ----

# %% [markdown]
# ## <center>Proximity Based

# %% [markdown]
# #### <center>Proximity Matrix for mixed data-types

# %% [markdown]
# We do not consider weights as for our implementation we will maintain uniform weights, so we do not bother to implement them.

# %%
def proximity_matrix(data_x, data_y=None, metrics={'numeric': 'euclidean', 'categorical': 'hamming'}, cat_features=[]):
    """
    Computes the proximity matrix for a given dataset(s).

    Parameters:
    data_x (pandas DataFrame): The input dataset.
    data_y (pandas DataFrame): The dataset to compare with. If None, it will be the same as data_x.
    metrics (dict): A dictionary containing the metrics to use for the numeric and categorical features. 
    It should have the following structure:
        ```
        {
            'numeric': 'euclidean',
            'categorical': 'hamming'
        }
        ```
        Available metrics are: 'euclidean', 'manhattan', 'hamming', 'cosine'
    cat_features (list): A list containing the names of the categorical features.

    Returns:
    prox_matrix (numpy array): The proximity matrix, where each element represents the proximity between two data points.
    """
    if data_y is None: data_y = data_x

    fun_metric = {
        'euclidean': lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=1)),
        'manhattan': lambda x, y: np.sum(np.abs(x - y), axis=1),
        'hamming': lambda x, y: np.sum(x != y, axis=1),
        'cosine': lambda x, y: np.sum(x * y, axis=1) / (np.sqrt(np.sum(x ** 2, axis=1)) * np.sqrt(np.sum(y ** 2, axis=1)))
    }

    X = data_x.values
    Y = data_y.values

    X_num = X[:, [i for i, col in enumerate(data_x.columns) if col not in cat_features]].astype(float)
    X_cat = X[:, [i for i, col in enumerate(data_x.columns) if col in cat_features]]
    Y_num = Y[:, [i for i, col in enumerate(data_y.columns) if col not in cat_features]].astype(float)
    Y_cat = Y[:, [i for i, col in enumerate(data_y.columns) if col in cat_features]]

    X_num = np.array(X_num)
    X_cat = np.array(X_cat)

    metric_num = fun_metric[metrics['numeric']]
    metric_cat = fun_metric[metrics['categorical']]

    prox_matrix = np.zeros((data_x.shape[0], data_y.shape[0]))
    for i in tqdm(range(data_x.shape[0]), desc='Computing proximity matrix'):
        x_num = X_num[i, :]
        x_cat = X_cat[i, :]

        num_dist = metric_num(x_num, Y_num)
        cat_dist = metric_cat(x_cat, Y_cat)

        prox_matrix[i,:] = (num_dist + cat_dist) / (X_num.shape[1] + X_cat.shape[1])

    return prox_matrix


# %%
def proximity_matrix_diag(data_x, data_y=None, metrics={'numeric': 'euclidean', 'categorical': 'hamming'}, cat_features=[]):
    """
    Computes the diagonal only of the proximity matrix for a given dataset(s).
    This is useful when we only need the proximity of the data points with themselves, or when we want to compare the proximity of the data points with themselves with the proximity of the data points with other data points.

    Parameters:
    data_x (pandas DataFrame): The input dataset.
    data_y (pandas DataFrame): The dataset to compare with. If None, it will be the same as data_x.
    metrics (dict): A dictionary containing the metrics to use for the numeric and categorical features. 
    It should have the following structure:
        ```
        {
            'numeric': 'euclidean',
            'categorical': 'hamming'
        }
        ```
        Available metrics are: 'euclidean', 'manhattan', 'hamming', 'cosine'
    cat_features (list): A list containing the names of the categorical features.

    Returns:
    prox_matrix (numpy array): The proximity matrix, where each element represents the proximity between two data points.
    """
    if data_y is None: data_y = data_x

    fun_metric = {
        'euclidean': lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'hamming': lambda x, y: np.sum(x != y),
        'cosine': lambda x, y: np.sum(x * y) / (np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2)))
    }

    X = data_x.values
    Y = data_y.values

    X_num = X[:, [i for i, col in enumerate(data_x.columns) if col not in cat_features]].astype(float)
    X_cat = X[:, [i for i, col in enumerate(data_x.columns) if col in cat_features]]
    Y_num = Y[:, [i for i, col in enumerate(data_y.columns) if col not in cat_features]].astype(float)
    Y_cat = Y[:, [i for i, col in enumerate(data_y.columns) if col in cat_features]]

    X_num = np.array(X_num)
    X_cat = np.array(X_cat)

    metric_num = fun_metric[metrics['numeric']]
    metric_cat = fun_metric[metrics['categorical']]

    prox_matrix = np.zeros((data_x.shape[0], data_y.shape[0]))
    for i in range(data_x.shape[0]):
        x_num = X_num[i, :]
        x_cat = X_cat[i, :]
        y_num = Y_num[i, :]
        y_cat = Y_cat[i, :]

        num_dist = metric_num(x_num, y_num)
        cat_dist = metric_cat(x_cat, y_cat)

        prox_matrix[i,:] = (num_dist + cat_dist) / (X_num.shape[1] + X_cat.shape[1])

    return prox_matrix


# %% [markdown]
# Since there is no information available regarding the semantic of the features, all weights are set to one.

# %%
bool_cols_idx = [df.columns.get_loc(col) for col in bool_cols]
cols = np.full(df.shape[1], False, dtype=bool)
cols[bool_cols_idx] = True

prox_mat = proximity_matrix(data_x=df, cat_features=bool_cols)

# %%
prox_mat_diag = proximity_matrix_diag(data_x=df, cat_features=bool_cols)

# %%
np.allclose(np.diag(prox_mat), prox_mat_diag)

# %%
# symmetry check
print(np.allclose(prox_mat, prox_mat.T))
# check zero diagonal
print(np.allclose(np.diag(prox_mat), 0))

# %%
plt.style.use('default')

N, M = prox_mat.shape
fig1 = plt.figure(figsize=(20,15))

# Plot 2: proximity matrix
plt.imshow(prox_mat, interpolation='nearest', aspect='auto', cmap='viridis')
plt.colorbar()

plt.xlabel('Observations', fontsize=16)
plt.xticks(np.arange(0, N, step=1000))
plt.ylabel('Observations', fontsize=16)
plt.yticks(np.arange(0, N, step=1000))
plt.title('Proximity matrix')

plt.show()

# %% [markdown]
# ----
# ### <center>Distance Based: NN Approach

# %% [markdown]
# The k-Nearest Neighbour (k-NN) algorithm for distance-based anomaly detection relies on the principle that anomalous data objects exhibit significantly larger distances to their k-nearest neighbours compared to ordinary data objects. By exploiting this property, the algorithm can effectively identify anomalies within the dataset.
# To apply this approach, we compute the distance between each data object and its fifth nearest neighbour and then exclude the ones that have it the furthest.

# %%
k = 5 # number of neighbors
knn = NearestNeighbors(n_neighbors=k-1, metric='precomputed') # if we query the same points then the first one will be the point itself and ignored by default, so to get k=5 we need to set k=4
knn.fit(prox_mat);
dist, idx= knn.kneighbors()
knn_score = dist[:, -1]

# %%
plt.hist(dist[:, -1], bins=100, label='Distance')
# t1_bound = iqr_bound(knn_score)
# t2_bound = two_stage_iqr_bound(knn_score)
# plt.axvline(t1_bound, color='orange', linestyle='dashed', linewidth=1, label='Threshold: Stage 1')
# plt.axvline(t2_bound, color='red', linestyle='dashed', linewidth=1, label='Threshold: Stage 2')
# plt.axvline(t1_bound, color='orange', linestyle='dashed', linewidth=1, label='Threshold')

plt.title('Distance to 5th nearest neighbor')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.xlim(0, .6)
# plt.legend()
plt.show()

# %%
sorted_dist_idx = np.argsort(knn_score)[::-1]
print(*dist[sorted_dist_idx, -1])

# %% [markdown]
# For this first method we will show different way of thresholding, while for the following ones we will stick just to one for consistency.
#
# The first methods assumes the percentage of anomalous data objects within the dataset, for instance 5%

# %%
anomaly_perc = 0.05
n_anomalies = np.round(anomaly_perc*df.shape[0])

anomalies = sorted_dist_idx[:int(n_anomalies)]
dist_sorted = dist[sorted_dist_idx, -1]
anomalies.shape

# %%
# anomalies visualization
NN_labels = np.ones(df.shape[0])
NN_labels[anomalies] = -1
PCA_tSNE_visualization(df, 2, NN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='NN')
plot_float_comb_dimensions(df, NN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# This second one uses the knee method and a knee locator object from the kneed library to automatically decide where is the knee/elbow point

# %%
from kneed import KneeLocator
knee = KneeLocator(range(N), dist_sorted, S=1, curve='convex', direction='decreasing', interp_method='polynomial', online=True)     # see other examples: https://kneed.readthedocs.io/en/stable/parameters.html
'''
S - The sensitivity parameter allows us to adjust how aggressive we want Kneedle to be when detecting knees.
    Smaller values for S detect knees quicker, while larger values are more conservative.
    Put simply, S is a measure of how many “flat” points we expect to see in the unmodified data curve before declaring a knee.
'''
knee_x = knee.knee
knee_y = knee.knee_y    # OR: distances[knee.knee]

print([knee_x, np.round(knee_y,2)])

# Plot distances
fig3 = plt.figure(figsize=(18,2))

ax1 = fig3.add_subplot(121)
plt.plot(dist[:,-1])
ax1.set_xlabel('Data points', fontsize=10)
ax1.set_xticks(np.arange(0, N, step=1000))
ax1.set_ylabel('Distances\n(not sorted, gower distance)', fontsize=10)
# ax1.title.set_text('Proximity matrix (%s distance)' % distance_metric)
plt.grid()

ax2 = fig3.add_subplot(122)
plt.plot(dist_sorted, 'o-')
ax2.set_xlabel('Data points', fontsize=10)
ax2.set_ylabel('Distances (sorted)', fontsize=10)
plt.axvline(x=knee_x, color='k', linestyle='--')
plt.axhline(y=knee_y, color='k', linestyle='--')
plt.plot((knee_x), (knee_y), 'o', color='r')
plt.grid()

plt.show()

# %%
knee_outliers_idx = np.where(dist[:, -1] > knee_y)[0]
print(f'Number of outliers: {len(knee_outliers_idx)}')
knee_labels = np.ones(N)
knee_labels[knee_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, knee_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='NN')
plot_float_comb_dimensions(df, knee_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# The third ones uses the Bhattcharyya distance, comparing the distribution of the anomaly score (the distance from the fifth nearest neighbour) to a gaussian distribution

# %%
thres = DSN(metric='BHT')
dsn_knn_labels = thres.eval(knn_score)

print(len(np.where(dsn_knn_labels == 1)[0]))

PCA_tSNE_visualization(df, 2, dsn_knn_labels, ['gray', 'red'], legend=['Anomalous', 'Normal'], title_addition='DSN')
plot_float_comb_dimensions(df, dsn_knn_labels, ['gray', 'red'], legend=['Anomalous', 'Normal'])

# %% [markdown]
# This last one is a two-step approach using a classic IQR thresholding. Two-step methods are more effective as due to their nature of recomputing the bound once the most apparent anomalies are removed, letting more nuanced anomalies to be found.
#
# For the subsequent anomaly detection methods we will settle for this approach of thresholding.

# %%
t2_bound = two_stage_iqr_bound(knn_score)

t2_outliers_idx = np.where(knn_score > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
NN_labels = np.ones(N)
NN_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, NN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='NN')
plot_float_comb_dimensions(df, NN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# It seems to work best when k is low, but in any case the results don't seem that great. This might be because we're dealing with a high number of dimensions with respect to the number of samples available.

# %% [markdown]
# ----
# ### <center>Density Based: LOF

# %% [markdown]
# In this context, applying a density based approach is quite a risk, as density-based approaches tend to be very sensitive to high-dimensionality, significantly more than distance based ones. This is because density based approaches rely on accurate local distance measurements which tends to lose information as the data becomes more sparse as the dimensionality rises.

# %%
# Apply the algorithm
from sklearn.neighbors import LocalOutlierFactor

lof_model  = LocalOutlierFactor(n_neighbors  = 5,
                                metric='precomputed',
                                contamination = 0.05)

LOF_labels = lof_model.fit_predict(prox_mat)     # predict the labels (1 inlier, -1 outlier) of X according to LOF
LOF_values     = lof_model.negative_outlier_factor_
np.where(LOF_labels == -1)[0].shape

# %%
np.min(LOF_values), np.max(LOF_values)

# %%
plt.hist(LOF_values, bins=100)
plt.title('Local Outlier Factor')
plt.xlabel('LOF')
plt.ylabel('Frequency')
plt.show()

# %%
plt.plot(LOF_values)
plt.title('Local Outlier Factor')
plt.xlabel('Data points')
plt.ylabel('LOF')
plt.show()

# %%
t2_bound = two_stage_iqr_bound(-LOF_values)

t2_outliers_idx = np.where(-LOF_values > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
LOF_labels = np.ones(N)
LOF_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, LOF_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='LOF')
plot_float_comb_dimensions(df, LOF_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# ----
# ### <center>DBSCAN

# %% [markdown]
# While not explicitly designed for anomaly detection, the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm can be adapted for this purpose. We are not necessarily interested in the clusters formed, but rather those data object marked as "noise points" that are removed during the clustering procedure. As we may notice, this method does not allow for any thresholding as the implementation of sklearn that we're going to use directly a label (-1) for noise data objects.

# %%
# dbscan
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.075, min_samples=5, metric='precomputed')
DBSCAN_labels = dbscan.fit_predict(prox_mat)
DBSCAN_labels[np.where(DBSCAN_labels >= 0)[0]] = 1
print(f'Number of outliers: {np.where(DBSCAN_labels == -1)[0].shape[0]}, {np.where(DBSCAN_labels == -1)[0].shape[0]/N*100:.2f}%')

PCA_tSNE_visualization(df, 2, DBSCAN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='DBSCAN')

plot_float_comb_dimensions(df, DBSCAN_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# ----
# ### <center>Graph Based: COF
# COF is designed to identify outliers based on the connectivity structure, which can be more robust in high-dimensional spaces or in datasets with complex structures where traditional distance-based methods might struggle. By leveraging graph theory, COF can capture more nuanced relationships between points that pure distance metrics might miss.

# %%
near_neigh = NearestNeighbors(n_neighbors=5, metric='precomputed')
near_neigh.fit(prox_mat)
dist, idx = near_neigh.kneighbors()

# %%
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

adjacency_matrix = np.full((N, N), np.inf)
np.fill_diagonal(adjacency_matrix, 0) # diagonal is 0
for i, neighbour in enumerate(idx):
        adjacency_matrix[i, neighbour] = dist[i, np.where(idx[i] == neighbour)]
adjacency_matrix[adjacency_matrix == np.inf] = 1000 # ensures connectivity for the shortest path algorithm

# %%
shortest_paths = shortest_path(adjacency_matrix, directed=False) # it takes around 9 minutes on a i9-9900K

# %%
# check if there are inf values
np.any(np.isinf(shortest_paths))

# %%
avg_shortest_path = np.mean(shortest_paths, axis=1)
print(*avg_shortest_path)


# %%
# plt.hist(x=range(7200), y=avg_shortest_path, bins=1000)
# plt.show()

# %%
def compute_cof(avg_shortest_paths, indices):
    cof_scores = np.zeros_like(avg_shortest_paths)
    for i in range(len(avg_shortest_paths)):
        neighbors = indices[i]
        neighbor_avg_paths = avg_shortest_paths[neighbors]
        cof_scores[i] = avg_shortest_paths[i] / np.mean(neighbor_avg_paths)
    return cof_scores


# %%
cof_scores = compute_cof(avg_shortest_path, idx)
t2_bound = two_stage_iqr_bound(cof_scores)


# %%
np.max(cof_scores)

# %%
plt.hist(cof_scores, bins=100)
plt.title('COF scores')
# plt.axvline(t2_bound, color='red', linestyle='dashed', linewidth=1, label='Threshold')
plt.xlabel('COF')
plt.ylabel('Frequency')
plt.show()

# %%
t2_bound = two_stage_iqr_bound(cof_scores)

t2_outliers_idx = np.where(cof_scores > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}', f'{len(t2_outliers_idx)/N*100:.2f}%')
COF_labels = np.ones(N)
COF_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, COF_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='COF')
plot_float_comb_dimensions(df, COF_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])


# %% [markdown]
# ----
# ## <center>Clustering based
# ----

# %% [markdown]
# ### Prototype based clusters: K-Means++
# Since we will be using K-means++, to measure the distance between the datapoints and the centroid, by definition it is used the euclidean distance. Hence each boolean column will be interpreted as a float. Although this is a common practice, such a cast leads to losing information. This is why we may call this first approach Naive, as we allow ourselves to loose such knowledge.
#
# **But how would we approach this problem otherwise?**<br>
# By rededfining the centroid structure, the distance and the update of the centroid.<br>
#
# __Centroid structure__<br>
# Instead of using a cluster defined as a homogeneous array of floats, we will be using an heterogenous one, such that the "representative" for a boolean feature is a boolean, and the "representative" for a float is a float.
#
# __Distance__<br>
# For the distance, we will be using the same approach used for computing the proximity matrix in the previous sections. This type of distance is also known as __Gower distance__ with uniform weights.<br>
#
# __Centroid update__<br>
# After that we can properly assign each datapoint to its closest centroid and compute the next iteration's centroids.
# The classic approach wants to assign to the next iteration centroid the average value of each datapoint of a cluster. 
# In our version we keep computing the average for the float features, and the mode for the booleans.
#

# %% [markdown]
# Now we want to pinpoint a few observations that arise from using the __hamming distance__ for boolean features and the __mode__ while updating the centroid.
#
# Computing the mode for boolean feature consists in checking the "label" that has the highest frequency, and a shortcut can be found by considering True as 1 and False as 0 then
# $$
#     \forall i \in \mathrm{BooleanFeatures} \quad \frac{1}{\vert \mathrm{data} \vert}\sum_{\mathrm{point}\in\mathrm{data}} \mathrm{point}_i = \begin{cases} \mathrm{True} & \mathrm{if }> 0.5\\ \mathrm{False}&\mathrm{else}\end{cases}
# $$
#
#

# %% [markdown]
# So if the boolean features were to be considered as floats, updating the centroid wouldn't be loosing any information if we were to round to the nearest integer after computing such average.
#
# The next observation regards the the hamming distance $h$ and any dissimilarity function $f$ such that $f(x, y) = 0 \Leftrightarrow x=y$, that is satisfied by any Minkowski distance. Now we notice 
# $$
#     h(x, y) = \begin{cases}1&\mathrm{if}\; x \ne y\\ 0&\mathrm{if}\; x=y\end{cases} \\
#     \forall x,y\in\{0,1\}\quad f(x, y) = \begin{cases}1&\mathrm{if}\; x \ne y\\ 0&\mathrm{if}\; x=y\end{cases}
# $$ 

# %% [markdown]
# ### K-Means with gower distance implementation(s)
# Since sklearn does not offer the possibility of running the kmeans++ algorithm with a precomputed proximity matrix as we did for the NN anomaly detection, we decide to implement it.<br>
# The downside of reimplementing such an algorithm is efficiency. Sklearn does an exceptional job at providing extremely optimized implementations of many algorithms, one of which being Kmeans++, by implementing it in Cython, a superset of Python that allows for the inclusion of C/C++ code.<br>
# Since optimizing algorithms with such tools is out of the scope of this project, our implementation will stick to pure python. We will nonetheless try to optimize it with the tools that we have. <br><br>
# We first propose a naïve implementation of such (` kmeans_gower `), where the above stated considerations are not taken into account, and then a second implementation (` kmeans_gower_revisited `) that shows how simple mathematical observation can drastically increase the performances.

# %%
def kmeans_gower(data, n_clusters,metrics=None, weights=None, max_iter=300, random_state=None):

    if metrics is None:
        metrics = {np.bool_: 'hamming', np.float64: 'euclidean', float: 'euclidean', bool: 'hamming'}

    # Initialize centroids using k-means++ initialization
    centroids = data.sample(n_clusters, random_state=random_state).values
    bool_indices = [data.columns.get_loc(col) for col in bool_cols]
    float_indices = [data.columns.get_loc(col) for col in float_cols]
    data = data.values

    print(data.shape)
    for _ in tqdm(range(max_iter)):
        # Compute distances from each data point to centroids
        distances = np.array([np.array([gower_distance(point, centroid, metrics, weights) for centroid in centroids]) for point in data])
        
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy() # i need a placeholder and i don't know how to do it
        for k in range(n_clusters):
            cluster_data = data[labels == k]
            len_cluster_data = len(cluster_data)
            new_centroids[k][bool_indices] = np.sum(cluster_data[:, bool_indices], axis=0) / len_cluster_data > .5 # homemade mode
            new_centroids[k][float_indices] = np.mean(cluster_data[:, float_indices], axis=0)

        # Check for convergence, i fucking hate this loosly typed shit
        # if np.allclose(new_centroids[:, float_indices], centroids[:, float_indices]) and (new_centroids[:, bool_indices] == centroids[:, bool_indices]):
        #     break

        centroids = new_centroids
    inertia = np.sum([gower_distance(data[i], centroids[labels[i]], metrics, weights) for i in range (data.shape[0])])
    
    return labels, centroids, inertia



# %%
def kmeans_gower_optimized(data, n_clusters, max_iter=300, random_state=None, keep_types=False, result_queue=None):
    """
    Perform k-means clustering using the Gower distance metric.

    Parameters:
    - data: The input data for clustering.
    - n_clusters: The number of clusters to create.
    - max_iter: The maximum number of iterations for the k-means algorithm.
    - random_state: The random seed for centroid initialization.
    - keep_types: Whether to keep the original data types of the centroids.
    - result_queue: A queue to store the clustering results.

    Returns:
    - labels: The cluster labels for each data point.
    - centroids: The final centroid positions.
    - inertia: The sum of squared distances between each data point and its nearest centroid.
    """
    template = data.sample(n_clusters, random_state=random_state).values
    centroids = data.sample(n_clusters, random_state=random_state).values.astype(float)
    bool_indices = [data.columns.get_loc(col) for col in bool_cols]
    data = data.values.astype(float)

    centroids_prev = np.zeros_like(centroids)  # Initialize centroids_prev

    for _ in range(max_iter):
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2)) # the 
        labels = np.argmin(distances, axis=1)

        for k in range(n_clusters):
            centroids[k] = np.mean(data[labels == k], axis=0)

        centroids[:, bool_indices] = np.round(centroids[:, bool_indices])

        if np.allclose(centroids, centroids_prev):
            break

        centroids_prev = centroids.copy()

    inertia = np.sum((data - centroids[labels]) ** 2)

    if keep_types:
        template[:,bool_indices] = centroids[:,bool_indices] > .5
        template[:,not bool_indices] = centroids[:,not bool_indices]
        centroids = template
    if result_queue is not None:
        result_queue.put((labels, centroids, inertia))

    return labels, centroids, inertia



# %% [markdown]
# Here we show how the two implementation perform drastically different while computing the same results.

# %%
# %%time
l, c, i = kmeans_gower(df, 10, max_iter=10)

# %%
# %%time
lr, cr, ir =kmeans_gower_optimized(df, 10, max_iter=10)

# %%
# %%time
KMeans(n_clusters=10, random_state=0).fit(df);

# %%
# kmeans++ clustering
nk = 4
kmeans = KMeans(n_clusters=nk, init='k-means++', max_iter=1000, ).fit(df)
labels = kmeans.labels_
# kmeans++ visualization
PCA_tSNE_visualization(df, 4, lr, 'viridis')
plot_float_comb_dimensions(df, lr, 'viridis')

# %% [markdown]
# #### Finding the optimal number of clusters: Elbow method

# %% [markdown]
# The first step to run prototype-based anomaly detection with Kmeans++ is to find the optimal number of centroids. We will do so by using the elbow method.
# To determine the optimal number of clusters with the elbow method we usually look for the "elbow" in the analyzed function, in our case a function having on the x-axis the number of clusters and on the y-axis the inertia (or intra-cluster distance).<br>
# If we run the cells below we can notice how the elbow tends to be somewhat shallow, as it doesn't really mark a very steep elbow. So we decide to make a few observations, trying to define where is this "elbow point".<br><br>
# By definition, the function taken into account with the elbow method is a decreasing function (although some fluctuations are present), and the "elbow" is the point where the function stops decreasing drastically. This behaviour can be analyzed by looking at the second derivative of such function. We can define the elbow method as the minima of the second derivative, the point where the function stops decreasing drastically.<br><br>
# Although such definition tends to identify the elbow point very early, we are fine with it, as by looking at the datset we can guess that the number of clusters is probably below 15.
# This definition has also a second shortcoming, if the function decreases slowly and it has fluctuations, it can propose false positives.<br>
# As a matter of fact, the cells below will show how the elbow point tends to fluctuate between 4 and 9. So we take a statistical approach by running this computation for a few hundreds time and observe the frequencies of the proposed optimal number of clusters in those runs.

# %% [markdown]
# For the sake of the project we will run the process twice, one using the sklearn Kmeans++ implementation, that does not take into account the mixed data types, and one with our Kmeans++ implementation. The number of runs is as high as needed to obtain a consistent result.<br><br>

# %%
# Single run of the elbow method with sklearn's Kmeans++
inertia = []
r = range(1,20)
for k in r:
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(df)
    inertia.append(kmeans.inertia_)

#derivative
first_derivative = np.diff(inertia)
second_derivative = np.diff(first_derivative)

# reasoning for the min of second derivative:
# the first derivative is the slope of the inertia, while the second derivative is the acceleration of the inertia
# the "elbow" represents where the inertia starts to decrease at a slower rate, i.e. where the acceleration is the smallest
optimal_k = np.argmin(second_derivative) + 2 + 1 # +1 because we start from 1 clusters and +2 because each derivative is 1 element shorter than the previous one

plt.plot(r[1:], first_derivative, marker='o', color='r', label='first derivative', linestyle='--', alpha=.6)
plt.plot(r[2:], second_derivative, marker='o', color='g', label='second derivative', linestyle='--', alpha=.6)
plt.plot(r, inertia, marker='o', color='b', label='inertia', alpha=.6)
plt.axvline(x=optimal_k, color='black', linestyle='dotted', label='optimal number of clusters: {}'.format(optimal_k))

# plt.axvline(x=???, color='g', linestyle='dotted', label='optimal number of clusters')
plt.xticks(r)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()
plt.show()

# %%

# %%
# single run of the elbow method with the custom kmeans
inertia = []
r = range(1,16)
res_queue = queue.Queue()
threads = []

for k in r:
    t = threading.Thread(target=kmeans_gower_optimized, args=(df, k, 300, None, False, res_queue))
    threads.append(t)
    if len(threads) % 8 == 0:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        while not res_queue.empty():
            l, c, i = res_queue.get()
            inertia.append(i)
        threads = []
        gc.collect()

# execute the remaining threads
for t in threads:
    t.start()
for t in threads:
    t.join()
while not res_queue.empty():
    l, c, i = res_queue.get()
    inertia.append(i)

    # l, c, i = kmeans_gower_revisited(df, k, max_iter=100)
    # inertia.append(i)

#derivative
first_derivative = np.diff(inertia)
second_derivative = np.diff(first_derivative)

# reasoning for the min of second derivative:
# the first derivative is the slope of the inertia, while the second derivative is the acceleration of the inertia
# the "elbow" represents where the inertia starts to decrease at a slower rate, i.e. where the acceleration is the smallest
optimal_k = np.argmin(second_derivative) + 2 + 1 # +2 because we start from 2 clusters and +2 because each derivative is 1 element shorter than the previous one

plt.plot(r[1:], first_derivative, marker='o', color='r', label='first derivative', linestyle='--', alpha=.6)
plt.plot(r[2:], second_derivative, marker='o', color='g', label='second derivative', linestyle='--', alpha=.6)
plt.plot(r, inertia, marker='o', color='b', label='inertia', alpha=.6)
plt.axvline(x=optimal_k, color='black', linestyle='dotted', label='optimal number of clusters: {}'.format(optimal_k))

# plt.axvline(x=???, color='g', linestyle='dotted', label='optimal number of clusters')
plt.xticks(r)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()
plt.show()


# %%
def elbow_method_run(k_range, result_queue):
    """
    Runs the elbow method to determine the optimal number of clusters (k) for K-means clustering.

    Parameters:
    k_range (list): A list of integers representing the range of k values to consider.
    result_queue (Queue): A queue to store the optimal k value.

    Returns:
    None, results are stored in result_queue.
    """
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++').fit(df)
        inertia.append(kmeans.inertia_)

    # derivative
    first_derivative = np.diff(inertia)
    second_derivative = np.diff(first_derivative)
    optimal_k = np.argmin(second_derivative) + 1 + 2

    result_queue.put(optimal_k)


# %%
def elbow_method_run_gower_optimized(data, k_range, result_queue, max_iter=100, max_workers=8):
    """
    Runs the elbow method using the Gower distance metric to determine the optimal number of clusters.

    Args:
        k_range (list): A list of integers representing the range of values for k (number of clusters).
        result_queue (Queue): A queue to store the result of the optimal k value.

    Returns:
        None, results are stored in the result_queue.
    """
    inertia = []
    local_queue = queue.Queue()
    threads = []
    
    for k in k_range:
        t = threading.Thread(target=kmeans_gower_optimized, args=(data, k, max_iter, None, False, local_queue))
        threads.append(t)

        if len(threads) % max_workers == 0:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            while not local_queue.empty():
                l, c, i = local_queue.get()
                inertia.append(i)
            threads = []

    # execute the remaining threads
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not local_queue.empty():
        l, c, i = local_queue.get()
        inertia.append(i)

    # derivative
    first_derivative = np.diff(inertia)
    second_derivative = np.diff(first_derivative)
    optimal_k = np.argmin(second_derivative) + 2 + 1

    result_queue.put(optimal_k)


# %% [markdown]
# **Disclaimer**<br>
# As mentioned before, our implementation is not as efficient as the sklearn one. The following cells can take hours to compute, as we run our non-optimized code for hundreds of times. We will nonetheless use multithreading to cut significantly computational time by 2-4 times depending on the CPU

# %%
# For sklearn's kmeans: 500 runs with 20 clusters takes around 2 minutes (i9-9900K)

max_k = 11
optimal_ks = np.zeros(max_k)
runs = 500
threads = []
results = queue.Queue()

# elbow method
for run in tqdm(range(runs)):
    t = threading.Thread(target=elbow_method_run, args=(range(1, max_k), results))
    threads.append(t)
    if len(threads) % 8 == 0:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            optimal_ks[results.get()] += 1
        threads = []
        gc.collect()

# execute the remaining threads
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
    optimal_ks[results.get()] += 1
    
most_recurrent_k = np.argmax(optimal_ks) + 1
print(f'The most recurrent optimal number of clusters is {most_recurrent_k}')

# %%
# optimal number of clusters histogram
plt.bar(range(1, len(optimal_ks)+1), optimal_ks, alpha=.9)
plt.axvline(x=most_recurrent_k, color='r', linestyle='dotted', alpha=.6)
plt.xlabel('Optimal number of clusters')
plt.ylabel('Frequency')
plt.title('Frequency of optimal number of clusters')
plt.xticks(range(1, max_k+1), rotation='vertical')
plt.show()

# %%
max_k = 11
optimal_ks = np.zeros(max_k)
runs = 500 # since our implementation does not optimize the initial cluster (it is a simple k-means, not k-means++), we need a bit more time to converge
threads = []
results = queue.Queue()
max_workers = 8
# elbow method
for run in tqdm(range(runs)):
    t = threading.Thread(target=elbow_method_run_gower_optimized, args=(df, range(1, max_k), results, 100, max_workers))
    threads.append(t)
    if len(threads) % max_workers == 0:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            optimal_ks[results.get()] += 1
        threads = []

# execute the remaining threads
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
    optimal_ks[results.get()] += 1
    
most_recurrent_k = np.argmax(optimal_ks) + 1
print(f'The most recurrent optimal number of clusters is {most_recurrent_k}')

# %%
# optimal number of clusters histogram
plt.bar(range(1, len(optimal_ks)+1), optimal_ks, alpha=.9)
plt.axvline(x=most_recurrent_k, color='r', linestyle='dotted', alpha=.6)
plt.xlabel('Optimal number of clusters')
plt.ylabel('Frequency')
plt.title('Frequency of optimal number of clusters')
plt.xticks(range(1, max_k+1), rotation='vertical')
plt.show()

# %% [markdown]
# Turns out that the two methods are not that different, and that could be expected as we proved that they differ only of a rounding operation.<br>
# Best results are attained with **5-6 clusters** 

# %% [markdown]
# ### Investigating on outliers

# %%
labels, centroids, inertia = kmeans_gower_optimized(df, 5, max_iter=300, keep_types=True)

# %%
PCA_tSNE_visualization(df, 5, labels, 'viridis')

# %%
centroids_df = pd.DataFrame(centroids, columns=df.columns).convert_dtypes()
df = df.convert_dtypes()

(df.dtypes == centroids_df.dtypes).all()

# %% [markdown]
# We first compute the proximity matrix (using our custom distance functions) of the datapoints and the centroids. We then extract only the distance of each point with its centroid and proceed the investigation with that distances

# %%
# proximity of datapoints to each centroid
prox_centers = proximity_matrix(data_x=df, data_y=centroids_df, cat_features=bool_cols)

# %%
# we still have to process it as we want the distance of each element to its own cluster center and not to all cluster centers
prox_centers.shape

# %%
km_scores = np.min(prox_centers, axis=1) # we don't need to use labels, as by definition an element's centroid is the closest to it

# %%
plt.hist(km_scores, bins=100)
plt.title('Distance to centroids')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# %%
t2_bound = two_stage_iqr_bound(km_scores)

t2_outliers_idx = np.where(km_scores > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
KM_labels = np.ones(N)
KM_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, KM_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='K-Means')
plot_float_comb_dimensions(df, KM_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])


# %% [markdown]
# ----
# ## <center>Reconstruction Based
# ---- 

# %% [markdown]
# ### <center>PCA

# %%
def pca_reconstruction_error(data, n_components):
    """
    Calculate the reconstruction error between the original data and the reconstructed data.

    Parameters:
    data (pandas DataFrame): The original data.
    reconstructed (pandas DataFrame): The reconstructed data.
    metrics (dict): A dictionary containing the metrics to be used for each type of element in the vectors.

    Returns:
    float: The average reconstruction error.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    reconstructed = pca.inverse_transform(pca.transform(data))
    bool_cols_idx = [df.columns.get_loc(col) for col in bool_cols]
    reconstructed[:, bool_cols_idx] = np.round(reconstructed[:, bool_cols_idx])

    df_reconstructed = pd.DataFrame(reconstructed, columns=df.columns)
    df_reconstructed[bool_cols] = df_reconstructed[bool_cols].astype(bool)

    metrics = {np.bool_: 'hamming', np.float64: 'euclidean', float: 'euclidean'}
    # error = np.array([gower_distance(df.iloc[i], df_reconstructed.iloc[i], metrics) for i in range(df.shape[0])])
    error = proximity_matrix_diag(data_x=df, data_y=df_reconstructed, cat_features=bool_cols)
    return np.sum(error)/reconstructed.shape[0]


# %%
errors = [] 
max_components = len(df.columns)

for n in tqdm(range(1, max_components)):
    error = pca_reconstruction_error(df, n)
    errors.append(error)

plt.plot(range(1, max_components), errors, marker='o', label='Reconstruction error')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction error')
# plt.title('Reconstruction error vs number of components')
plt.xticks(range(1, max_components))
plt.axvline(x=6, color='r', linestyle='dotted', label='Optimal number of components')
plt.legend()
plt.grid()
plt.show()

# %%
pca = PCA(n_components=6)
pca.fit(df)

reconstructed = pca.inverse_transform(pca.transform(df))

bool_cols_idx = [df.columns.get_loc(col) for col in bool_cols]
reconstructed[:, bool_cols_idx] = np.round(reconstructed[:, bool_cols_idx])

df_reconstructed = pd.DataFrame(reconstructed, columns=df.columns)
df_reconstructed[bool_cols] = df_reconstructed[bool_cols].astype(bool)

# reconstruction error with gower distance
metrics = {np.bool_: 'hamming', np.float64: 'euclidean', float: 'euclidean'}
pca_score = np.diag(proximity_matrix(data_x=df, data_y=df_reconstructed, cat_features=bool_cols))

# %%
plt.hist(pca_score, bins=100)
plt.title('Reconstruction error')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.xlim(0, 0.25)
plt.show()

# %%
t2_bound = two_stage_iqr_bound(pca_score)

t2_outliers_idx = np.where(pca_score > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
PCA_labels = np.ones(N)
PCA_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, PCA_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='PCA')
plot_float_comb_dimensions(df, PCA_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])


# %% [markdown]
# ----
# ### <center>Encoder-Decoder

# %%
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(autoencoder, dataloader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        for data in dataloader:
            data = data[0].to(device)
            data = data.float()
            optimizer.zero_grad()
            reconstructed = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

def compute_reconstruction_error(data, reconstructed_data, bool_cols):
    reconstructed_data[bool_cols] = np.round(reconstructed_data[bool_cols]).astype(bool)
    
    data = data.astype(np.float32)
    reconstructed_data = reconstructed_data.astype(np.float32)
    
    error = proximity_matrix_diag(data_x=data, data_y=reconstructed_data, cat_features=bool_cols)
    return error


# %%
bool_data = df[bool_cols].values.astype(float)
float_data = df.drop(columns=bool_cols).values.astype(float)

bool_tensor = torch.tensor(bool_data, dtype=torch.float32)
float_tensor = torch.tensor(float_data, dtype=torch.float32)
tensor_data = torch.cat((bool_tensor, float_tensor), dim=1)

dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = tensor_data.shape[1]
encoding_dim = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
summary(autoencoder, (input_dim,));

# %%
train_autoencoder(autoencoder, dataloader, epochs=100, device=device)

autoencoder.eval()
with torch.no_grad():
    reconstructed_data = autoencoder(tensor_data.to(device)).cpu().numpy()

reconstructed_df = pd.DataFrame(reconstructed_data, columns=df.columns)

ed_score = compute_reconstruction_error(df, reconstructed_df, bool_cols)

# %%
plt.hist(ed_score, bins=100)
plt.title('Reconstruction error - Encoder-Decoder')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# %%
t2_bound = two_stage_iqr_bound(ed_score)

t2_outliers_idx = np.where(ed_score > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
ED_labels = np.ones(N)
ED_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, ED_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='Autoencoder')
plot_float_comb_dimensions(df, ED_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %% [markdown]
# ----
# ## <center>Ensembles

# %% [markdown]
# ### <center> AND

# %%
# find all the points that are outliers in all the methods
methods = [KM_labels, ED_labels, NN_labels]

def common_outliers(methods):
    common_outliers = methods[0]
    for method in methods[1:]:
        common_outliers = np.where(np.logical_and(common_outliers == -1, method == -1), -1, 1)


    return common_outliers

common_outliers_labels = common_outliers(methods)
print(f'Number of common outliers: {np.sum(common_outliers_labels == -1)} ({np.sum(common_outliers_labels == -1)/N*100:.2f}%)')

PCA_tSNE_visualization(df, 2, common_outliers_labels, ['red', 'gray'], legend=anomaly_legend, title_addition='Ensemble: AND')
plot_float_comb_dimensions(df, common_outliers_labels, ['red', 'gray'], legend=anomaly_legend)

# %% [markdown]
# ### <center> OR

# %%
methods = [KM_labels, ED_labels, NN_labels]

def sum_outliers(methods):
    sum_outliers = methods[0]
    for method in methods[1:]:
        sum_outliers = np.where(np.logical_or(sum_outliers == -1, method == -1), -1, 1)

    return sum_outliers

sum_outliers_labels = sum_outliers(methods)
print(f'Number of sum outliers: {np.sum(sum_outliers_labels == -1)} ({np.sum(sum_outliers_labels == -1)/N*100:.2f}%)')

PCA_tSNE_visualization(df, 2, sum_outliers_labels, ['red', 'gray'], legend=anomaly_legend, title_addition='Ensemble: OR')
plot_float_comb_dimensions(df, sum_outliers_labels, ['red', 'gray'], legend=anomaly_legend)

# %% [markdown]
# ### <center>Weighted sum

# %%
methods_scores = [km_scores, ed_score, knn_score]
weights = [.3333, .3333, .3333]

# normalize the scores
methods_scores = [score/np.max(score) for score in methods_scores]

def weighted_sum(scores, weights):
    weighted_sum = np.zeros_like(scores[0])
    for i in range(len(scores)):
        weighted_sum += scores[i]*weights[i]
    return weighted_sum

ws_scores = weighted_sum(methods_scores, weights)

# %%
t2_bound = two_stage_iqr_bound(ws_scores)

t2_outliers_idx = np.where(ws_scores > t2_bound)[0]
print(f'Number of outliers: {len(t2_outliers_idx)}, {len(t2_outliers_idx)/N*100:.2f}%')
WS_labels = np.ones(N)
WS_labels[t2_outliers_idx] = -1
PCA_tSNE_visualization(df, 2, WS_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'], title_addition='Weighted sum')
plot_float_comb_dimensions(df, WS_labels, ['red', 'gray'], legend=['Normal', 'Anomalous'])

# %%
plt.hist(ws_scores, bins=100)
plt.title('Weighted sum scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ----
# # <center>Detections coherence

# %%
del metrics

# %%
import sklearn.metrics as metrics

# %%
y1 = WS_labels
y2 = COF_labels

print(f"Shapes: {-np.sum(y1[np.where(y1==-1)])}, {-np.sum(y2[np.where(y2==-1)])}")
print(f"Homogeneity: {metrics.homogeneity_score(y1, y2):.3f}")
print(f"Completeness: {metrics.completeness_score(y1, y2):.3f}")
print(f"V-measure: {metrics.v_measure_score(y1, y2):.3f}")
R = metrics.adjusted_rand_score(y1, y2)
print(f"Adjusted Rand Index: {R:.3f}")
print("Adjusted Mutual Information:" f" {metrics.adjusted_mutual_info_score(y1, y2):.3f}")


fig20 = plt.figure('Comparison spotted outliers', figsize=(40,2))
mask1 = (y1 == -1)
mask2 = (y2 == -1)
plt.scatter(np.where(mask1)[0], y1[mask1], color='blue', marker="o", label='NN')
plt.scatter(np.where(mask2)[0], y2[mask2], color='red', marker="s", label='LOF')

plt.xlabel('Data points')
plt.ylabel('Predicted label \n (outlier=-1)', fontsize=10)
plt.title('Match on outlier detection between NN and LOF (Rand index = %.3f)' %R)
plt.legend(["NN", "LOF"])
plt.grid()
plt.show()

# %%
methods_labels = [NN_labels, LOF_labels, DBSCAN_labels, COF_labels, KM_labels, ED_labels, PCA_labels, WS_labels]
methods_names = ['NN', 'LOF', 'DBSCAN', 'COF', 'K-Means', 'Autoencoder', 'PCA', 'Weighted sum']

# ARI table
ARI = np.zeros((len(methods_labels), len(methods_labels)))
for i in range(len(methods_labels)):
    for j in range(i, len(methods_labels)):
        ARI[i, j] = metrics.adjusted_rand_score(methods_labels[i], methods_labels[j])
        ARI[j, i] = ARI[i, j]

ARI_df = pd.DataFrame(ARI, columns=methods_names, index=methods_names)
ARI_df


# %%
# V-measure table
V_measure = np.zeros((len(methods_labels), len(methods_labels)))
for i in range(len(methods_labels)):
    for j in range(i, len(methods_labels)):
        V_measure[i, j] = metrics.v_measure_score(methods_labels[i], methods_labels[j])
        V_measure[j, i] = V_measure[i, j]

V_measure_df = pd.DataFrame(V_measure, columns=methods_names, index=methods_names)
V_measure_df

# %% [markdown]
# # Attach anomaly probability column to the original dataset

# %%
df_anom_proba = pd.read_csv('datasets/data.csv', sep=';', index_col='Row')
proba_column = 'Anomaly Probability'
df_anom_proba[proba_column] = ws_scores
df_anom_proba.to_csv('datasets/data_with_anomaly.csv', sep=';', index=True)
df_anom_proba

