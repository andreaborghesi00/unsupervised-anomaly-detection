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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import threading
import queue
import concurrent.futures
import gc

# %%

# Load the data
df = pd.read_csv('datasets/data.csv', sep=';', index_col='Row')
df.head(3)


# %%
def PCA_tSNE_visualization(data2visualize, NCOMP, LABELS, PAL='viridis'):

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
  fig1000.suptitle('Dimensionality reduction of the dataset', fontsize=16)


  # Plot 1: 2D image of the entire dataset
  ax1 = fig1000.add_subplot(121)
  sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], ax=ax1, hue=LABELS, palette=PAL)
  ax1.set_xlabel('Dimension 1', fontsize=10)
  ax1.set_ylabel('Dimension 2', fontsize=10)
  ax1.title.set_text('PCA')
  plt.grid()

  ax2= fig1000.add_subplot(122)
  sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], ax=ax2, hue=LABELS, palette=PAL)
  ax2.set_xlabel('Dimension 1', fontsize=10)
  ax2.set_ylabel('Dimension 2', fontsize=10)
  ax2.title.set_text('tSNE')
  plt.grid()
  plt.show()

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

# %%
# Visualization
# -------------
# Choose your preferred style: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html


plt.style.use('default')

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist as pdist
from scipy.spatial.distance import squareform as sf

distance_metric = 'euclidean'
PM = pdist(df[float_cols], metric=distance_metric)
PM = sf(PM).round(2)
[N,M] = np.shape(df[float_cols])

fig1 = plt.figure(figsize=(30,10))
fig1.suptitle('Visual inspection of float columns', fontsize=20)


# Plot 1: 2D image of the entire dataset
ax1 = fig1.add_subplot(121)
im1 = ax1.imshow(df[float_cols], interpolation='nearest', aspect='auto', cmap='seismic')

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)

ax1.set_xlabel('Attributes', fontsize=16)
ax1.set_xticks(np.arange(0, M, step=1))
ax1.set_ylabel('Observations', fontsize=16)
ax1.set_yticks(np.arange(0, N, step=10))
ax1.title.set_text('Dataset')


# Plot 2: proximity matrix
ax2 = fig1.add_subplot(122)
im2 = ax2.imshow(PM, interpolation='nearest', aspect='auto', cmap='coolwarm')

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)

ax2.set_xlabel('Observations', fontsize=16)
ax2.set_xticks(np.arange(0, N, step=10))
ax2.set_ylabel('Observations', fontsize=16)
ax2.set_yticks(np.arange(0, N, step=10))
ax2.title.set_text('Proximity matrix (%s distance)' % distance_metric)

plt.show()


# %% [markdown]
# # Anomaly detection

# %% [markdown]
# ## Proximity Based

# %% [markdown]
# ### Proximity Matrix for mixed data-types
# We should store these functions in as a library

# %%
def proximity_feat(x, y, metric):
    """
    Calculate the proximity between two input vectors using the specified metric.
    This is intended to compute the proximity between two single features.
    
    Parameters:
    x (array-like): The first input vector.
    y (array-like): The second input vector.
    metric (str): The metric to use. Supported metrics are:
        - 'euclidean': Euclidean distance.
        - 'cosine': Cosine similarity.
        - 'manhattan': Manhattan distance.
        - 'jaccard': Jaccard similarity.
        - 'pearson': Pearson correlation coefficient.
        - 'spearman': Spearman correlation coefficient.
        - 'hamming': Hamming distance.

    Returns:
    float: The proximity between the two input vectors.

    Raises:
    ValueError: If the two input vectors are not of the same type.
    ValueError: If an unknown metric is specified.
    """
    if(type(x) != type(y)):
        raise ValueError(f'The two inputs must be of the same type, got {type(x)} and {type(y)}')

    if metric == 'euclidean':
        return np.linalg.norm(x - y)
    elif metric == 'cosine':
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'jaccard':
        return 1 - np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))
    elif metric == 'pearson':
        return np.corrcoef(x, y)[0, 1]
    elif metric == 'spearman':
        return 1 - 6 * np.sum((x - y) ** 2) / (len(x) * (len(x) ** 2 - 1))
    elif metric == 'hamming':
        return np.sum(x != y) # assuming this is a single value
    else:
        raise ValueError('Unknown metric')


# %%
def gower_distance(x, y, metrics, weights=None):
    """
    Calculates the overall proximity between two vectors, x and y, using a set of metrics and optional weights.

    Parameters:
    x (list): The first input vector.
    y (list): The second input vector.
    metrics (dict): A dictionary containing the metrics to be used for each type of element in the vectors. The keys represent the type of the element and the values represent the metric to be used.
    weights (list, optional): A list of weights for each element in the vectors. If not provided, all elements are assumed to have equal weight.

    Returns:
    float: The overall proximity between the two input vectors.

    Raises:
    ValueError: If the input metrics is not a non-empty dictionary or if the two input vectors have different lengths.
    ValueError: If the weights are negative.
    """
    
    if type(metrics) != dict or len(metrics) == 0:
        raise ValueError("The input metrics must be a non-empty dictionary in the form: {type: metric}") 
    if len(x) != len(y):
        raise ValueError(f'The two input vectors must have the same length found {len(x)} and {len(y)}')
    if weights is None:
        weights = np.ones_like(x)
    elif weights < 0:
        raise ValueError('The weights must be non-negative')
    
    prox = 0
    for xk, yk, wk in zip(x, y, weights):
        curr_metric = metrics[type(xk)]
        prox += wk*proximity_feat(xk, yk, curr_metric)        
    return prox/len(x)


# %%
def proximity_matrix_symmetric(data, metrics, weights=None):
    """
    Calculates the proximity matrix for a given dataset.

    Parameters:
    data (pandas DataFrame): The input dataset.
    metrics (dict): A dictionary containing the metrics to be used for each type of element in the vectors. The keys represent the type of the element and the values represent the metric to be used.
    weights (list, optional): A list of weights for each element in the vectors. If not provided, all elements are assumed to have equal weight.

    Returns:
    prox_matrix (numpy array): The proximity matrix, where each element represents the proximity between two data points.
    """

    prox_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in tqdm(range(data.shape[0]), desc='Computing proximity matrix'):
        for j in range(i, data.shape[0]): # since the matrix is symmetric we start from i, computing the upper triangle
            prox_matrix[i, j] = gower_distance(data.iloc[i], data.iloc[j], metrics, weights)
            prox_matrix[j, i] = prox_matrix[i, j]
    return prox_matrix


# %%
def proximity_matrix_asymmetric(data1, data2, metrics, weights=None):
    """
    Calculates the proximity matrix between two datasets.

    Parameters:
    data1 (pandas DataFrame): The first input dataset.
    data2 (pandas DataFrame): The second input dataset.
    metrics (dict): A dictionary containing the metrics to be used for each type of element in the vectors. The keys represent the type of the element and the values represent the metric to be used.
    weights (list, optional): A list of weights for each element in the vectors. If not provided, all elements are assumed to have equal weight.

    Returns:
    prox_matrix (numpy array): The proximity matrix, where each element represents the proximity between two data points.
    """

    prox_matrix = np.zeros((data1.shape[0], data2.shape[0]))
    for i in tqdm(range(data1.shape[0]), desc='Computing proximity matrix'):
        for j in range(data2.shape[0]):
            prox_matrix[i, j] = gower_distance(data1.iloc[i], data2.iloc[j], metrics, weights)
    return prox_matrix


# %%
def overall_proximity_thread(x, y, metrics, i, j, local_queue, weights=None):
    """
    Calculate the overall proximity between two vectors using different metrics.

    Args:
        x (list): The first input vector.
        y (list): The second input vector.
        metrics (dict): A dictionary of metrics in the form {type: metric}.
        i (int): The index i.
        j (int): The index j.
        local_queue (Queue): A queue to store the result.
        weights (list, optional): The weights for each element in the vectors. Defaults to None.

    Raises:
        ValueError: If the input metrics is not a non-empty dictionary or the two input vectors have different lengths.
        ValueError: If the weights are negative.

    Returns:
        None, the result is stored in the local_queue.
    """
    if type(metrics) != dict or len(metrics) == 0:
        raise ValueError("The input metrics must be a non-empty dictionary in the form: {type: metric}") 
    if len(x) != len(y):
        raise ValueError('The two input vectors must have the same length')
    if weights is None:
        weights = np.ones(len(x))
    elif weights < 0:
        raise ValueError('The weights must be non-negative')
    
    prox = 0
    for xk, yk, wk in zip(x, y, weights):
        curr_metric = metrics[type(xk)]
        prox += wk*proximity_feat(xk, yk, curr_metric)        
    local_queue.put([i,j,prox/len(x)])


# %%
# work in progress, i will try with threadingpool, right now too many threads are created and it crashes
def proximity_matrix_multithreaded(data, metrics, weights=None, n_threads=100):
    """
    Compute the proximity matrix using multiple threads.

    Args:
        data (pandas.DataFrame): The input data.
        metrics (list): List of proximity metrics to be used.
        weights (list, optional): List of weights for each metric. Defaults to None.
        n_threads (int, optional): Number of threads to use. Defaults to 100.

    Returns:
        numpy.ndarray: The proximity matrix.
    """
    threads = []
    q_results = queue.Queue()
    k = 0
    prox_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in tqdm(range(data.shape[0]), desc='Appending threads'):
        for j in range(i, data.shape[0]): # since the matrix is symmetric we start from i, computing the upper triangle
            t = threading.Thread(target=overall_proximity_thread, args=(data.iloc[i], data.iloc[j], i, j, metrics, q_results, weights))
            threads.append(t)

            if len(threads) % n_threads == 0:
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                while not q_results.empty():
                    i, j, prox = q_results.get()
                    prox_matrix[i, j] = prox
                    prox_matrix[j, i] = prox
                threads = []
                gc.collect()
        
    # execute the remaining threads
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not q_results.empty():
        i, j, prox = q_results.get()
        prox_matrix[i, j] = prox
        prox_matrix[j, i] = prox

    return prox_matrix


# %% [markdown]
# Since there is no information available regarding the semantic of the features, all weights are set to one.

# %%
# if proximity_matrix.npy exists, load it
try:
    prox_mat = np.load('datasets/proximity_matrix.npy')
except:
    # it takes 1.5 hours to compute the proximity matrix (i9-9900K)
    prox_mat = proximity_matrix_symmetric(df, {np.bool_: 'hamming', np.float64: 'euclidean'})
    np.save('datasets/proximity_matrix.npy', prox_mat)

# %%
# prox_mat_mt = proximity_matrix_multithreaded(df, {np.bool_: 'hamming', np.float64: 'euclidean'}, 16)

# %%
# symmetry check
print(np.allclose(prox_mat, prox_mat.T))
# print(np.allclose(prox_mat_mt, prox_mat_mt.T))

# %%
# check zero diagonal
print(np.allclose(np.diag(prox_mat), 0))
# print(np.allclose(np.diag(prox_mat_mt), 0))

# %%
# check if the two matrices are similar
# print(np.allclose(prox_mat, prox_mat_mt))

# %% [markdown]
# ----
# ### Distance Based: NN Approach

# %%
from sklearn.neighbors import NearestNeighbors

# %%
# Apply the algorithm
neighborhood_order = 5

# Find neighborhood
neighborhood_set   = NearestNeighbors(n_neighbors=neighborhood_order, algorithm='ball_tree').fit(df[float_cols])
distances, indices = neighborhood_set.kneighbors(df[float_cols])

# compute distances from 5th nearest neighbors and sort them
dk_sorted     = np.sort(distances[:,-1])
dk_sorted_ind = np.argsort(distances[:,-1])


# Identify the outliers as those points with too high distance from their own 5th nearest neighbor
from kneed import KneeLocator
i = np.arange(len(distances))
knee = KneeLocator(i, dk_sorted, S=100, curve='convex', direction='increasing', interp_method='polynomial', online=True)     # see other examples: https://kneed.readthedocs.io/en/stable/parameters.html
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
plt.plot(distances[:,-1])
ax1.set_xlabel('Data points', fontsize=10)
ax1.set_xticks(np.arange(0, N, step=1000))
ax1.set_ylabel('Distances\n(not sorted, %s)' % distance_metric, fontsize=10)
# ax1.title.set_text('Proximity matrix (%s distance)' % distance_metric)
plt.grid()

ax2 = fig3.add_subplot(122)
plt.plot(dk_sorted, 'o-')
ax2.set_xlabel('Data points', fontsize=10)
ax2.set_ylabel('Distances (sorted)', fontsize=10)
plt.axvline(x=knee_x, color='k', linestyle='--')
plt.axhline(y=knee_y, color='k', linestyle='--')
plt.plot((knee_x), (knee_y), 'o', color='r')
plt.grid()

plt.show()

# %%
k = 2 # seems to work best with small k. notice how k=1 is not useful as the queried sample will be itself
knn = NearestNeighbors(n_neighbors=k-1, metric='precomputed') # if we query the same points then the first one will be the point itself and ignored by default, so to get k=5 we need to set k=4
knn.fit(prox_mat);

# %%
dist, idx= knn.kneighbors()

# %%
print(*idx)

# %%
knearest = dist[:,k-2]
sort_idx = np.argsort(knearest)

# %%
# sort idx based on sort_idx
# be careful not to run it more than once, otherwise the idx will be "sorted" again

idx = idx[sort_idx]
dist = dist[sort_idx]
print(dist)

# %%
anomaly_perc = 0.01
n_anomalies = int(anomaly_perc*df.shape[0])
anomalies = idx[df.shape[0]-n_anomalies:, -1]

# %%
anomalies.shape

# %%
anomalies

# %%
# anomalies visualization
labels = np.zeros(df.shape[0])
labels[anomalies] = 1
PCA_tSNE_visualization(df, 2, labels, ['gray', 'red'])


# %% [markdown]
# It seems to work best when k is low, but in any case the results don't seem that great. This might be because we're dealing with a high number of dimensions with respect to the number of samples available.

# %% [markdown]
# ### Density Based

# %% [markdown]
# ## Clustering based

# %% [markdown]
# ### Prototype based clusters: Naive K-Means++
# Since we will be using K-means++, to measure the distance between the datapoints and the centroid, by definition it is used the euclidean distance. Hence each boolean column will be interpreted as a float. Although this is a common practice, such a cast leads to losing information. This is why we may call this first approach Naive, as we allow ourselves to loose such knowledge.
#
# **But how would we approach this problem otherwise?**<br>
# By rededfining the medoid structure, the distance and the update of the medoid.<br>
#
# __Medoid structure__<br>
# Instead of using a cluster defined as a homogeneous array of floats, we will be using an heterogenous one, such that the "representative" for a boolean feature is a boolean, and the "representative" for a float is a float.
#
# __Distance__<br>
# For the distance, we will be using the same approach used for computing the proximity matrix in the previous sections. This type of distance is also known as __Gower distance__ with uniform weights.<br>
#
# __Medoid update__<br>
# After that we can properly assign each datapoint to its closest medoid and compute the next iteration's medoids.
# The classic approach wants to assign to the next iteration centroid the average value of each datapoint of a cluster. 
# In our version we keep computing the average for the float features, and the mode for the booleans.
#

# %% [markdown]
# Now we want to pinpoint a few observations that arise from using the __hamming distance__ for boolean features and the __mode__ while updating the medoid.
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
def kmeans_gower_revisited(data, n_clusters, metrics=None, weights=None, max_iter=300, random_state=None, keep_types=False, result_queue=None):
    if metrics is None:
        metrics = {np.bool_: 'hamming', np.float64: 'euclidean', float: 'euclidean', bool: 'hamming'}

    # Initialize centroids using k-means++ initialization
    template = data.sample(n_clusters, random_state=random_state).values
    centroids = data.sample(n_clusters, random_state=random_state).values.astype(float)
    bool_indices = [data.columns.get_loc(col) for col in bool_cols]
    # float_indices = [data.columns.get_loc(col) for col in float_cols]
    data = data.values.astype(float)

    # for _ in tqdm(range(max_iter), desc=f'K-means Gower for {n_clusters}'):
    for _ in range(max_iter):
        # Compute distances from each data point to centroids
        distances = np.array([np.array([np.linalg.norm(centroid - point) for centroid in centroids]) for point in data])
        labels = np.argmin(distances, axis=1)
        pass
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])
        new_centroids[:, bool_indices] = np.round(new_centroids[:, bool_indices])

        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
        
    inertia = np.sum([np.linalg.norm(data[i] - centroids[labels[i]]) for i in range (data.shape[0])])

    if keep_types:
        template[:,bool_indices] = centroids[:,bool_indices] > .5
        template[:,not bool_indices] = centroids[:,not bool_indices]
        centroids = template

    if result_queue is not None:
        result_queue.put((labels, centroids, inertia))
    return labels, centroids, inertia


# %%
# l, c, i = kmeans_gower(df, 10, max_iter=10)

# %%
lr, cr, ir =kmeans_gower_revisited(df, 10, max_iter=20, keep_types=True)
ir

# %%
# kmeans++ clustering
nk = 4
kmeans = KMeans(n_clusters=nk, init='k-means++', max_iter=1000, ).fit(df)
labels = kmeans.labels_
# kmeans++ visualization
PCA_tSNE_visualization(df, nk, labels, 'viridis')

# %% [markdown]
# #### Elbow method
# To find the optimal number of clusters we run the elbow method

# %%
inertia = []
r = range(2,16)
for k in r:
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(df)
    inertia.append(kmeans.inertia_)

#derivative
first_derivative = np.diff(inertia)
second_derivative = np.diff(first_derivative)

# reasoning for the min of second derivative:
# the first derivative is the slope of the inertia, while the second derivative is the acceleration of the inertia
# the "elbow" represents where the inertia starts to decrease at a slower rate, i.e. where the acceleration is the smallest
optimal_k = np.argmin(second_derivative) + 2 + 2 # +2 because we start from 2 clusters and +2 because each derivative is 1 element shorter than the previous one

plt.plot(r[1:], first_derivative, marker='o', color='r', label='first derivative', linestyle='--', alpha=.6)
plt.plot(r[2:], second_derivative, marker='o', color='g', label='second derivative', linestyle='--', alpha=.6)
plt.plot(r, inertia, marker='o', color='b', label='inertia', alpha=.6)
plt.axvline(x=optimal_k, color='black', linestyle='dotted', label='optimal number of clusters: {}'.format(optimal_k))

# plt.axvline(x=???, color='g', linestyle='dotted', label='optimal number of clusters')
plt.xticks(range(2,16))
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()
plt.show()

# %%
inertia = []
r = range(2,16)
res_queue = queue.Queue()
threads = []

for k in r:
    t = threading.Thread(target=kmeans_gower_revisited, args=(df, k, None, None, 100, None, False, res_queue))
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
optimal_k = np.argmin(second_derivative) + 2 + 2 # +2 because we start from 2 clusters and +2 because each derivative is 1 element shorter than the previous one

plt.plot(r[1:], first_derivative, marker='o', color='r', label='first derivative', linestyle='--', alpha=.6)
plt.plot(r[2:], second_derivative, marker='o', color='g', label='second derivative', linestyle='--', alpha=.6)
plt.plot(r, inertia, marker='o', color='b', label='inertia', alpha=.6)
plt.axvline(x=optimal_k, color='black', linestyle='dotted', label='optimal number of clusters: {}'.format(optimal_k))

# plt.axvline(x=???, color='g', linestyle='dotted', label='optimal number of clusters')
plt.xticks(range(2,16))
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()
plt.show()

# %%
inertia = []
r = range(2,16)
res_queue = queue.Queue()
threads = []

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for k in r:
        executor.submit(kmeans_gower_revisited, df, k, None, None, 100, None, False, res_queue)
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
optimal_k = np.argmin(second_derivative) + 2 + 2 # +2 because we start from 2 clusters and +2 because each derivative is 1 element shorter than the previous one

plt.plot(r[1:], first_derivative, marker='o', color='r', label='first derivative', linestyle='--', alpha=.6)
plt.plot(r[2:], second_derivative, marker='o', color='g', label='second derivative', linestyle='--', alpha=.6)
plt.plot(r, inertia, marker='o', color='b', label='inertia', alpha=.6)
plt.axvline(x=optimal_k, color='black', linestyle='dotted', label='optimal number of clusters: {}'.format(optimal_k))

# plt.axvline(x=???, color='g', linestyle='dotted', label='optimal number of clusters')
plt.xticks(range(2,16))
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.legend()
plt.show()


# %% [markdown]
# If we run the cell above a few times we can clearly notice how the optimal value keeps changing in a range between 5-9. We therefore run it a number of times, say 100, and then plot the frequency of the optimal number of clusters for each run and take the most recurrent.
#
# The number of runs is as high as needed to obtain a consistent result.
# Multithreading is applied to significantly speed up the process.

# %%
# 500 runs with 20 clusters takes around 2 minutes (i9-9900K)
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
    optimal_k = np.argmin(second_derivative) + 2 + 2

    result_queue.put(optimal_k)


# %%
def elbow_method_run_gower_revisited(data, k_range, result_queue, max_iter=100, max_workers=8):
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
        t = threading.Thread(target=kmeans_gower_revisited, args=(data, k, None, None, max_iter, None, False, local_queue))
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
    optimal_k = np.argmin(second_derivative) + 2 + 2

    result_queue.put(optimal_k)


# %%
max_k = 11
optimal_ks = np.zeros(max_k)
runs = 500
threads = []
results = queue.Queue()

# elbow method
for run in tqdm(range(runs)):
    t = threading.Thread(target=elbow_method_run, args=(range(2, max_k), results))
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
runs = 50 # our implementations is much slower than the sklearn one, so we can't afford to run too many times, this should take around 4 hours
threads = []
results = queue.Queue()
max_workers = 8
# elbow method
for run in tqdm(range(runs)):
    t = threading.Thread(target=elbow_method_run_gower_revisited, args=(df, range(2, max_k), results, 100, max_workers))
    threads.append(t)
    if len(threads) % max_workers == 0:
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

# %% [markdown]
# Best results are attained with 5-6 clusters 

# %%
kmeans = KMeans(n_clusters=most_recurrent_k, init='k-means++', n_init=1000).fit(df)
labels = kmeans.labels_

# kmeans++ visualization
PCA_tSNE_visualization(df, most_recurrent_k, labels, 'viridis')

# %%
# cluster centers
centers = kmeans.cluster_centers_
centers_df = pd.DataFrame(centers, columns=df.columns)
float_df = df.copy()
float_df[bool_cols] = float_df[bool_cols].astype(np.float64)

# %%
float_df.dtypes

# %%
# proximity of the cluster centers
metrics = {np.bool_: 'hamming', np.float64: 'euclidean', float: 'euclidean'}
prox_centers = proximity_matrix_asymmetric(float_df, centers_df, metrics)

# %%
prox_centers

# %%
# TODO: define a threshold for the datapoints that are too far from the cluster centers
# TODO: move cells to have the analysis of the naive kmeans and the elbow method together and then the gower distance kmeans and the elbow method together

# %% [markdown]
# ----
# ### Investigation with LOF

# %%
# Apply the algorithm
from sklearn.neighbors import LocalOutlierFactor

lof_model  = LocalOutlierFactor(n_neighbors  = neighborhood_order,
                                algorithm='ball_tree',
                                metric='minkowski', p=2,
                                metric_params = None,
                                contamination = 0.05)
# dir(lof_model)
LOF_labels = lof_model.fit_predict(df[float_cols])     # predict the labels (1 inlier, -1 outlier) of X according to LOF
# dir(lof_model)
LOF_values     = lof_model.negative_outlier_factor_
# print(np.round(LOF_values,2))

# %%
df

# %%
# Verify the outlier detection, count and label the outliers

distances, indices = neighborhood_set.kneighbors(df[float_cols])

fig6 = plt.figure('LOF values', figsize=(8,5))
plt.plot(distances[:,-1], 'k-')
plt.xlabel('Data points', fontsize=10)
plt.xticks(np.arange(0, N, step=10))
plt.ylabel('Distances (not sorted) or LOF values', fontsize=10)
plt.plot(LOF_values, 'ro-')
plt.legend(["Distances (not sorted)", "LOF values"])
plt.grid()
plt.show()


fig7 = plt.figure('Scatterplot with the LOF method', figsize=(10,5))
sns.scatterplot( x = df['Dim_17'], y = df['Dim_18'], hue=LOF_labels, palette=['black','orange'])
# sns.scatterplot( x = tsne_results[:,0], y = tsne_results[:,1], hue=LOF_labels, palette=['black','orange'])
# sns.scatterplot( x = pca_results[:,0], y = pca_results[:,1], hue=LOF_labels, palette=['black','orange'])
sns.set_theme(style='dark')
plt.xlabel('Dimension no.1')
plt.ylabel('Dimension no.2')
plt.legend(['normal','outlier'])
plt.grid()
plt.show()


# Count
count4 = len(LOF_labels[LOF_labels==-1])
print(count4)

# %%
y1 = NN_labels2
y2 = LOF_labels

from sklearn import metrics
print(f"Homogeneity: {metrics.homogeneity_score(y1, y2):.3f}")
print(f"Completeness: {metrics.completeness_score(y1, y2):.3f}")
print(f"V-measure: {metrics.v_measure_score(y1, y2):.3f}")
R = metrics.adjusted_rand_score(y1, y2)
print(f"Adjusted Rand Index: {R:.3f}")
print("Adjusted Mutual Information:" f" {metrics.adjusted_mutual_info_score(y1, y2):.3f}")


# Visually inspect the match between the outliers found by the NN and LOF
fig20 = plt.figure('Comparison spotted outliers', figsize=(18,2))
plt.plot(y1, color='blue', marker="o", label='NN')
plt.plot(y2, color='red', marker="x", label='LOF')
plt.xlabel('Data points')
plt.ylabel('Predicted label \n (outlier=-1, normal=1)', fontsize=10)
plt.title('Match on outlier detection between NN and LOF (Rand index = %.2f)' %R)
plt.legend(["NN", "LOF"])
plt.grid()
plt.show()
