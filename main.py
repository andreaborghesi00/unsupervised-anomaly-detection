# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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


# %% [markdown]
# # Anomaly detection

# %% [markdown]
# ## Proximity Based

# %% [markdown]
# ### Proximity Matrix for mixed data-types

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
        raise ValueError('The two inputs must be of the same type')

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
def overall_proximity(x, y, metrics, weights=None):
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
        raise ValueError('The two input vectors must have the same length')
    if weights is None:
        weights = np.ones(len(x))
    elif weights < 0:
        raise ValueError('The weights must be non-negative')
    
    prox = 0
    for xk, yk, wk in zip(x, y, weights):
        curr_metric = metrics[type(xk)]
        prox += wk*proximity_feat(xk, yk, curr_metric)        
    return prox/len(x)


# %%
def proximity_matrix(data, metrics, weights=None):
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
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]): # since the matrix is symmetric we start from i, computing the upper triangle
            prox_matrix[i, j] = overall_proximity(data.iloc[i], data.iloc[j], metrics, weights)
            prox_matrix[j, i] = prox_matrix[i, j]
    return prox_matrix


# %%
# if proximity_matrix.npy exists, load it
try:
    prox_mat = np.load('proximity_matrix.npy')
except:
    prox_mat = proximity_matrix(df, {np.bool_: 'hamming', np.float64: 'euclidean'})
    np.save('proximity_matrix.npy', prox_mat)

# %% [markdown]
# ### Distance Based: NN Approach

# %%
from sklearn.neighbors import NearestNeighbors as knn

# %%
