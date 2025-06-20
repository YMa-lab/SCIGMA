import os
import ot
import pickle
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import anndata
import sklearn
import scipy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri


'''
Utility Functions

Some utility functions have been modified from https://github.com/JinmiaoChenLab/SpatialGlue/blob/main/SpatialGlue/utils.py (2024)
'''
def seed(seed=1000):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def lsi(
        adata: anndata.AnnData, n_components=20,
        use_highly_variable=True, **kwargs
       ):
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable:
        sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000)
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:,1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf    
          

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=1000):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def embedding_separation_scores(adata, emb_key='emb_combined', label_key='SCIGMA'):
    """
    Separation scores - mean distances between centers of each cluster
    
    Params:
    adata - AnnData object with a .obsm field containing the embedding, a .obs field containing the cluster labels
    emb_key - key for .obsm field containing the embedding
    label_key - key of .obs field containing cluster labels
    
    Return:
    n_cluster x n_cluster DataFrame S, where S_ij is the separation score for cluster i and j
    """
    n_clusters = len(adata.obs[label_key].unique())
    cluster_ids = list(adata.obs[label_key].unique())
    centroids = {}
    
    # define centroids for each cluster
    for i in range(n_clusters):
        cluster = cluster_ids[i]
        embeddings = adata[adata.obs[label_key]==cluster].obsm[emb_key]
        centroids[cluster] = np.sum(embeddings, axis=0) / embeddings.shape[0]
    
    S = np.empty(shape=(n_clusters,n_clusters))
    # calculate separation scores
    for i in range(n_clusters):
        cluster_i = cluster_ids[i]
        for j in range(i, n_clusters):
            cluster_j = cluster_ids[j]
            sep_score = np.linalg.norm(centroids[cluster_i] - centroids[cluster_j])
            S[i][j] = sep_score
            S[j][i] = sep_score
            
    res = pd.DataFrame(S, index=cluster_ids, columns=cluster_ids)
    return res


def separation_scores(adata, coord_key='spatial', label_key='SCIGMA'):
    """
    Separation scores - mean distances between centers of each cluster
    
    Params:
    adata - AnnData object with a .obsm field containing the embedding, a .obs field containing the cluster labels
    coord_key - key for .obsm field containing the embedding
    label_key - key of .obs field containing cluster labels
    
    Return:
    n_cluster x n_cluster DataFrame S, where S_ij is the separation score for cluster i and j
    """
    n_clusters = len(adata.obs[label_key].unique())
    cluster_ids = list(adata.obs[label_key].unique())
    centroids = {}
    
    # define centroids for each cluster
    for i in range(n_clusters):
        cluster = cluster_ids[i]
        coordinates = adata[adata.obs[label_key]==cluster].obsm[coord_key]
        centroids[cluster] = np.sum(coordinates, axis=0) / coordinates.shape[0]
    
    S = np.empty(shape=(n_clusters,n_clusters))
    # calculate separation scores
    for i in range(n_clusters):
        cluster_i = cluster_ids[i]
        for j in range(i, n_clusters):
            cluster_j = cluster_ids[j]
            sep_score = np.linalg.norm(centroids[cluster_i] - centroids[cluster_j])
            S[i][j] = sep_score
            S[j][i] = sep_score
            
    res = pd.DataFrame(S, index=cluster_ids, columns=cluster_ids)
    return res


def compactness_scores(adata, emb_key='emb_combined', label_key='SCIGMA'):
    """
    Compactness scores - mean distances of each cell to its cluster's center
    
    Params:
    adata - AnnData object with a .obsm field containing the embedding, a .obs field containing the cluster labels
    emb_key - key for .obsm field containing the embedding
    label_key - key of .obs field containing cluster labels
    
    Return:
    n_cluster DataFrame S, where S_i is the compactness score for cluster i
    """
    n_clusters = len(adata.obs[label_key].unique())
    cluster_ids = list(adata.obs[label_key].unique())
    centroids = {}
    
    # define centroids for each cluster
    for i in range(n_clusters):
        cluster = cluster_ids[i]
        embeddings = adata[adata.obs[label_key]==cluster].obsm[emb_key]
        centroids[cluster] = np.sum(embeddings, axis=0) / embeddings.shape[0]
    
    S = np.empty(shape=(n_clusters,))
    # calculate compactness scores
    for i in range(n_clusters):
        cluster_i = cluster_ids[i]
        embeddings = adata[adata.obs[label_key]==cluster].obsm[emb_key]
        compactness_score = np.linalg.norm(np.mean(embeddings - centroids[cluster_i], axis=0))
        S[i] = compactness_score
    
    res = pd.DataFrame(np.array(S), index=cluster_ids)
    return res


def boundary_scores(adata, spatial_key='spatial', label_key='SCIGMA', neighbors=12):
    """
    Boundary scores - absolute difference in distances of each cell to its cluster's center
    
    Params:
    adata - AnnData object with a .obsm field containing the embedding, a .obs field containing the cluster labels
    spatial_key - key for .obsm field containing the spatial info
    label_key - key of .obs field containing cluster labels
    
    Return:
    array of fractions
    """
    from scipy.spatial import KDTree
    
    # Format data
    points = np.array(adata.obsm[spatial_key])
    labels = np.array(adata.obs[label_key])

    # Build KDTree
    tree = KDTree(points)

    # Initialize the fractions list
    fractions = []

    # Query the KDTree for the 6 nearest neighbors of each point
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=neighbors+1)  # k=7 includes the point itself
        indices = indices[1:]  # Exclude the point itself

        # Count neighbors with the same label
        same_label_count = np.sum(labels[indices] == labels[i])

        # Calculate the fraction
        fraction = same_label_count / neighbors
        fractions.append(fraction)

    return np.array(fractions)
        

        
def clustering(adata, n_clusters=7, key='emb', add_key='SCIGMA', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=True, n_components=20, weight=None):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       pca = PCA(n_components=n_components, random_state=42) 
       adata.obsm[key + '_pca'] = pca.fit_transform(adata.obsm[key].copy())
       if weight:
           adata.obsm[key + '_pca'] = adata.obsm[key + '_pca'] * weight
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
        
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res  
