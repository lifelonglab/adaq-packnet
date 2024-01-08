from sklearn import cluster
import scipy.cluster.vq as vq
from kmeans_scaler_hist import kmeans_scaler_hist
import numpy as np
from numpy import linalg as LA


def vq_and_back(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    X = X.cpu().numpy()
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0
    
    print("Cluster Values = {}".format(values))
    out = np.take(values, labels)
    out.shape = filt.shape
    return out, values, labels


def vq_and_back_fast(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    sz = X.shape
    if False:#sz[0] > 1000000:
        idx = np.random.choice(sz[0],100000)
        x_short = X[idx,:]
    else:
        x_short = X
    k_means.fit(x_short)
    values = k_means.cluster_centers_#.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0

    # create an array from labels and values
    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X, values)
    print("shape labels")
    print(labels)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vq_and_back_fastest(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    sz = X.shape
    print(sz)
    idx = np.random.choice(sz[0],100000)
    x_short = X[idx,:]
    values, edges = kmeans_scaler_hist(x_short, clusters_used)
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0

    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X.flatten(), values)
    print("shape labels")
    print(labels)
    ids, counts = np.unique(labels, return_counts=True)
    print("Counts")
    print(counts)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vquant(in_tensor, n_clusters=16, sparsity_threshold=0, fast=False):
    in_np = in_tensor
    np.random.seed(0)
    shape = in_np.shape
    out_combined = np.zeros(in_np.shape)
    print('shape' + str(in_np.shape))
    filt = in_np
    if fast == True:
        out = vq_and_back_fastest(filt, n_clusters, sparsity_threshold=sparsity_threshold)
    else:
        out, values, labels = vq_and_back(filt, n_clusters, sparsity_threshold=sparsity_threshold)

    out_combined = out
    out_tensor = out_combined

    return out_tensor, values, np.reshape(labels, out_tensor.shape)