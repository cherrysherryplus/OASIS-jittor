import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from mmd_numpy_sklearn import *


def cluster(affinity_option, data, labels_path):
    if affinity_option == 'euclidean':
        linkage = 'ward'
        n_clusters = 10
    elif affinity_option == 'precomputed':
        assert len(data.shape)==2 and data.shape[0] == data.shape[1], "precomputed requires data to be a distances matrix between every two items"
        linkage = 'average'
        n_clusters = 8
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity_option, linkage=linkage)
    labels_ = cluster.fit_predict(data)
    np.save(labels_path, labels_)
    print("cluster done")    


if __name__ == "__main__":
    affinity_option = 'precomputed'
    distances_matrix = np.load("mmd.npy")
    cluster(affinity_option="precomputed", data=distances_matrix, labels_path="labels_mmd.npy")
    
    