import torch
from sklearn.cluster import KMeans
import numpy as np
import time
import random


X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans2 = KMeans(n_clusters=2, random_state=0).fit_transform(X)

km = kmeans.transform(X)
print(kmeans.labels_)

kmeans.predict([[0, 0], [4, 4]])

print(kmeans.cluster_centers_)


def get_dist_graph(all_points, num_anchor=300):
    """
    get the cluster center as anchor by K-means++
    and calculate distance graph (n data points vs m anchors),
    :param all_points: n data points
    :param num_anchor:  m anchors, default = 300
    :return: distance graph n X m
    """
    # kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit_transform(all_points)
    # print ('dist graph done!')
    # return np.asarray(kmeans)
    ## smaple
    sample_rate = 3000
    num_data = np.size(all_points,0)
    ind = random.sample(range(num_data),sample_rate)
    sample_points = all_points[ind,:]
    kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit(sample_points)
    km = kmeans.transform(all_points)
    print ('dist graph done!')
    return np.asarray(km)

dist_graph = np.random.rand(25000, 4096)
start = time.time()
a = get_dist_graph(dist_graph)
end = time.time()
print end-start