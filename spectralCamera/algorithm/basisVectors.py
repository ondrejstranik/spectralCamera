# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:07:14 2023

@author: ungersebastian
"""
#%%
import numpy as np
from sklearn.cluster import KMeans
import scipy.spatial.distance as distance

def lattice_basis_vectors(com, n_use = 10, remove_outliers = 1, threshold = 0.4):
    
    center = [np.mean(com[:,0]), np.mean(com[:,1])]
    
    dist = np.linalg.norm(com - center, axis = 1)
    min_vec = com[np.argsort(dist)[0:n_use]]
    min_vec = min_vec[0:n_use] - min_vec[0]

    cdist = distance.cdist(min_vec, min_vec)
    
    basvec = np.array([min_vec[np.argsort(d)[1:5]]-v for d, v in zip(cdist, min_vec)]).reshape(-1,2)
    kmeans = KMeans(n_clusters=4).fit(basvec)
    
    if remove_outliers == False:
        return kmeans.cluster_centers_
    else:
        
        labels = kmeans.labels_
        basgroups = [basvec[labels == i1] for i1 in range(4)]
        
        median = [np.median(v, axis = 0) for v in basgroups]
        dist = [np.linalg.norm((b - m)/np.linalg.norm(m), axis = 1) for b, m in zip(basgroups, median)]
        basgroups = [b[d<threshold] for b, d in zip(basgroups, dist)]
        
        center = [np.mean(b, axis = 0) for b in basgroups]
    
        return np.array(center)


if __name__ == 'main':
    pass