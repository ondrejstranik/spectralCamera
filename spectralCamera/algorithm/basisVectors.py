# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:07:14 2023

@author: ungersebastian
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
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

    v1 = 2
    v2 = 1.4
    ang = 30*np.pi/180

    rotmat = np.array([
        [np.cos(ang), np.sin(ang)],
        [-np.sin(ang), np.cos(ang)]
        ])

    true = np.array([
        np.dot(rotmat, np.array([0,v2])),
        np.dot(rotmat, np.array([0,-v2])),
        np.dot(rotmat, np.array([v1,0])),
        np.dot(rotmat, np.array([-v1,0]))
        
        ])

    scale = 3
    n = 100

    com = np.reshape(np.array([[np.dot(rotmat, np.array([(i1+(np.random.rand(1)-0.5)/scale)*v1, (i2+(np.random.rand(1)-0.5)/scale)*v2])) for i2 in range(n)] for i1 in range(n)]),(n*n,2))

    plt.figure()
    plt.scatter(*com.T)


    v1 = lattice_basis_vectors(com)

    plt.figure()
    plt.scatter(*lattice_basis_vectors(com, 10).T, c = 'green')
    plt.scatter(*lattice_basis_vectors(com, 1000).T, c = 'red')
    plt.scatter(*true.T, c = 'blue')
    np.median(v1[0])

