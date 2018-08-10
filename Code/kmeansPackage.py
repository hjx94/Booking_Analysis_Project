# =============================================================
# CS5010 Final Project
# Team Avocado
# Jiangxue Han(jh6rg), Sijia Tang(st4je), Rakesh Ravi K U(rk9cx)
# Sakshi Jawarani(sj8em), Elena Gillis(emg3sc)
# =============================================================

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow(dataframe):
    headers = list(dataframe)
    x = []
    for header in headers:
        x.append(dataframe[header])
    x = np.array(x)
    x = x.transpose()
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

    
    for n_cluster in range(2, 11):
        kmeans = KMeans(n_clusters=n_cluster).fit(x)
        label = kmeans.labels_
        sil_coeff = silhouette_score(x, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
        
def kmeans_clustering(dataframe, k, dimension, variable1, variable2, variable3=None):
    headers = list(dataframe)
    x = []
    for header in headers:
        x.append(dataframe[header])
    x = np.array(x)
    x = x.transpose()
    kmeans = KMeans(n_clusters=3, max_iter=1000).fit(x)
    df_label = pd.DataFrame(data=kmeans.labels_)
    df_withLabel = pd.concat([dataframe,df_label],axis=1)
    print (dataframe.columns)
    colNames = list(dataframe.columns)
    print (colNames)
    colNames.append('Label')
    length = len(colNames) - 1
    df_withLabel.columns = colNames
    label = df_withLabel.columns
    centers = kmeans.cluster_centers_

    if dimension == 2:
        data = df_withLabel.values[:,0:length]
        category = df_withLabel.values[:,length]
        n = df_withLabel.shape[0]
        colors=['orange', 'blue', 'green']
        for i in range(n):
            plt.scatter(data[i, variable1], data[i,variable2], s=7, color = colors[int(category[i])])
        plt.scatter(centers[:,variable1], centers[:,variable2], marker='*', c=colors, s=150)
        plt.xlabel(label[variable1])
        plt.ylabel(label[variable2])

        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        print (dict(zip(unique, counts)))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.array(df_withLabel[label[variable1]])
        y = np.array(df_withLabel[label[variable2]])
        z = np.array(df_withLabel[label[variable3]])
        
        ax.scatter(x,y,z, marker="s", c=df_withLabel["Label"], s=5, cmap="summer")
        
        ax.set_xlabel(label[variable1])
        ax.set_ylabel(label[variable2])
        ax.set_zlabel(label[variable3])
        
        plt.show()
    return centers
