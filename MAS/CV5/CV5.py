import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv


def gaussian_kernel_matrix(X, sigma):
    distances = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)
    kernel_matrix = np.exp(-distances / (2 * sigma ** 2))

    return kernel_matrix

def KNN_neighbours(matrix, k):
    neighbours = []
    for row in matrix:
        row_neighbours = []
        for i in range(0, k):
            max_index = np.argmax(row)
            row_neighbours.append(max_index)
            row[max_index] = 0
        neighbours.append(row_neighbours)
    return neighbours

def e_radius(matrix, e):
    neighbours = []
    for row in matrix:
        row_neighbours = []
        for i in range(0, len(row)):
            if row[i] > e:
                row_neighbours.append(i)
        neighbours.append(row_neighbours)
    return neighbours

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Iris.csv')
df = pd.read_csv(csv_path, sep=';', decimal=',')

# encode target column
df['Species'] = df['Species'].astype('category')
df['Species'] = df['Species'].cat.codes

X = df.drop('Species', axis=1).values
y = df['Species'].values

# calculate gausian kernel
kernel_matrix = gaussian_kernel_matrix(X, 1)

# change every 1 in kernel matrix to 0, beacause we don't want to have (a,a) vertices
kernel_matrix = np.where(kernel_matrix == 1, 0, kernel_matrix)

# KNN neighbours
K_neighbours = KNN_neighbours(kernel_matrix, 3)

# remove symetric edges
# todo fix removing symetric edges
for i in range(0, len(K_neighbours)):
    for neighbour in K_neighbours[i]:
        if i not in K_neighbours[neighbour]:
            K_neighbours[i].remove(neighbour)


e_neighbours = e_radius(kernel_matrix, 0.9)



print("e_neighbours total count:")
count = 0
for row in e_neighbours:
    count += len(row)

print(count)


#csv in format- Source;target;similarity