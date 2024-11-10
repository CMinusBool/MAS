import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv

def remove_symetric_edges(matrix):
    number_of_symetric_edges = 0
    number_of_edges = 0
    for i in range(0, len(matrix)):
        for neighbour in matrix[i]:
            number_of_edges += 1
            if i in matrix[neighbour]:
                matrix[neighbour].remove(i)
                number_of_symetric_edges += 1
    return number_of_symetric_edges, number_of_edges, matrix

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
number_of_symetric_edges, number_of_edges,K_neighbours = remove_symetric_edges(K_neighbours)

print(f"K_neighbours edge count: {number_of_edges}")
print(f"Number of symetric edges: {number_of_symetric_edges}")

e_neighbours = e_radius(kernel_matrix, 0.9)

number_of_symetric_edges, number_of_edges, e_neighbours = remove_symetric_edges(e_neighbours)

print(f"e_neighbours edge count: {number_of_edges}")
print(f"Number of symetric edges: {number_of_symetric_edges}")

#TODO save similarity of k neighbours to csv
# add header
#csv in format- Source;target;similarity

with open('K_neighbours.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    for i in range(0, len(K_neighbours)):
        for neighbour in K_neighbours[i]:
            writer.writerow([i, neighbour, kernel_matrix[i][neighbour]])

with open('e_neighbours.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    for i in range(0, len(e_neighbours)):
        for neighbour in e_neighbours[i]:
            writer.writerow([i, neighbour, kernel_matrix[i][neighbour]])
