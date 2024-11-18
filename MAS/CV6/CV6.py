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
    matrix_copy = matrix.copy()
    for row in matrix_copy:
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

def generate_graph(nodes = 550, probability_for_edge = 0.001):
    edges = []
    for node in range(nodes):
        for edge in range(node, nodes):
            if edge != node:
                if np.random.uniform() < probability_for_edge:
                    edges.append((node, edge))

    return edges

number_of_nodes = 550
graph1 = generate_graph(nodes = number_of_nodes ,probability_for_edge=0.004)

with open('graph2Nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Id"])
    for i in range(0, number_of_nodes):
        writer.writerow([i])

with open('graph2Edges.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Source", "Target"])
    for i in range(0, len(graph1)):
        writer.writerow(graph1[i])



