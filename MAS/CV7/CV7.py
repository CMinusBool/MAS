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
def generate_graph(nodes = 550, probability_for_edge = 0.001):
    edges = []
    for node in range(nodes):
        for edge in range(node, nodes):
            if edge != node:
                if np.random.uniform() < probability_for_edge:
                    edges.append((node, edge))

    return edges

def ba_model(start_nodes = 3, edges_for_start_nodes = 1,end_nodes = 10, edges_for_new = 2):
    #generating starting graph that is connected
    edges = [(0,1)]
    connected_nodes = [0,1]
    for i in range(2, start_nodes):
        for _ in range(edges_for_start_nodes):#TODO if there are not enough nodes yet
            connect_to = np.random.randint(0,i)
            edges.append((connect_to, i))
            connected_nodes.extend((connect_to, i))

    #coment
    for i in range(start_nodes, end_nodes):
        for _ in range(edges_for_new):
            random_iter = np.random.randint(0, len(connected_nodes))
            #they are not reflexive
            if i != connected_nodes[random_iter]:
                #TODO check if they are not alredy in edges

    return edges


graph = ba_model(start_nodes=10)
print(graph)

#for testing networkx barabasi...