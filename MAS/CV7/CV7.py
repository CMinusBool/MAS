import os
import time

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
    if start_nodes < edges_for_new:
        start_nodes = edges_for_new #minimum number of start nodes = edges_for_new
    if start_nodes < 2:
        start_nodes = 2 #minimum number of start nodes is 2
    #generating starting graph that is connected
    edges = [(0,1)]
    connected_nodes = [0,1]#each node is in here 1 time for each edge it is part of
    for i in range(2, start_nodes):
        for _ in range(edges_for_start_nodes):
            connect_to = np.random.randint(0,i)
            edges.append((i, connect_to))
            connected_nodes.extend((i, connect_to))

    #In each step add node and connect it to 2 other nodes (higher degree of node = higher connection prob.)
    for i in range(start_nodes, end_nodes):
        connections_made = 0
        while connections_made < edges_for_new:
            potential_edge = (i ,connected_nodes[np.random.randint(0, len(connected_nodes))])

            if i != potential_edge[1]:#not identity edge
                if potential_edge in edges:#no duplicity in edges
                    continue
                else:
                    edges.append(potential_edge)
                    connected_nodes.extend(potential_edge)
                    connections_made += 1

    return edges

start_time = time.time()
number_of_nodes = 550
number_of_edges = 2
graph1 = ba_model(end_nodes = 550, edges_for_new=number_of_edges)
#print(graph1)
print(f"Time to generate graph: {time.time() - start_time}")

with open(f'Node_list_m{number_of_edges}_Nodes_{number_of_nodes}.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Id"])
    for i in range(0, number_of_nodes):
        writer.writerow([i])

with open(f'Edge_list_m{number_of_edges}_Nodes_{number_of_nodes}.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Source", "Target"])
    for i in range(0, len(graph1)):
        writer.writerow(graph1[i])


start_time = time.time()
number_of_nodes = 550
number_of_edges = 3
graph2 = ba_model(end_nodes = 550, edges_for_new=number_of_edges)
#print(graph2)
print(f"Time to generate graph: {time.time() - start_time}")

with open(f'Node_list_m{number_of_edges}_Nodes_{number_of_nodes}.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Id"])
    for i in range(0, number_of_nodes):
        writer.writerow([i])

with open(f'Edge_list_m{number_of_edges}_Nodes_{number_of_nodes}.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Source", "Target"])
    for i in range(0, len(graph2)):
        writer.writerow(graph2[i])

#for testing networkx barabasi...