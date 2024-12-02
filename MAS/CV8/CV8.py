import os
import time

import matplotlib.pyplot as plt
import networkx
import networkx as nx
import numpy as np
import csv
import littleballoffur as lbf
from collections import Counter


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
nodes = 5000
m = 2
smpl_percentage = 0.15
original = networkx.barabasi_albert_graph(nodes, 2)

# random node sampling
sampler = lbf.RandomNodeSampler(number_of_nodes=int(nodes*smpl_percentage))
random_node_sampling = sampler.sample(original)

# random edge sampling
sampler = lbf.RandomEdgeSampler(number_of_edges=int(nodes*smpl_percentage*1.2)//2)
random_edge_sampling = sampler.sample(original)

# snowball sampling
sampler = lbf.SnowBallSampler(number_of_nodes=int(nodes*smpl_percentage))
snowball_sampling = sampler.sample(original)

# degree sequences
# random node sampling
degree_sequences_rns = sorted((d for n, d in random_node_sampling.degree()), reverse=True)
degree_sequences_counts_rns = Counter(degree_sequences_rns)
degree_array_rns = []
degree_count_array_rns = []

for _ in degree_sequences_counts_rns.items():
    degree_array_rns.append(_[0])
    degree_count_array_rns.append(_[1])

# random edge sampling
degree_sequences_res = sorted((d for n, d in random_edge_sampling.degree()), reverse=True)
degree_sequences_counts_res = Counter(degree_sequences_res)
degree_array_res = []
degree_count_array_res = []

for _ in degree_sequences_counts_res.items():
    degree_array_res.append(_[0])
    degree_count_array_res.append(_[1])

# snowball sampling
degree_sequences_sns = sorted((d for n, d in snowball_sampling.degree()), reverse=True)
degree_sequences_counts_sns = Counter(degree_sequences_sns)
degree_array_sns = []
degree_count_array_sns = []

for _ in degree_sequences_counts_sns.items():
    degree_array_sns.append(_[0])
    degree_count_array_sns.append(_[1])

# original graph
degree_sequences = sorted((d for n, d in original.degree()), reverse=True)
degree_sequences_counts = Counter(degree_sequences)
degree_array = []
degree_count_array = []

for _ in degree_sequences_counts.items():
    degree_array.append(_[0])
    degree_count_array.append(_[1])

# each sample statistics
print(f"Random Node Sampling: {len(random_node_sampling.nodes())} nodes, {len(random_node_sampling.edges())} edges")
print(f"Random Edge Sampling: {len(random_edge_sampling.nodes())} nodes, {len(random_edge_sampling.edges())} edges")
print(f"Snowball Sampling: {len(snowball_sampling.nodes())} nodes, {len(snowball_sampling.edges())} edges")
print(f"Original Graph: {len(original.nodes())} nodes, {len(original.edges())} edges")

# plotting the degree distributions
plt.plot(degree_array_rns, degree_count_array_rns, 'ro-')
plt.plot(degree_array_res, degree_count_array_res, 'bo-')
plt.plot(degree_array_sns, degree_count_array_sns, 'go-')
plt.plot(degree_array, degree_count_array, 'yo-')

plt.legend(['Random Node Sampling', 'Random Edge Sampling', 'Snowball Sampling', 'Original Graph'])


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution with Logarithmic Y-axis')

plt.show()

# comulative degree distribution
cumulative_degree_array_rns = []
cumulative_degree_count_array_rns = []
cumulative_degree_array_res = []
cumulative_degree_count_array_res = []
cumulative_degree_array_sns = []
cumulative_degree_count_array_sns = []
cumulative_degree_array = []
cumulative_degree_count_array = []

for i in range(len(degree_array_rns)):
    cumulative_degree_array_rns.append(degree_array_rns[i])
    cumulative_degree_count_array_rns.append(sum(degree_count_array_rns[i:]))
for i in range(len(degree_array_res)):
    cumulative_degree_array_res.append(degree_array_res[i])
    cumulative_degree_count_array_res.append(sum(degree_count_array_res[i:]))
for i in range(len(degree_array_sns)):
    cumulative_degree_array_sns.append(degree_array_sns[i])
    cumulative_degree_count_array_sns.append(sum(degree_count_array_sns[i:]))
for i in range(len(degree_array)):
    cumulative_degree_array.append(degree_array[i])
    cumulative_degree_count_array.append(sum(degree_count_array[i:]))

# plot for cumulative degree distributions
plt.figure()

plt.plot(cumulative_degree_array_rns, cumulative_degree_count_array_rns, 'ro-')
plt.plot(cumulative_degree_array_res, cumulative_degree_count_array_res, 'bo-')
plt.plot(cumulative_degree_array_sns, cumulative_degree_count_array_sns, 'go-')
plt.plot(cumulative_degree_array, cumulative_degree_count_array, 'yo-')

plt.legend(['Random Node Sampling', 'Random Edge Sampling', 'Snowball Sampling', 'Original Graph'])

plt.xlabel('Degree')
plt.ylabel('Cumulative Frequency')
plt.title('Cumulative Degree Distribution with Logarithmic Y-axis')

plt.show()

#TODO normalize comulative distributions
#TODO KS test

print(f"Time to finish: {time.time() - start_time}")

