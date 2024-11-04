import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv


def calculate_degrees(matrix):
    degrees = []
    for row in matrix:
        degrees.append(np.sum(row))
    degrees.sort()
    return degrees


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'KarateClub.csv')
df = pd.read_csv(csv_path, header=None, sep=';')

# using networkx populate graph
graph_x = nx.Graph()
for _, row in df.iterrows():
    graph_x.add_edge(row[0], row[1])

# calculate clustering coeficients for each node
coefficients = nx.clustering(graph_x)
sum_coefficients = 0
#print("Coefficients of nodes:")
for i in range(1, 35):
    #print(f"{i}, {coefficients[i]}")
    sum_coefficients += coefficients[i]

# average coefficient
average_coef = sum_coefficients / 34
#print(f"Average coeficient: {average_coef}")

degrees = nx.degree(graph_x)

count_coefficients = {}
degrees_counts = {}

for key, value in degrees:
    if value in count_coefficients:
        count_coefficients[value] += coefficients[key]
        degrees_counts[value] += 1
    else:
        count_coefficients[value] = coefficients[key]
        degrees_counts[value] = 1

average_coef_by_degree = []
for key, value in count_coefficients.items():
    average_coef_by_degree.append([key, value / degrees_counts[key]])

# calculate closenes centrality
closeness = nx.closeness_centrality(graph_x)

######################
# CV4 add communities
######################

# louvain_communities
communities = nx.community.louvain_communities(graph_x)
#each community is in different color
pos = nx.spring_layout(graph_x)
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
louvian_community = {}
for i, community in enumerate(communities):
    nx.draw_networkx_nodes(graph_x, pos, nodelist=community, node_color=colors[i])
    nx.draw_networkx_edges(graph_x, pos, edgelist=graph_x.edges(community), edge_color=colors[i])
    for node in community:
        louvian_community[node] = i
plt.title("Louvain communities")
plt.show()

louvain_modularity = nx.community.quality.modularity(graph_x, communities)
print(f"Louvain modularity: {louvain_modularity}")

# label propagation
communities = nx.community.label_propagation_communities(graph_x)
label_propagation_community = {}
for i, community in enumerate(communities):
    nx.draw_networkx_nodes(graph_x, pos, nodelist=community, node_color=colors[i])
    nx.draw_networkx_edges(graph_x, pos, edgelist=graph_x.edges(community), edge_color=colors[i])
    for node in community:
        label_propagation_community[node] = i

plt.title("Label propagation communities")
plt.show()

label_propagation_modularity = nx.community.quality.modularity(graph_x, communities)
print(f"Label propagation modularity: {label_propagation_modularity}")

# kernighan_lin_bisection
communities = nx.community.kernighan_lin_bisection(graph_x)
communities1 = nx.community.kernighan_lin_bisection(graph_x.subgraph(communities[0]))
communities2 = nx.community.kernighan_lin_bisection(graph_x.subgraph(communities[1]))
communities = [communities1[0], communities1[1], communities2[0], communities2[1]]

kernighan_lin_bisection_community = {}
for i, community in enumerate(communities):
    nx.draw_networkx_nodes(graph_x, pos, nodelist=community, node_color=colors[i])
    nx.draw_networkx_edges(graph_x, pos, edgelist=graph_x.edges(community), edge_color=colors[i])
    for node in community:
        kernighan_lin_bisection_community[node] = i

plt.title("Kernighan Lin Bisection communities")
plt.show()

kernighan_lin_bisection_modularity = nx.community.quality.modularity(graph_x, communities)
print(f"Kernighan Lin Bisection modularity: {kernighan_lin_bisection_modularity}")

# girvan_newman/edge_betweenness_partition
communities = nx.community.edge_betweenness_partition(graph_x, number_of_sets=4)

girvan_newman_community = {}

for i, community in enumerate(communities):
    nx.draw_networkx_nodes(graph_x, pos, nodelist=community, node_color=colors[i])
    nx.draw_networkx_edges(graph_x, pos, edgelist=graph_x.edges(community), edge_color=colors[i])
    for node in community:
        girvan_newman_community[node] = i

plt.title("Girvan Newman communities")
plt.show()

girvan_newman_modularity = nx.community.quality.modularity(graph_x, communities)
print(f"Girvan Newman modularity: {girvan_newman_modularity}")

# k_clique_communities
communities = nx.community.k_clique_communities(graph_x, k=3)

pos = nx.spring_layout(graph_x)
k_clique_community = {}

# in k_clique 1 node can be in multiple communities. - plt have problem with this
already_drawn = []
for i, community in enumerate(communities):
    nx.draw_networkx_edges(graph_x, pos, edgelist=graph_x.edges(community), edge_color=colors[i])
    nx.draw_networkx_nodes(graph_x, pos, nodelist=community, node_color=colors[i])
    for node in community:
        if node not in k_clique_community:
            k_clique_community[node] = [i]
        else:
            k_clique_community[node].append(i)

plt.title("K clique communities")
plt.show()

# in k_clique modularity cant be calculated

# Save to csv
data = []
# CSV header
data.append(["ID", "Degree", "Closeness", "Coefficient", "Louvain community", "Label propagation community",
             "Kernighan Lin Bisection community", "Girvan Newman community", "K clique community"])
for key, value in degrees:
    data.append([key, value, closeness[key], coefficients[key], louvian_community[key], label_propagation_community[key]
                 , kernighan_lin_bisection_community[key], girvan_newman_community[key], k_clique_community.get(key)])

# use delimiter `;` to separate values
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerows(data)







