import numpy as np
import networkx as nx
from networkx.classes import degree
import matplotlib.pyplot as plt
from pyvis.network import Network
from infomap import Infomap

#function for parsing mtx file
def load_mtx(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split() for line in lines]
        lines = lines[2:]
        lines = [[int(line[0]), int(line[1])] for line in lines]
    return lines

#function for converting mtx to graph
def mtx_to_graph(mtx):
    G = nx.Graph()
    for edge in mtx:
        G.add_edge(edge[0], edge[1])
    return G

#load socfb-GWU54.mtx
mtx = load_mtx("socfb-GWU54.mtx")
G = mtx_to_graph(mtx)
# # #I
# # print(f"Pocet vrcholu: {G.number_of_nodes()}")
# # print(f"Pocet hran: {G.number_of_edges()}")
# # print(f"Hustota: {nx.density(G)}")
# #II
# print(f"Prumerny stupen: {np.mean([d for n, d in G.degree()])}")
# print(f"Maximalni stupen: {np.max([d for n, d in G.degree()])}")
# print(f"Median stupnu: {np.median([d for n, d in G.degree()])}")
# degree_distribution = [d for n, d in G.degree()]
# #make histogram
# plt.hist(degree_distribution, bins = np.max(degree_distribution))
# plt.title("Histogram distribuce stupňů")
# plt.xlabel("Stupeň")
# plt.ylabel("Počet vrcholů")
# plt.xscale("log")
# plt.yscale("log")
# plt.show()
# #III
# #centrality
# # Degree centrality
# degree_centrality = nx.degree_centrality(G)
# print(f"Průměrná degree centrality: {np.mean([v for v in degree_centrality.values()])}")
# #graf pro degree centrality s body pro kazdou cetnost centrality
# degree_centrality_distribution = [v for v in degree_centrality.values()]
# degree_centrality_counts = np.unique(degree_centrality_distribution, return_counts=True)
# plt.scatter(degree_centrality_counts[0], degree_centrality_counts[1], s=3)
# plt.title("Degree centrality distribuce")
# plt.xlabel("Degree centrality")
# plt.ylabel("Počet vrcholů")
# plt.yscale("log")
# plt.show()
#
# # Eigenvector centrality
# eigenvector_centrality = nx.eigenvector_centrality(G)
# print(f"Average eigenvector centrality: {np.mean([v for v in eigenvector_centrality.values()])}")
# #graf pro eigenvector centrality s body pro kazdou cetnost centrality
# eigenvector_centrality_distribution = [v for v in eigenvector_centrality.values()]
# eigenvector_centrality_counts = np.unique(eigenvector_centrality_distribution, return_counts=True)
# plt.scatter(eigenvector_centrality_counts[0], eigenvector_centrality_counts[1], s=3)
# plt.title("Eigenvector centrality distribution")
# plt.xlabel("Eigenvector centrality")
# plt.ylabel("Počet vrcholů")
# plt.show()

# # IV
# # shlukovaci koeficient
# # prumerny shlukovaci koeficient
# clustering_coefficient = nx.clustering(G)
# print(f"Prumerny shlukovaci koeficient: {np.mean([v for v in clustering_coefficient.values()])}")
# # shlukovaci efekt (prumerny shlukovaci koeficient vzhledem k stupni)
# degree_to_cc = {}  # Slovník: stupeň -> seznam CC
# for node, degree in G.degree():
#     cc = clustering_coefficient[node]
#     if degree not in degree_to_cc:
#         degree_to_cc[degree] = []
#     degree_to_cc[degree].append(cc)
#
# # Vypočítat průměrný CC pro každý stupeň
# avg_cc_by_degree = {degree: np.mean(ccs) for degree, ccs in degree_to_cc.items()}
#
# # 4. Vizualizace shlukovacího efektu
# plt.figure(figsize=(8, 6))
# plt.scatter(avg_cc_by_degree.keys(), avg_cc_by_degree.values(), color="blue", label="Průměrný CC", s=5)
# plt.xlabel("Stupeň vrcholu (Degree)")
# plt.ylabel("Průměrný shlukovací koeficient (CC)")
# plt.title("Shlukovací efekt (Clustering Effect)")
# plt.grid(True)
# plt.legend()
# plt.show()

# #V
# # Souvislost - počet souvislých komponent a distribuce jejich velikostí.
# # Počet souvislých komponent
# connected_components = list(nx.connected_components(G))  # Seznam komponent (množin vrcholů)
# num_components = len(connected_components)
# print(f"Počet souvislých komponent: {num_components}")
#
# # Velikosti komponent (počet vrcholů v každé komponentě)
# component_sizes = [len(component) for component in connected_components]
# print(f"Velikosti komponent: {component_sizes}")
#
# # Distribuce velikostí komponent
# component_size_counts = np.unique(component_sizes, return_counts=True)
# plt.scatter(component_size_counts[0], component_size_counts[1], s=10)
# plt.title("Distribuce velikostí komponent")
# plt.xlabel("Velikost komponenty")
# plt.ylabel("Počet komponent")
# plt.xscale("log")
# plt.show()

# VI
# Vizualizace grafu pomoci gephi
# export do souboru
#nx.write_gexf(G, "graph.gexf")

# # 2
# #I Komunity
# # Louvain
# print("Louvain")
# louvain_communities = nx.algorithms.community.louvain_communities(G)
# print(f"Pocet komunit = {len(louvain_communities)}")
# # prumerna velikost komunity
# avg_community_size = np.mean([len(community) for community in louvain_communities])
# print(f"Prumerna velikost komunity = {avg_community_size}")
# # minimalni velikost komunity
# min_community_size = np.min([len(community) for community in louvain_communities])
# print(f"Minimalni velikost komunity = {min_community_size}")
# # maximalni velikost komunity
# max_community_size = np.max([len(community) for community in louvain_communities])
# print(f"Maximalni velikost komunity = {max_community_size}")
# # modularity
# modularity = nx.algorithms.community.quality.modularity(G, louvain_communities)
# print(f"Modularity = {modularity}")
# #II
# # distribuce velikosti komunit
# community_sizes = [len(community) for community in louvain_communities]
# community_size_counts = np.unique(community_sizes, return_counts=True)
# plt.scatter(community_size_counts[0], community_size_counts[1], s=10)
# plt.title("Distribuce velikosti komunit")
# plt.xlabel("Velikost komunity")
# plt.ylabel("Počet komunit")
# plt.xscale("log")
# plt.show()
# #III
# # export komunit do gephi
# community_dict = {}
# for i, community in enumerate(louvain_communities):
#     for node in community:
#         community_dict[node] = i
# nx.set_node_attributes(G, community_dict, "community")
# nx.write_gexf(G, "graph_communities_louvain.gexf")

# #3
# #I kuminity
# # Label propagation
# print("Label propagation")
# label_propagation_communities_dict = nx.algorithms.community.label_propagation_communities(G)
# label_propagation_communities = [list(community) for community in label_propagation_communities_dict]
# print(f"Pocet komunit = {len(label_propagation_communities)}")
# # prumerna velikost komunity
# avg_community_size = np.mean([len(community) for community in label_propagation_communities])
# print(f"Prumerna velikost komunity = {avg_community_size}")
# # minimalni velikost komunity
# min_community_size = np.min([len(community) for community in label_propagation_communities])
# print(f"Minimalni velikost komunity = {min_community_size}")
# # maximalni velikost komunity
# max_community_size = np.max([len(community) for community in label_propagation_communities])
# print(f"Maximalni velikost komunity = {max_community_size}")
# # modularity
# modularity = nx.algorithms.community.quality.modularity(G, label_propagation_communities)
# print(f"Modularity = {modularity}")
# #II
# # distribuce velikosti komunit
# community_sizes = [len(community) for community in label_propagation_communities]
# community_size_counts = np.unique(community_sizes, return_counts=True)
# y = np.arange(0, 32, 1)
# plt.scatter(community_size_counts[0], community_size_counts[1], s=10)
# plt.title("Distribuce velikosti komunit")
# plt.xlabel("Velikost komunity")
# plt.ylabel("Počet komunit")
# plt.xscale("log")
# plt.yticks(y)
# plt.show()
# #III
# # export komunit do gephi
# community_dict = {}
# for i, community in enumerate(label_propagation_communities):
#     for node in community:
#         community_dict[node] = i
# nx.set_node_attributes(G, community_dict, "community")
# nx.write_gexf(G, "graph_communities_gn.gexf")

# infomap
print("Infomap")
infomap = Infomap()
for edge in G.edges():
    infomap.addLink(*edge)
infomap.run()
infomap_communities_dict = infomap.getModules()
#info map vraci data ve formatu {node: community}
infomap_communities = []
for node, community in infomap_communities_dict.items():
    while len(infomap_communities) <= community:
        infomap_communities.append([])
    infomap_communities[community].append(node)

print(f"Pocet komunit = {len(infomap_communities)}")
# prumerna velikost komunity
avg_community_size = np.mean([len(community) for community in infomap_communities])
print(f"Prumerna velikost komunity = {avg_community_size}")
# minimalni velikost komunity
min_community_size = np.min([len(community) for community in infomap_communities])
print(f"Minimalni velikost komunity = {min_community_size}")
# maximalni velikost komunity
max_community_size = np.max([len(community) for community in infomap_communities])
print(f"Maximalni velikost komunity = {max_community_size}")
# modularity
modularity = nx.algorithms.community.quality.modularity(G, infomap_communities)
print(f"Modularity = {modularity}")
# distribuce velikosti komunit
community_sizes = [len(community) for community in infomap_communities]
community_size_counts = np.unique(community_sizes, return_counts=True)
plt.scatter(community_size_counts[0], community_size_counts[1], s=10)
plt.title("Distribuce velikosti komunit")
plt.xlabel("Velikost komunity")
plt.ylabel("Počet komunit")
plt.xscale("log")
plt.show()
# export komunit do gephi
community_dict = {}
for i, community in enumerate(infomap_communities):
    for node in community:
        community_dict[node] = i
nx.set_node_attributes(G, community_dict, "community")
nx.write_gexf(G, "graph_communities_infomap.gexf")




