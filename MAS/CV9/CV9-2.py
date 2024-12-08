import networkx as nx
import matplotlib.pyplot as plt

# Parameters
num_nodes = 5500
probability_for_edge = 0.001  # Probability for edge creation in random graph
m = 2  # Number of edges to attach from a new node to existing nodes in preferential attachment model

# Generate random graph using the Erdős-Rényi model
random_graph = nx.erdos_renyi_graph(num_nodes, probability_for_edge)

# Generate preferential attachment graph using the Barabási-Albert model
preferential_attachment_graph = nx.barabasi_albert_graph(num_nodes, m)

# degree distributions

# degree distribution in log scale

# cdf a ccdf

# plot the degree distributions, cdf, ccdf and fit a power law, poisson, exponential distribution and normal distribution