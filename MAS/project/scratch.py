from pyvis.network import Network
import networkx as nx

# test network
G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 1)

# create a network
nt = Network("500px", "500px")
nt.from_nx(G)
nt.show("example.html")