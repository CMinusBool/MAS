import math
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import csv

import pandas as pd
import numpy as np

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
print("Coefficients of nodes:")
for i in range(1, 35):
    print(f"{i}, {coefficients[i]}")
    sum_coefficients += coefficients[i]

# average coefficient
average_coef = sum_coefficients / 34
print(f"Average coeficient: {average_coef}")

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

# plot average coefficient by degree
x, y = zip(*average_coef_by_degree)
plt.plot(x, y, 'o')
plt.show()

# calculate closenes centrality
closeness = nx.closeness_centrality(graph_x)

# save ID, degree, coeficients to csv
data = []
for key, value in degrees:
    data.append([key, value, closeness[key],coefficients[key]])

# use delimiter `;` to separate values
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerows(data)







