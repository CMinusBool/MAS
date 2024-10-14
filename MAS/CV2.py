import math
import os
import sys

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

# find number of nodes in data (assuming there are numbered in a series)
number_of_nodes = df.max().max()

print(f"Number of nodes in data is: {number_of_nodes}")

adjacencyMatrix = [[0] * number_of_nodes for _ in range(number_of_nodes)]

# populate adjecency matrix
for _, row in df.iterrows():
    adjacencyMatrix[row[0]-1][row[1]-1] = 1
    adjacencyMatrix[row[1] - 1][row[0] - 1] = 1

# initialize minimal distance matrix
minimal_distance = adjacencyMatrix.copy()
# put max int where there are no edges and it isn't on diagonal
x = 0
for row in minimal_distance:
    y = 0
    for element in row:
        if element == 0 and y is not x:
            minimal_distance[x][y] = math.inf
        y += 1
    x += 1

print(f"Initialized minimal distance matrix:")
for row in minimal_distance:
    print(row)

# calculating minimal distances in graph using Floyd-Marshal alg
for k in range(0, number_of_nodes):
    for i in range(0, number_of_nodes):
        for j in range(0, number_of_nodes):
            if minimal_distance[i][j] > minimal_distance[i][k] + minimal_distance[k][j]:
                minimal_distance[i][j] = minimal_distance[i][k] + minimal_distance[k][j]

print(f"Calculated minimal distance:")
for row in minimal_distance:
    print(row)

# calculate eccentricity
eccentricity = []
for node in minimal_distance:
    max_distance = 0
    for distance in node:
        if distance is not math.inf and distance > max_distance:
            max_distance = distance
    eccentricity.append(max_distance)

#print(f"eccentricity: {eccentricity}")

# calculate diameter
diameter = max(eccentricity)
print(f"Diameter: {diameter}")

# calculate mean average distance
sum_of_minimal_distance = 0
for row in minimal_distance:
    sum_of_minimal_distance += sum(row)

mean_average_distance = sum_of_minimal_distance * (1/(number_of_nodes*(number_of_nodes-1)))
print(f"Prumerna vzdalenost: {mean_average_distance}")

#closeness centrality
closeness_centralities = []
for row in minimal_distance:
    sum_of_minimal_distance = sum(row)
    closeness_centralities.append(number_of_nodes/sum_of_minimal_distance)

print("Closeness centralities")
print("ID   |   Closeness centrality")
i = 1
for element in closeness_centralities:
    print(f"{i}   |  {element}")
    i += 1



