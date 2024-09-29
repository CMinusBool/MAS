import pandas as pd
import numpy as np

def calculate_degrees(matrix):
    degrees = []
    for row in matrix:
        degrees.append(np.sum(row))
    degrees.sort()
    return degrees



df = pd.read_csv('C:/Users/kadip/Documents/skoly/ING/MAS/KarateClub.csv', header=None, sep=';')

people = 34
adjacencyMatrix = [[0] * people for _ in range(people)]

for index, row in df.iterrows():
    adjacencyMatrix[row[0]-1][row[1]-1] = 1
    adjacencyMatrix[row[1] - 1][row[0] - 1] = 1

# degrees
print(adjacencyMatrix)
degrees = np.array(calculate_degrees(adjacencyMatrix))
max = degrees.max()
print("Min: ")
print(degrees.min())

print("Max: ")
print(max)

print("Average: ")
print(np.average(degrees))

print("Pocet | cetnost | Relativni cetnost")
cetnosti = np.unique(degrees, return_counts=True)
for i in range(len(cetnosti[0])):
    print(str(cetnosti[0][i]), "   |   ", cetnosti[1][i], "   |   ", cetnosti[1][i]/len(adjacencyMatrix))

#dodelat histogram i s chybejicimi hodnotami. druhy typ uchovavani dat.
