import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_degrees(matrix):
    degrees = []
    for row in matrix:
        degrees.append(np.sum(row))
    degrees.sort()
    return degrees

def calculate_degrees_dict(adjacency_List):
    degrees = []
    for _, node in adjacency_List.items():
        degrees.append(len(node))
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

# adjacency list
adjacency_List = {}
for i in range(number_of_nodes):
    adjacency_List[i] = []

for _, row in df.iterrows():
    adjacency_List[row[0]-1].append(row[1]-1)
    adjacency_List[row[1] - 1].append(row[0] - 1)
print(f"Adjacency list: {adjacency_List}")


# degrees
print(f"Adjecency matrix:  {adjacencyMatrix}")
degrees = np.array(calculate_degrees(adjacencyMatrix))
degrees_adj_lst = calculate_degrees_dict(adjacency_List)
print(degrees_adj_lst)
maximum = degrees.max()
maximum_adj_lst = max(degrees_adj_lst)
print("Min: ")
print(degrees.min())

print(f"Min from Adjecency list: {min(degrees_adj_lst)}")

print("Max: ")
print(maximum)
print(f"Max from Adjecency list: {maximum_adj_lst}")

print("Average: ")
print(np.average(degrees))
print(f"Average from Adjecency list: {np.average(degrees_adj_lst)}")

frequencies = np.unique(degrees, return_counts=True)

# print all degrees from 0 to max and convert to pandas
frequencies_iter = 0
frequencies_df = pd.DataFrame(data=None,columns=['degree', 'frequency', 'relative_frequency'])
print("Pocet | cetnost | Relativni cetnost")
for degree in range(maximum):
    if degree == frequencies[0][frequencies_iter]:
        frequencies_df.loc[len(frequencies_df)] = [degree, frequencies[1][frequencies_iter], frequencies[1][frequencies_iter] / len(adjacencyMatrix)]
        print(degree, "   |   ", frequencies[1][frequencies_iter], "   |   ", frequencies[1][frequencies_iter] / len(adjacencyMatrix))
        frequencies_iter += 1
    else:
        frequencies_df.loc[len(frequencies_df)] = [degree, 0, 0]
        print(degree, "   |   ", 0, "   |   ", 0)


print("Pocet | cetnost | Relativni cetnost - From adjecency list")
frequencies_iter = 0
frequencies_df_adj_lst = pd.DataFrame(data=None,columns=['degree', 'frequency', 'relative_frequency'])
frequencies_adj_lst = np.unique(degrees_adj_lst, return_counts=True)
for degree in range(maximum_adj_lst):
    if degree == frequencies_adj_lst[0][frequencies_iter]:
        frequencies_df_adj_lst.loc[len(frequencies_df_adj_lst)] = [degree, frequencies_adj_lst[1][frequencies_iter], frequencies_adj_lst[1][frequencies_iter] / len(adjacency_List)]
        print(degree, "   |   ", frequencies_adj_lst[1][frequencies_iter], "   |   ", frequencies_adj_lst[1][frequencies_iter] / len(adjacency_List))
        frequencies_iter += 1
    else:
        frequencies_df_adj_lst.loc[len(frequencies_df_adj_lst)] = [degree, 0, 0]
        print(degree, "   |   ", 0, "   |   ", 0)

#Histogram frequency
plt.bar(frequencies_df['degree'], frequencies_df['frequency'])
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Frequency Histogram')
plt.xticks(np.arange(0, maximum, 1))
plt.yticks(np.arange(0, frequencies_df['frequency'].max()+1, 1))
plt.show()

#Histogram relative frequency
plt.bar(frequencies_df['degree'], frequencies_df['relative_frequency'])
plt.xlabel('Degree')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequency Histogram')
plt.show()

#Histogram frequency from adjecency list
plt.bar(frequencies_df_adj_lst['degree'], frequencies_df_adj_lst['frequency'])
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Frequency Histogram from adjecency list')
plt.xticks(np.arange(0, maximum_adj_lst, 1))
plt.yticks(np.arange(0, frequencies_df_adj_lst['frequency'].max()+1, 1))
plt.show()

#Histogram relative frequency from adjecency list
plt.bar(frequencies_df_adj_lst['degree'], frequencies_df_adj_lst['relative_frequency'])
plt.xlabel('Degree')
plt.ylabel('Relative Frequency')
plt.title('Relative Frequency Histogram from adjecency list')
plt.show()


