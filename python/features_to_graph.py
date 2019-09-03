import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import numpy as np
os.chdir("../Dataset/third dataset")
def unique(list1): 
    x = np.array(list1) 
    return list(np.unique(x))


file_name = 'features network.xlsx'
df = pd.read_excel(file_name)
df = df.drop(['Unnamed: 0'], axis=1)

nodes = []
for item in df['from']:
	nodes.append(item)
for item in df['to']:
	nodes.append(item)
nodes = unique(nodes)


G = nx.Graph()
for i in nodes:
    G.add_node(i)
for i in range(len(df)):
        G.add_edge(df.iloc[i]['from'], df.iloc[i]['to'], w = df.iloc[0]['Average']+10)

pos = {'Bottom MD (ft)': (0, 0),
       'Bottom0hole Pressure (psi)': (0, 1),
       'Carbonate_Sandstone': (0, 2),
       'Fracture Gradient (psi_ft)': (0, 3),
       'Fracture Pressure (psi)': (0, 4),
       'Initial Skin': (0, 5),
       'Interval Length (ft)': (0, 6),
       'Latitude': (1, 0),
       'Longitude': (1, 1),
       'Loss/Gain': (1, 2),
       'Oil_Gas': (1, 3),
       'Open_Cased': (1, 4),
       'Permeability (mD)': (1, 5),
       'Porosity': (2, 1),
       'Pressure Gradient (psi_ft)': (2, 2),
       'Radius (in)': (2, 3),
       'Rock Type (DRT)': (2, 4),
       'Shot Density': (2, 5),
       'Top MD (ft)': (2, 6),
       'Water_Oil-Based': (2, 7),
       'Wettability': (2, 8)}
nx.draw_networkx_nodes(G, pos = pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G, pos)
plt.show()














