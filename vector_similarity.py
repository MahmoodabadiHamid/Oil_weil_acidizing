import pandas as pd
import os
os.chdir("dataSet")
import math
import matplotlib.pyplot as plt
from collections import Counter
class Well:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

def vector_len(v):
    return math.sqrt(sum([x*x for x in v]))

def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([x*y for (x,y) in zip(v1, v2)])

def euclidean_distance_similarity(v1, v2):
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(v1, v2)))

def manhattan_distance(v1,v2):
    return sum(abs(a-b) for a,b in zip(v1,v2))

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))

def plot_k_most_similar_to_well(dataSet, well_name, k):
    plt.figure(figsize=(15,7))
    v = [ well.vector for well in dataSet if well.name == well_name][0]
    
    plt.subplot(221)
    plt.title('Base well time serie')
    plt.plot(range(len(v)), v[0:], label = str(well_name))
    plt.legend()
    dict ={}
    for i in dataSet:
        dict[i.name] = cosine_similarity(i.vector, v)
    d = Counter(dict)
    plt.subplot(222)
    plt.title('Cosine similarity')
    for name, vector in d.most_common(k):
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.plot(range(len(vector[0:])), vector[0:], label = str(name))
        plt.legend()


    dict ={}
    for i in dataSet:
        dict[i.name] = euclidean_distance_similarity(i.vector, v)
    d = Counter(dict)
    plt.subplot(223)
    plt.title('Euclidian similarity')
    for name, vector in (d.most_common()[:-k-1:-1]):
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.plot(range(len(vector[0:])), vector[0:], label = str(name))
        plt.legend()

    dict ={}
    for i in dataSet:
        dict[i.name] = manhattan_distance(i.vector, v)
    d = Counter(dict)
    plt.subplot(224)
    plt.title('Manhatan similarity')
    for name, vector in d.most_common()[:-k-1:-1]:
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.plot(range(len(vector[0:])), vector[0:], label = str(name))
        plt.legend()
    plt.show()


def readDataSet():
    file_name = '02_preprocessing_after_normalizing_values.xlsx'
    df = pd.read_excel(file_name, index_col=0, index=False)
    
    dataSet = []
    for i in range(df.shape[0]):
        well = str('well ' + str(i))
        wellVector = list(df.iloc[i].values)
        dataSet.append(Well(well, wellVector))
    return dataSet

dataSet = readDataSet()
print('here')
plot_k_most_similar_to_well(dataSet, 'well 0', 5)





