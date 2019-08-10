import random
import pandas as pd
import os
os.chdir("../Dataset/Second dataset")
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

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))

def plot_k_most_similar_to_well(dataSet, well_name, k):
    v = [ well.vector for well in dataSet if well.name == well_name][0]
    dict ={}
    for i in dataSet:
        dict[i.name] = cosine_similarity(i.vector, v)
    d = Counter(dict)
    return dict(d)


def readDataSet():
    file_name = '022_preprocessing_after_normalizing_values.xlsx'
    df = pd.read_excel(file_name, index=False)
    dataSet = []
    for i in range(df.shape[0]):
        well = str('well_' + str(int(df.iloc[i]['Id'])) + '_'+ str(int(df.iloc[i]['Level'])))
        wellVector = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).iloc[i].values)
        dataSet.append(Well(well, wellVector))
    columns = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).columns)
    return dataSet, columns
dataSet, columns = readDataSet()
        
d = plot_k_most_similar_to_well(dataSet, "well_"+str(random.randint(1, (100)))+"_" + str(random.randint(1, 5)), 10)




