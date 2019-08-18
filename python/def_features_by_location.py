import math
import pandas as pd
import os
from collections import Counter
from collections import OrderedDict

os.chdir("../Dataset/Second dataset")

#Counter(lst).most_common(1)[0][0]


def calculate_distance(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Location():
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
#, columns, loc, k


class Well():
    def __init__(self, name, loc, vector):
        self.name = name
        self.location = loc
        self.vector = vector


def readDataset():
    file_name = '022_preprocessing_after_normalizing_values.xlsx'
    df = pd.read_excel(file_name, index=False)
    dataSet = []
    for i in range(df.shape[0]):
        well = str('well_' + str(int(df.iloc[i]['Id'])) + '_'+ str(int(df.iloc[i]['Level'])))
        loc = Location (df.iloc[i]['Latitude'], df.iloc[i]['Longitude'])
        wellVector = (df.drop(columns = ["Id", 'Latitude', 'Longitude']).iloc[i])
        dataSet.append(Well(well, loc, wellVector))
    columns = list(df.drop(columns = ["Id", 'Longitude', 'Latitude']).columns)
    return dataSet, columns


def k_nearest_well(loc, k):
    import operator
    import matplotlib.pyplot as plt
    dataSet, columns = readDataset()
    vector_output = Well('output_vec', loc, pd.DataFrame(columns = columns))
    d = {}
    for i in range(len(dataSet)):
        d[dataSet[i].name] = calculate_distance(dataSet[i].location.lat, dataSet[0].location.lat,
                                                   dataSet[i].location.lon, dataSet[0].location.lon)
    d = dict(sorted(d.items(), key=operator.itemgetter(1))[:k])

    '''
    plt.title('Based on location feature assigning')
    for i in range(len(columns)):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5)
    
    for name, vector in Counter(d).most_common(k):
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.scatter(range(len(vector[0:])), vector[0:], label = str(name) + str())
        plt.legend( loc='right', fancybox=True)
        
    for i, txt in enumerate(columns):
            plt.annotate(txt, (i, 0), rotation = -90, fontsize = 'x-small')
    plt.show()
    '''
    return vector_output


lc = Location(10, 10)
vc = k_nearest_well(lc, k = 20)














