import numpy as np
import os
os.chdir("../Dataset/Second dataset")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn as sns
from itertools import combinations



def vector_len(v):
    return math.sqrt(sum([x*x for x in v]))


def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([x*y for (x,y) in zip(v1, v2)])

def mean(x):
    return sum(x)/len(x)

def covariance(x,y):
    calc = []
    for i in range(len(x)):
        xi = x[i] - mean(x)
        yi = y[i] - mean(y)
        calc.append(xi * yi)
    return sum(calc)/(len(x) - 1)


#___________________________ Pearson __________________
import math
def stDev(x):
    variance = 0
    for i in x:
        variance += (i - mean(x) ** 2) / len(x)
    return math.sqrt(variance)
    
def Pearsons(x,y):
    cov = covariance(x,y)
    return cov / (stDev(x) * stDev(y))



#_________________________ cosTheta ___________________
def dotProduct(x,y):
    calc = 0
    for i in range(len(x)):
        calc += x[i] * y[i]
    return calc

def magnitude(x):
    x_sq = [i ** 2 for i in x]
    return math.sqrt(sum(x_sq))

def cosTheta(x,y):
    mag_x = magnitude(x)
    mag_y = magnitude(y)
    return dotProduct(x,y) / (mag_x * mag_y)



#_________________________ Pearson algorithm _________
def pearson(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))



#_________________________ Spearman algorithm ________
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))



#_________________________ Kendall algorithm _________
def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0 #concordant count
    d = 0 #discordant count
    t = 0 #tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)



#_________________________euclidean ________________
def euclidean(v1, v2):
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(v1, v2)))

#_________________________ Manhatan ________________
def manhattan(v1,v2):
    return sum(abs(a-b) for a,b in zip(v1,v2))

#_________________________ Manhatan ________________
def cosine(v1, v2):
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))




df = pd.read_excel("022_preprocessing_after_normalizing_values.xlsx")
df = df.drop(['Id', 'Level', 'Loss/Gain'], axis=1)
cols = df.columns
print(cols)

corr = np.zeros([df.shape[1], df.shape[1]])
corr_pearson = np.zeros([df.shape[1], df.shape[1]])
corr_cos = np.zeros([df.shape[1], df.shape[1]])
corr_spearman = np.zeros([df.shape[1], df.shape[1]])
corr_kendall = np.zeros([df.shape[1], df.shape[1]])
corr_euclidean = np.zeros([df.shape[1], df.shape[1]])
corr_manhatan = np.zeros([df.shape[1], df.shape[1]])
corr_correlation = np.zeros([df.shape[1], df.shape[1]])
corr_mut_info = np.zeros([df.shape[1], df.shape[1]])


for i in range(len(cols)):
    print(i)
    for j in range(len(cols)):
        corr[i, j] += pearson(df[cols[i]].values, df[cols[j]].values)
        corr[i, j] += cosine(df[cols[i]].values, df[cols[j]].values)
        corr[i, j] += spearman(df[cols[i]].values, df[cols[j]].values)
        corr[i, j] += kendall(df[cols[i]].values, df[cols[j]].values)
        corr[i, j] += euclidean(df[cols[i]].values, df[cols[j]].values)
        corr[i, j] += manhattan(df[cols[i]].values, df[cols[j]].values)
        #corr[i, j] += correlation(df[cols[i]].values, df[cols[j]].values)
        #corr[i, j] += mut_info(df[cols[i]].values, df[cols[j]].values)

corr = corr/6
pl, ax = plt.subplots(figsize=(20, 15))
corr = corr.astype(np.int64)
hm = sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm",fmt='.0f', linewidths=.05, annot_kws={"size": 7})
hm.set_xticklabels(cols, rotation = 75, fontsize = 8)
hm.set_yticklabels(cols, rotation = 30, fontsize =11)
pl.subplots_adjust(top=0.93, left=0.15, bottom = 0.2)
t = pl.suptitle('Well Attributes Correlation Heatmap (our method)', fontsize=14)
pl.savefig('manhattan_Correlation_Plot_SUM_2.jpg')
plt.show()




