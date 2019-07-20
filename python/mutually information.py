import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.chdir("../Dataset/Second dataset")

file_name = '022_preprocessing_after_normalizing_values.xlsx'
df = pd.read_excel(file_name, index = True)

sns.heatmap(
    data=df.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(20, 16)

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI
def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

  
for i in range(1,26):
   for j in range(1,26):
     print(i,j,calc_MI(df.iloc[:,-i],df.iloc[:,-j],500))

       
   

   






