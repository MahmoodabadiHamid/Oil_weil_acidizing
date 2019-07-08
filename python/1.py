import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline

df = pd.read_excel (r'E:\company\python\Oil_well_acidizing\DataSet\02_preprocessing_after_normalizing_values.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'

#df.head()
#print(df.info())
#print(df.describe())
#df["Level"].value_counts()
#df.hist(bins=5,figsize=(20,15))
#plt.show()

#correlation matrix

sns.heatmap(
    data=df.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(20, 16)

#plt.show()
#mutually information
#print(df[1:])
x=df.Level
y=df.Id
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
calc_MI(df.Longitude,df.Latitude,1)
#print(df.iloc[:,-26])
for i in range(1,26):
   for j in range(1,26):
      print( i,j,calc_MI(df.iloc[:,-i],df.iloc[:,-j],500))
#print(MI())
#print(calc_MI(df.iloc[:,-26],df.iloc[:,-24],5))
       
   

   






