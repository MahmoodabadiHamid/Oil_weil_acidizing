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

plt.show()





