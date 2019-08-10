import networkx
import pandas as pd
import os
os.chdir('..')

os.chdir('Dataset')
os.chdir('Second Dataset')
data = pd.read_csv('Arc.csv')



print(data.shape)
