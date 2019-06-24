import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
df = pd.read_excel (r'E:\company\python\Oil_well_acidizing\DataSet\02_preprocessing_after_normalizing_values.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
#print (df)
df.head()
print(df.info())
print(df.describe())
df["Level"].value_counts()
df.hist(bins=5,figsize=(20,15))
plt.show()

