from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def normalizeValues(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    return x_scaled

   
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

def main():
   from sklearn import preprocessing
   os.chdir("../Dataset/Second dataset")
   file_name = '022_preprocessing_after_normalizing_values.xlsx'
   df = pd.read_excel(file_name, index = True)
   corr_df = pd.DataFrame({"First Feature":[], "Second Feature":[], "Correlation":[]})
   df = df.drop(['Id', 'Level'], axis=1)
   fig = plt.gcf()
   fig.set_size_inches(20, 16)
   mutual_matrix = np.zeros([df.shape[1],df.shape[1]])
   for i in range(df.shape[1]):
      for j in range(df.shape[1]):
         mutual_matrix[i, j] = calc_MI(df.iloc[:,-i],df.iloc[:,-j],500)
         
         corr_df = corr_df.append({
            "First Feature":[(df.columns[i])],
            "Second Feature":[(df.columns[j])],
            "Correlation" : round(mutual_matrix[i, j],2)
                    }, ignore_index = True)
   corr_df.to_excel("second_dataset_non_linear_correlation_ranking.xlsx")


   normalizedMutualMatrix = np.zeros([df.shape[1],df.shape[1]])
   for i in range(mutual_matrix.shape[0]):
      normalizedMutualMatrix[i] = [i[0] for i in normalizeValues(mutual_matrix[i].reshape(-1, 1))]

   pl, ax = plt.subplots(figsize=(20, 15))
   hm = sns.heatmap(mutual_matrix, annot=True, ax=ax, cmap=ListedColormap(['white']),fmt='.2f',
                     linewidths=.05, annot_kws={"size": 7}, cbar = False)
   hm.set_xticklabels(df.columns, rotation = 75, fontsize = 8)
   hm.set_yticklabels(df.columns, rotation = 30, fontsize =11)
   pl.subplots_adjust(top=0.93, left=0.15, bottom = 0.2)
   t= pl.suptitle('Well Attributes non-linear Correlation Heatmap', fontsize=14)
   pl.savefig('non_linear_Correlation_Plot.jpg')
   plt.show()

if __name__ == '__main__':
    main()











         
#############################################################
##   min_max_scaler = preprocessing.MinMaxScaler()
##   mutual_matrix = min_max_scaler.fit_transform(mutual_matrix)
##   
##   corr_df[["Correlation"]] = min_max_scaler.fit_transform(corr_df[["Correlation"]])
##   corr_df.to_excel("second_dataset_non_linear_correlation_ranking.xlsx")
##   
##   pl, ax = plt.subplots(figsize=(20, 15))
##   hm = sns.heatmap(np.round(mutual_matrix,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
##                     linewidths=.05, annot_kws={"size": 7})
##   hm.set_xticklabels(df.columns, rotation = 75, fontsize = 8)
##   hm.set_yticklabels(df.columns, rotation = 30, fontsize =11)
##   pl.subplots_adjust(top=0.93, left=0.15, bottom = 0.2)
##   t= pl.suptitle('Well Attributes non-linear Correlation Heatmap', fontsize=14)
##   pl.savefig('non_linear_Correlation_Plot.jpg')
##   #plt.show()
##    
## 
##if __name__ == '__main__':
##    main()
##
##
##
##
##
