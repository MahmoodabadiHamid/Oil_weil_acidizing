import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("dataSet")
import pandas as pd
from pandas.plotting import scatter_matrix

def labelEncoder(df):
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    df["Oil/Gas"] = lb_make.fit_transform(df["Oil/Gas"])
    df["Open/Cased"] = lb_make.fit_transform(df["Open/Cased"])
    df["Water/Oil-Based"] = lb_make.fit_transform(df["Water/Oil-Based"])
    df["Loss/Gain"] = lb_make.fit_transform(df["Loss/Gain"])
    df["Carbonate/Sandstone"] = lb_make.fit_transform(df["Carbonate/Sandstone"])
    df["Wettability"] = lb_make.fit_transform(df["Wettability"])
    df["Oil/Gas"] = lb_make.fit_transform(df["Oil/Gas"])
    df.to_excel('01_preprocessing_after_encoding_label.xlsx')
    return df


def normalizeValues(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x = df.values
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.to_excel('02_preprocessing_after_normalizing_values.xlsx')
    return df


def plot_corr(df,size=10):
    corr = df.corr()
    pl, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                     linewidths=.05, annot_kws={"size": 7})
    hm.set_xticklabels(corr.columns, rotation = 75, fontsize = 8)
    hm.set_yticklabels(corr.columns, rotation = 30, fontsize =11)
    pl.subplots_adjust(top=0.93, left=0.15, bottom = 0.2)
    t= pl.suptitle('Well Attributes Correlation Heatmap', fontsize=14)
    pl.savefig('Correlation_Plot.jpg')
    plt.show()
    
def featureCorrelationRanking(df):
    c = df.corr()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")
    so = pd.DataFrame(so.dropna())
    so.to_excel("feture_correlation_ranking.xlsx")
    return so
    
def PCA(X, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_components)
    X = pca.fit(X).transform(X)
    return  pd.DataFrame(X)


def save_box_plot(df1, df2):
    arr1 = np.array(df1)
    arr2 = np.array(df2)
    for i in range((arr1.shape[1])):
        fig = plt.figure(1)
        fig.suptitle(str(df1.columns[i]), fontsize=20)
        plt.subplot(121)
        plt.boxplot(arr1[:, i])
        plt.subplot(122)
        plt.boxplot(arr2[:, i])
        fig.savefig(str('../figures/fig' + str(i) + '.jpg'))
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        
def draw_pair_wise_scatter_plots(df):
    sns.pairplot(df)
    plt.show()
#____________________________________________________________ Step One ___________________________________________
file_name = 'preprocessed.xlsx'
df = pd.read_excel(file_name, index_col=0, index=False)

#____________________________________________________________ Step Two ___________________________________________
enc_df = labelEncoder(df)
print(enc_df.iloc[0])
#____________________________________________________________ Step Three ___________________________________________
#norm_df = normalizeValues(df)

#____________________________________________________________ Step Four ___________________________________________
#save_box_plot(enc_df, norm_df)

#____________________________________________________________ Step Five ___________________________________________
#plot_corr(enc_df,size=10)

#____________________________________________________________ Step Six ___________________________________________

#file_name = '02_preprocessing_after_normalizing_values.xlsx'
#df = pd.read_excel(file_name, index_col=0, index=False)

#____________________________________________________________ Step Seven ___________________________________________
#draw_pair_wise_scatter_plots(enc_df)

#____________________________________________________________ Step Eight ___________________________________________



#pca_df = pd.DataFrame(df)#PCA(df, 2))





#pca_df.plot(style=['o','rx'])

#plt.show()

#scatter_matrix(pca_df, alpha=0.2)













