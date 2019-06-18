import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("dataSet")
import pandas as pd


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
    import matplotlib
    plt.imshow(corr, cmap=plt.cm.seismic)
    plt.colorbar()
    tick_marks = [i for i in range(len(corr.columns))]
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical',verticalalignment='top');
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.savefig('../figures/Correlation_matrix.jpg')
    plt.show()
    

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

file_name = 'preprocessed.xlsx'
df = pd.read_excel(file_name, index_col=0, index=False)

enc_df = labelEncoder(df)

norm_df = normalizeValues(df)

#save_box_plot(enc_df, norm_df)

plot_corr(enc_df,size=10)

















