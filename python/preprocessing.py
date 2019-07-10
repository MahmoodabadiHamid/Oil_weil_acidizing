import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("../Second dataset")
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA


def labelEncoder(df, label):
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    df[label] = lb_make.fit_transform(df[label])
    df.to_excel('011_preprocessing_after_encoding_label.xlsx')
    return df


def normalizeValues(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x = df.values
    col = df.columns
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = col
    df.to_excel('022_preprocessing_after_normalizing_values.xlsx')
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
    return corr
def featureCorrelationRanking(df):
    c = df.corr()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")
    so = pd.DataFrame(so.dropna())
    so.to_excel("feture_correlation_ranking.xlsx")
    return so

def PCA_(X, n_components):
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

def plot_2d_features(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = df.drop(['Id', 'Level', 'Column Volume (ft3)'], axis=1)
    for col in df.columns:
        print(col)
        target = df[col]
        tmp_df = df#df.drop([col], axis = 1)
        pca = PCA(n_components=2)
        tmp_df = pca.fit_transform(tmp_df)
        principalDf = pd.DataFrame(data = tmp_df
                 , columns = ['pc1', 'pc2'])
        principalDf.insert(2, str(col), target.values, True)


        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x = principalDf.values
        cols = principalDf.columns
        x_scaled = min_max_scaler.fit_transform(x)
        principalDf = pd.DataFrame(x_scaled)
        principalDf.columns = cols
        
        sns.scatterplot(principalDf['pc1'], principalDf['pc2'], hue = principalDf[str(col)])
        #plt.title('Carbonate_sanstone wells based on its location')
        plt.legend()
        plt.savefig('2D_'+str(col))
        plt.show()

def number_of_optimal_k_means_classes(df):
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt


    data = df
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.savefig('Number of optimal classes')
    plt.show()

#__________________________________________________________
def k_means(df):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = df.drop(['Id', 'Level', 'Column Volume (ft3)'], axis=1)
    
    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = df, columns = ['pc1', 'pc2'])
    print(principalDf.shape)
    kmeans = KMeans(n_clusters=3, max_iter=1000).fit(df)
    sns.scatterplot(principalDf['pc1'], principalDf['pc2'], hue = kmeans.labels_)
    plt.legend()
    plt.show()
        

def evaluate_pca_effect(df):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error
    df = df.drop(['Id', 'Level', 'Column Volume (ft3)'], axis=1)
    kmeans = KMeans(n_clusters=3, max_iter=1000).fit(df)
    f24 = kmeans.labels_
    errors = {} 
    for itr in range(100):
        print(itr)
        for i in range(1, df.shape[1]):
            pca = PCA(n_components=i)
            tmp_df = pd.DataFrame(pca.fit_transform(df))
            kmeans = KMeans(n_clusters=3, max_iter=1000).fit(tmp_df)
            errors[i] = errors[i]  + mean_squared_error(f24, kmeans.labels_) if (i in errors) else mean_squared_error(f24, kmeans.labels_)
            #errors[i-1] += mean_squared_error(f24, kmeans.labels_)
        #errors.append(mean_squared_error(f24, f24))

    print(errors)
    plt.plot(*zip(*sorted(errors.items())), 'bx')
    plt.plot(*zip(*sorted(errors.items())), 'r')

    #plt.plot(zip(*sorted(errors.items())), errors, 'bx')
    #plt.plot(zip(*sorted(errors.items())), errors,  'r')
    plt.title(str(errors), fontsize=5)
    plt.xlabel('PCA Size')
    plt.ylabel('MSE')
    plt.savefig('PCA_error.png')
    plt.show()
    return errors
   
#__________________________________________________________

    

#____________________________________________________________ Step One ___________________________________________
#file_name = 'Sample Synthetic Data - Revised.xlsx'
#df = pd.read_excel(file_name, index_col=0, index=False)
cols=["Oil/Gas","Open/Cased","Water/Oil0Based","Loss/Gain","Carbonate/Sandstone","Wettability"]
#for col in cols:
 #   enc_df[col]=labelEncoder(df,"col")


#____________________________________________________________ Step Two ___________________________________________
#enc_df = labelEncoder(df,cols)
#print(enc_df.iloc[0])
#____________________________________________________________ Step Three ___________________________________________
#norm_df = normalizeValues(df)
#input(norm_df.iloc[0])
#____________________________________________________________ Step Four ___________________________________________
#save_box_plot(enc_df, norm_df)

#____________________________________________________________ Step Five ___________________________________________
#plot_corr(enc_df,size=10)

#____________________________________________________________ Step Six ___________________________________________

#file_name = '02_preprocessing_after_normalizing_values.xlsx'
#df = pd.read_excel(file_name, index = True)

#plot_corr(df, 5)

#plot_2d_features(df)


#number_of_optimal_k_means_classes(df)
#k_means(df)

#er = evaluate_pca_effect(df)

#____________________________________________________________ Step Seven ___________________________________________
#draw_pair_wise_scatter_plots(enc_df)

#____________________________________________________________ Step Eight ___________________________________________



#pca_df = pd.DataFrame(df)#PCA(df, 2))





#pca_df.plot(style=['o','rx'])

#plt.show()

#scatter_matrix(pca_df, alpha=0.2)










