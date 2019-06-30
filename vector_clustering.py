import pandas as pd
def readDataSet():
    file_name = 'dataSet/02_preprocessing_after_normalizing_values.xlsx'
    df = pd.read_excel(file_name, index=False)
    dataSet = []
    for i in range(df.shape[0]):
        well = str('well_' + str(int(df.iloc[i]['Id'])) + '_'+ str(int(df.iloc[i]['Level'])))
        wellVector = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).iloc[i].values)
        dataSet.append(Well(well, wellVector))
    columns = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).columns)
    return dataSet, columns



def plot_scatter():
    import matplotlib.pyplot as plt
    df = pd.read_excel('dataSet/02_preprocessing_after_normalizing_values.xlsx')
    c = ['Carbonate' if df['Carbonate/Sandstone'][i] == 0 else 'Sanstone' for i in range(len(df['Id']))]
    import seaborn as sns
    
    #plt.scatter(df['Longitude'], df['Latitude'], color = c, hue=c)
    sns.scatterplot(df['Longitude'], df['Latitude'], hue=c)
    plt.title('Carbonate_sanstone wells based on its location')
    plt.legend()
    plt.savefig('Carbonate_sanstone wells based on its location')
    plt.show()

    
plot_scatter()
