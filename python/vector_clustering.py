
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
