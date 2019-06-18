import os
os.chdir("dataSet")
import pandas

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
    return df



import pandas as pd


file_name = 'preprocessed.xlsx'
df = pd.read_excel(file_name, index_col=0, index=False)
df = labelEncoder(df)

print(df.shape) 
