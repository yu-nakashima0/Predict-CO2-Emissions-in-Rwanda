import pandas as pd


#read data
df = pd.read_csv('./train.csv')
#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.columns)


#check for missing values
#print(df.isnull().sum())
"""
check, whether there are missing values in each column
return : list of columns with missing values
"""
def check_missing_percentage(dataframe):
    list_of_missing = []
    for i in dataframe.columns:
        if dataframe[i].isnull().sum() > 0:
            #print(f"{i}: {dataframe[i].isnull().sum()}")
            list_of_missing.append(i)
    return list_of_missing

list_of_MS = check_missing_percentage(df)



"""
fill missing values in a column with the mode of that column
retrun : dataframe with missing values filled
"""
def fill_missing_with_interpolate(df, columns):
    for col in columns:
        df[col] = df[col].interpolate(method='linear')
    return df

df = fill_missing_with_interpolate(df, list_of_MS)

print(df.isnull().sum())

#check for duplicates
#print(df.duplicated().sum())
