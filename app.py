import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
import streamlit as st
from sklearn.preprocessing import PowerTransformer


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
fill missing values in a column with the interpolate of that column
retrun : dataframe with missing values filled
"""
def fill_missing_with_interpolate(df, columns):
    for col in columns:
        df[col] = df[col].interpolate(method='linear')
    return df

df = fill_missing_with_interpolate(df, list_of_MS)

#print(df.isnull().sum())

#check for duplicates
#print(df.duplicated().sum())


"""
group features by pollutant type
return : dictionary with feature groups
"""
def group_features_by_pollutant(df):
    feature_groups = {}
    pollutants = ["SulphurDioxide", "CarbonMonoxide", "NitrogenDioxide", "Formaldehyde", "Ozone", "UvAerosol", "Cloud"]
    for pollutant in pollutants:
        feature_groups[pollutant] = [col for col in df.columns if pollutant in col]
    return feature_groups

feature_groups = group_features_by_pollutant(df)


"""
detect outliers in a column 
return : dataframe with outliers marked
"""
def detect_outliers(df, feature_groups):
    for group_name, cols in feature_groups.items():
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        X = df[cols]
        df[f"{group_name}_outlier"] = iso.fit_predict(X)
    return df

df = detect_outliers(df, feature_groups)
feature_groups = group_features_by_pollutant(df)

"""
visualize boxplots for each feature group
"""
def visualize_boxplots(df, feature_groups, unique_key):
    st.title("Boxplots by feature group")
    group = st.selectbox("select a group", feature_groups.keys(), key=unique_key)
    cols = feature_groups[group]
    st.subheader(f"Boxplots for groups: {group}")
    for c in cols:
        fig, ax = plt.subplots()
        ax.boxplot(df[c].dropna())
        ax.set_title(c)
        st.pyplot(fig)

visualize_boxplots(df, feature_groups, key="before_processing_boxplots")

    
"""
Handling Skewed Distribution
return: dataframe with transformed data
"""
def handle_skewed_distribution(df, feature_groups):
    for group_name, cols in feature_groups.items():
        for col in cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(df[[col]])
    return df

#df = handle_skewed_distribution(df, feature_groups)

"""
normalize data
return: dataframe with normalized data
"""
def normalize_data(df, feature_groups):
    for group_name, cols in feature_groups.items():
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

df = normalize_data(df, feature_groups)


#check data after data processing
visualize_boxplots(df, feature_groups, key="after_processing_boxplots") 

