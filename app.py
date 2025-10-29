import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
import streamlit as st
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objects as go

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
        feature_groups[pollutant] = [col for col in df.columns if (pollutant in col or col == 'emission')]
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

visualize_boxplots(df, feature_groups, "before_processing_boxplots")

    
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
visualize_boxplots(df, feature_groups, "after_processing_boxplots") 


"""
visualize correlation, feature importance and mutual information 
"""
def visualize_additional_insights(df, feature_groups):
    st.title("Additional Insights")
    group = st.selectbox("select a group for insights", feature_groups.keys(), key="additional_insights")
    cols = feature_groups[group]
    
    # Correlation Heatmap
    st.subheader(f"Correlation Heatmap for {group}")
    corr = df[cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Feature Importance using Z-score
    st.subheader(f"Feature Importance (Z-score) for {group}")
    z_scores = np.abs(stats.zscore(df[cols]))
    feature_importance = pd.Series(z_scores.mean(axis=0), index=cols).sort_values(ascending=False)
    fig, ax = plt.subplots()
    feature_importance.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # mutual information 
    st.subheader(f"Mutual Information for {group}")
    X = df[cols].drop(columns=['emission'])
    y = df['emission']
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    mi_series.plot(kind='bar', ax=ax)
    st.pyplot(fig)

visualize_additional_insights(df, feature_groups)


def visualize_additional_insights_with_plotly(df, feature_groups):
    st.title("Additional Insights (Plotly Interactive)")

    group = st.selectbox("select a group for insights", feature_groups.keys(), key="additional_insights_plotly")
    cols = feature_groups[group]

    # ------------------- 1. Correlation Heatmap -------------------
    st.subheader(f"📌 Correlation Heatmap ({group})")
    corr = df[cols].corr()

    fig_corr = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale="RdBu",
        origin="lower",
        title=f"Correlation Heatmap: {group}"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------- 2. Feature Importance (Z-score) -------------------
    st.subheader(f"📌 Feature Importance (Z-score) ({group})")
    z_scores = abs(stats.zscore(df[cols]))
    feature_importance = pd.Series(z_scores.mean(axis=0), index=cols).sort_values(ascending=False)

    fig_zscore = px.bar(
        feature_importance,
        x=feature_importance.index,
        y=feature_importance.values,
        labels={"x": "Feature", "y": "Z-score"},
        title="Z-score Based Feature Importance"
    )
    st.plotly_chart(fig_zscore, use_container_width=True)

    # ------------------- 3. Mutual Information -------------------
    st.subheader(f"📌 Mutual Information ({group})")
    X = df[cols].drop(columns=['emission'], errors="ignore")
    y = df['emission']

    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    fig_mi = px.bar(
        mi_series,
        x=mi_series.index,
        y=mi_series.values,
        labels={"x": "Feature", "y": "Mutual Information"},
        title="Mutual Information Scores"
    )
    st.plotly_chart(fig_mi, use_container_width=True)


visualize_additional_insights_with_plotly(df, feature_groups)