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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner
import optuna 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

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

#visualize_boxplots(df, feature_groups, "before_processing_boxplots")

    

"""
create new 20 features
return : new dataframe with additional features
"""
def make_emission_features(df):
    features = pd.DataFrame(index=df.index)

    #SO2 
    features['SO2_corrected_column'] = np.nan_to_num(
        df['SulphurDioxide_SO2_slant_column_number_density'] /
        df['SulphurDioxide_SO2_column_number_density_amf']
    )

    features['SO2_15km_ratio'] = np.nan_to_num(
        df['SulphurDioxide_SO2_column_number_density_15km'] /
        (features['SO2_corrected_column'] + 1e-10)
    )

    #CO 
    features['CO_column_density'] = df['CarbonMonoxide_CO_column_number_density']

    features['CO_H2O_ratio'] = np.nan_to_num(
        df['CarbonMonoxide_CO_column_number_density'] /
        (df['CarbonMonoxide_H2O_column_number_density'] + 1e-10)
    )

    #NO2
    features['NO2_tropospheric_fraction'] = np.nan_to_num(
        df['NitrogenDioxide_tropospheric_NO2_column_number_density'] /
        (df['NitrogenDioxide_NO2_column_number_density'] + 1e-10)
    )

    features['NO2_absorbing_aerosol_index'] = df['NitrogenDioxide_absorbing_aerosol_index']

    #HCHO
    features['HCHO_tropospheric_column'] = df['Formaldehyde_tropospheric_HCHO_column_number_density']

    features['HCHO_NO2_ratio'] = np.nan_to_num(
        df['Formaldehyde_tropospheric_HCHO_column_number_density'] /
        (df['NitrogenDioxide_tropospheric_NO2_column_number_density'] + 1e-10)
    )

    #O3
    features['O3_column_density'] = df['Ozone_O3_column_number_density']
    features['O3_effective_temperature'] = df['Ozone_O3_effective_temperature']

    #Aerosol (UV Layer + Index)
    features['Aerosol_optical_depth'] = df['UvAerosolLayerHeight_aerosol_optical_depth']
    features['Aerosol_height'] = df['UvAerosolLayerHeight_aerosol_height']
    features['Aerosol_absorbing_index'] = df['UvAerosolIndex_absorbing_aerosol_index']

    #Cloud
    features['Cloud_fraction'] = df['Cloud_cloud_fraction']
    features['Cloud_top_height'] = df['Cloud_cloud_top_height']
    features['Cloud_optical_depth'] = df['Cloud_cloud_optical_depth']
    features['Surface_albedo'] = df['Cloud_surface_albedo']

    #Observation Geometry
    sensor_zenith_cols = [c for c in df.columns if 'sensor_zenith_angle' in c]
    solar_zenith_cols = [c for c in df.columns if 'solar_zenith_angle' in c]
    features['View_angle_mean'] = df[sensor_zenith_cols].mean(axis=1)
    features['Solar_angle_mean'] = df[solar_zenith_cols].mean(axis=1)

    #Interaction term
    features['Aerosol_cloud_interaction_index'] = (
        df['UvAerosolLayerHeight_aerosol_optical_depth'] * df['Cloud_cloud_fraction']
    )

    features['emission'] = df['emission']

    expected_features = [
        'SO2_corrected_column', 'SO2_15km_ratio', 'CO_column_density', 'CO_H2O_ratio',
        'NO2_tropospheric_fraction', 'NO2_absorbing_aerosol_index',
        'HCHO_tropospheric_column', 'HCHO_NO2_ratio',
        'O3_column_density', 'O3_effective_temperature',
        'Aerosol_optical_depth', 'Aerosol_height', 'Aerosol_absorbing_index',
        'Cloud_fraction', 'Cloud_top_height', 'Cloud_optical_depth', 'Surface_albedo',
        'View_angle_mean', 'Solar_angle_mean', 'Aerosol_cloud_interaction_index','emission'
    ]

    return features[expected_features]

df = make_emission_features(df)


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

#df = normalize_data(df, feature_groups)


def normalize_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

df = normalize_data(df)

#check data after data processing
#visualize_boxplots(df, feature_groups, "after_processing_boxplots") 


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

#visualize_additional_insights(df, feature_groups)


def visualize_additional_insights_with_plotly(df, feature_groups):
    st.title("Additional Insights (Plotly Interactive)")

    group = st.selectbox("select a group for insights", feature_groups.keys(), key="additional_insights_plotly")
    cols = feature_groups[group]

    st.subheader(f"Correlation Heatmap ({group})")
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

    st.subheader(f"Feature Importance (Z-score) ({group})")
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

    st.subheader(f"Mutual Information ({group})")
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


#visualize_additional_insights_with_plotly(df, feature_groups)


"""
split data into train and validation sets
return : X_train, y_train, X_val, y_val
"""
def split_data(df, test_size=0.2):
    y = df['emission'].astype('float32')
    X = df.drop(['emission'], axis=1)
    X = pd.get_dummies(X).astype('float32')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = split_data(df, test_size=0.2)

"""
training neural network model
return: trained model
"""
def neural_network_model(X_train, y_train, X_val, y_val,trial):
    # Hyperparameters from Optuna
    n_units_1 = trial.suggest_int('n_units_1', 32, 256, step=32)
    n_units_2 = trial.suggest_int('n_units_2', 16, 128, step=16)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Build model using the suggested hyperparameters
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  
        Dense(n_units_1, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_units_2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    # Evaluate
    mse, mae = model.evaluate(X_val, y_val, verbose=0)
    rmse = np.sqrt(mse)
    y_pred = model.predict(X_val)
    
    return model, history, mse, mae, rmse, y_pred


#model,history, mse, mae, rmse, y_pred = neural_network_model(X_train, y_train, X_val, y_val)


"""
visualize training history
"""
def visualize_training_history_plotly(history, log_scale=False, start_epoch=0):
    st.title("Neural Network Training History (Plotly)")
    epochs = np.arange(start_epoch, len(history.history['loss'])) + 1

    # Loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs,
        y=history.history['loss'][start_epoch:],
        mode='lines+markers',
        name='Train Loss'
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs,
        y=history.history['val_loss'][start_epoch:],
        mode='lines+markers',
        name='Validation Loss'
    ))
    fig_loss.update_layout(
        title="Model Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )
    if log_scale:
        fig_loss.update_yaxes(type="log")
    st.plotly_chart(fig_loss, use_container_width=True)

    # MAE
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Scatter(
        x=epochs,
        y=history.history['mae'][start_epoch:],
        mode='lines+markers',
        name='Train MAE'
    ))
    fig_mae.add_trace(go.Scatter(
        x=epochs,
        y=history.history['val_mae'][start_epoch:],
        mode='lines+markers',
        name='Validation MAE'
    ))
    fig_mae.update_layout(
        title="Model MAE",
        xaxis_title="Epoch",
        yaxis_title="MAE"
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # RMSE
    train_rmse = np.sqrt(history.history['loss'][start_epoch:])
    val_rmse = np.sqrt(history.history['val_loss'][start_epoch:])
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(
        x=epochs,
        y=train_rmse,
        mode='lines+markers',
        name='Train RMSE'
    ))
    fig_rmse.add_trace(go.Scatter(
        x=epochs,
        y=val_rmse,
        mode='lines+markers',
        name='Validation RMSE'
    ))
    fig_rmse.update_layout(
        title="Model RMSE",
        xaxis_title="Epoch",
        yaxis_title="RMSE"
    )
    if log_scale:
        fig_rmse.update_yaxes(type="log")
    st.plotly_chart(fig_rmse, use_container_width=True)


#visualize_training_history_plotly(history, log_scale=True, start_epoch=5)


"""
k-fold cross validation
return: average mse, mae, rmse across folds
"""
def k_fold_cross_validation(df, k, trial):
    y = df['emission'].astype('float32')
    X = df.drop(['emission'], axis=1)
    X = pd.get_dummies(X).astype('float32')
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_list, mae_list, rmse_list = [], [], []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model, history , mse, mae, rmse, y_pred = neural_network_model(X_train, y_train, X_val, y_val, trial)
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)

    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)

    print(f"Average MSE across {k} folds: {avg_mse:.7f}")
    print(f"Average MAE across {k} folds: {avg_mae:.7f}")
    print(f"Average RMSE across {k} folds: {avg_rmse:.7f}")

    return avg_mse, avg_mae, avg_rmse




def objective(trial):
    avg_mse, avg_mae, avg_rmse = k_fold_cross_validation(
        df, 5, trial
    )
    return avg_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial

print(f"RMSE: {trial.value}")
print("Best hyperparameters:", trial.params)