## 🗺️ Predicting CO₂ Emissions:
Accurately tracking carbon emissions is a vital foundation for addressing climate change.<br>
<br>
This project aims to build machine learning models that use open-source CO₂ emission data collected by the Sentinel-5P satellite to forecast future carbon output.<br>
<br>



## 📚 used Python libraries: 
- pandas<br>
- matplotlib<br>
- scikit-learn<br>
- scipy<br>
- streamlit<br>
- plotly<br>
- PyTorch<br>
- Keras<br>

<br>

## ✨ features of original dataset:
-**basic information:**<br>
- ID_LAT_LON_YEAR_WEEK<br>
- latitude<br>
- longitude<br>
- year<br>
- week_no<br>

-**SulphurDioxide:**<br>
- SulphurDioxide_SO2_column_number_density<br>
- SulphurDioxide_SO2_column_number_density_amf<br>
- SulphurDioxide_SO2_slant_column_number_density<br>
- SulphurDioxide_cloud_fraction<br>
- SulphurDioxide_sensor_azimuth_angle<br>
- SulphurDioxide_sensor_zenith_angle<br>
- SulphurDioxide_solar_azimuth_angle<br>
- SulphurDioxide_solar_zenith_angle<br>
- SulphurDioxide_SO2_column_number_density_15km<br>

-**CarbonMonoxide:**<br>
- CarbonMonoxide_CO_column_number_density<br>
- CarbonMonoxide_H2O_column_number_density<br>
- CarbonMonoxide_cloud_height<br>
- CarbonMonoxide_sensor_altitude<br>
- CarbonMonoxide_sensor_azimuth_angle<br>
- CarbonMonoxide_sensor_zenith_angle<br>
- CarbonMonoxide_solar_azimuth_angle<br>
- CarbonMonoxide_solar_zenith_angle<br>

-**NitrogenDioxide:**<br>
- NitrogenDioxide_NO2_column_number_density<br>
- NitrogenDioxide_tropospheric_NO2_column_number_density<br>
- NitrogenDioxide_stratospheric_NO2_column_number_density<br>
- NitrogenDioxide_NO2_slant_column_number_density<br>
- NitrogenDioxide_tropopause_pressure<br>
- NitrogenDioxide_absorbing_aerosol_index<br>
- NitrogenDioxide_cloud_fraction<br>
- NitrogenDioxide_sensor_altitude<br>
- NitrogenDioxide_sensor_azimuth_angle<br>
- NitrogenDioxide_sensor_zenith_angle<br>
- NitrogenDioxide_solar_azimuth_angle<br>
- NitrogenDioxide_solar_zenith_angle<br>

-**Formaldehyde:**<br>
- Formaldehyde_tropospheric_HCHO_column_number_density<br>
- Formaldehyde_tropospheric_HCHO_column_number_density_amf<br>
- Formaldehyde_HCHO_slant_column_number_density<br>
- Formaldehyde_cloud_fraction<br>
- Formaldehyde_solar_zenith_angle<br>
- Formaldehyde_solar_azimuth_angle<br>
- Formaldehyde_sensor_zenith_angle<br>
- Formaldehyde_sensor_azimuth_angle<br>

-**UvAerosolIndex:**<br>
- UvAerosolIndex_absorbing_aerosol_index<br>
- UvAerosolIndex_sensor_altitude<br>
- UvAerosolIndex_sensor_azimuth_angle<br>
- UvAerosolIndex_sensor_zenith_angle<br>
- UvAerosolIndex_solar_azimuth_angle<br>
- UvAerosolIndex_solar_zenith_angle<br>

-**Ozone:**<br>
- Ozone_O3_column_number_density<br>
- Ozone_O3_column_number_density_amf<br>
- Ozone_O3_slant_column_number_density<br>
- Ozone_O3_effective_temperature<br>
- Ozone_cloud_fraction<br>
- Ozone_sensor_azimuth_angle<br>
- Ozone_sensor_zenith_angle<br>
- Ozone_solar_azimuth_angle<br>
- Ozone_solar_zenith_angle<br>

-**UvAerosolLayerHeight:**<br>
- UvAerosolLayerHeight_aerosol_height<br>
- UvAerosolLayerHeight_aerosol_pressure<br>
- UvAerosolLayerHeight_aerosol_optical_depth<br>
- UvAerosolLayerHeight_sensor_zenith_angle<br>
- UvAerosolLayerHeight_sensor_azimuth_angle<br>
- UvAerosolLayerHeight_solar_azimuth_angle<br>
- UvAerosolLayerHeight_solar_zenith_angle<br>

-**Cloud:**<br>
- Cloud_cloud_fraction<br>
- Cloud_cloud_top_pressure<br>
- Cloud_cloud_top_height<br>
- Cloud_cloud_base_pressure<br>
- Cloud_cloud_base_height<br>
- Cloud_cloud_optical_depth<br>
- Cloud_surface_albedo<br>
- Cloud_sensor_azimuth_angle<br>
- Cloud_sensor_zenith_angle<br>
- Cloud_solar_azimuth_angle<br>
- Cloud_solar_zenith_angle<br>

-**Target:**<br>
- emission
<br>


## ✏️ methods: 
1. Data Collection<br>

2. Data Exploration/Processing<br>
-> missing values : filled with interpolate(linear)<br>
-> outlier Detection : IsolationForest<br>
-> data normalization<br>
-> Handling Skewed Distribution<br>

3. Data Visualization(streamlit/plotly)<br>
- Boxplot <br>
- Correlation <br>
- Feature importance<br>
- mutual information<br>
- interactive<br>

4. Feature Engineering<br>
- (Feature Creation)<br>
- (Feature Elimination)<br>

5. Modeling <br>
- neural network
- k fold cross validation
- early stopping
in progress<br>
<br>

<br>


## 🎉 run the programm: 
python -m streamlit run app.py