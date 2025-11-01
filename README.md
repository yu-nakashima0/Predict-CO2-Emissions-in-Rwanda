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
- TensorFlow<br>
- keras_tuner<br>
- optuna<br>

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


## 🌱 Understanding the dataset:
-**Sulphur Dioxide(SO₂):**<br>
SO₂ mainly comes from fossil fuel combustion (coal, oil), volcanic eruptions, and industrial processes like metal smelting.<br>
It contributes to acid rain and respiratory problems.<br>
- Main sources:<br>
Coal- and oil-fired power plants<br>
Metal smelters and refineries<br>
Volcanic emissions<br>
Ship exhaust (bunker fuel)<br>

-**Nitrogen Dioxide (NO₂):**<br>
NO₂ is primarily produced from high-temperature combustion, such as in vehicle engines, power plants, and industrial boilers.<br>
It contributes to urban smog and reacts with sunlight to form ground-level ozone.<br>
- Main sources:<br>
    - Vehicle exhaust<br>
    - Power generation (especially fossil fuels)<br>
    - Industrial boilers and furnaces<br>
    - Aircraft engines<br>

-**Carbon Monoxide (CO):**<br>
CO is produced by incomplete combustion — when carbon-based fuels burn with insufficient oxygen.<br>
It’s a colorless, odorless toxic gas mainly from traffic and fires.<br>
- Main sources:<br>
    - Vehicle exhaust (especially older engines)<br>
    - Domestic heating and cooking (stoves, fireplaces)<br>
    - Biomass burning and wildfires<br>
    - Industrial combustion<br>

-**Formaldehyde (HCHO):**<br>
Formaldehyde (HCHO) is both directly emitted and formed through photochemical oxidation of volatile organic compounds (VOCs).<br>
It’s a key intermediate in ozone and smog formation.<br>
- Main sources:<br>
    - Vehicle and industrial VOC emissions<br>
    - Biomass burning and forest fires<br>
    - Chemical manufacturing (resins, paints, glues)<br>
    - Atmospheric oxidation of VOCs<br>

-**Ozone (O₃):**<br>
There are two types of ozone:<br>
Stratospheric ozone (“good ozone”) protects from UV radiation.<br>
Tropospheric ozone (“bad ozone”) is formed when NO₂ and VOCs react under sunlight, causing smog and respiratory problems.<br>
- Main sources:<br>
    - Secondary formation from NO₂ and VOCs<br>
    - Sunlight-driven reactions in polluted cities<br>

-**Aerosols & UvAerosolIndex:**<br>
Aerosols are tiny particles suspended in the air, coming from both natural and anthropogenic sources.<br>
They affect climate (by scattering/absorbing sunlight) and air quality.<br>
- Main sources:<br>
    - Dust storms and desert emissions<br>
    - Volcanic ash<br>
    - Biomass burning and wildfires<br>
    - Industrial and vehicle emissions<br>

-**Cloud:**<br>
Cloud data are not pollutants but influence satellite observations through radiative transfer (reflection, absorption).<br>

<br>

## ✏️ methods: 
1. Data Collection<br>

2. Data Exploration/Processing<br>
-> missing values : filled with interpolate(linear)<br>
-> outlier detection : IsolationForest<br>
-> data normalization<br>
-> handling skewed distribution<br>

3. Data Visualization(streamlit/plotly)<br>
- Boxplot <br>
- Correlation <br>
- Feature importance<br>
- mutual information<br>
- interactive<br>

4. Feature Engineering<br>
- Feature Creation<br>
- Feature Elimination<br>

5. Modeling <br>
- neural network<br>
- k fold cross validation<br>
- optimizer<br>
- early stopping<br>
- hyperparameter optimization<br>
_in progress_<br>
<br>

<br>


## 🎉 run the programm: 
python -m streamlit run app.py