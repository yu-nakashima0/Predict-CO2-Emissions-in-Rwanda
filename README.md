#### used libraries: 
-pandas<br>
-matplotlib<br>
-skit-learn<br>
-scipy<br>
-streamlit<br>
-plotly<br>

#### features:
-**basic information:**
        -ID_LAT_LON_YEAR_WEEK<br>
        -latitude<br>
        -longitude<br>
        -year<br>
        -week_no<br>

-**SulphurDioxide:**
        -SulphurDioxide_SO2_column_number_density<br>
        -SulphurDioxide_SO2_column_number_density_amf<br>
        -SulphurDioxide_SO2_slant_column_number_density<br>
        -SulphurDioxide_cloud_fraction<br>
        -SulphurDioxide_sensor_azimuth_angle<br>
        -SulphurDioxide_sensor_zenith_angle<br>
        -SulphurDioxide_solar_azimuth_angle<br>
        -SulphurDioxide_solar_zenith_angle<br>
        -SulphurDioxide_SO2_column_number_density_15km<br>

-**CarbonMonoxide:**
        -CarbonMonoxide_CO_column_number_density<br>
        -CarbonMonoxide_H2O_column_number_density<br>
        -CarbonMonoxide_cloud_height<br>
        -CarbonMonoxide_sensor_altitude<br>
        -CarbonMonoxide_sensor_azimuth_angle<br>
        -CarbonMonoxide_sensor_zenith_angle<br>
        -CarbonMonoxide_solar_azimuth_angle<br>
        -CarbonMonoxide_solar_zenith_angle<br>

-**NitrogenDioxide:**
        -NitrogenDioxide_NO2_column_number_density<br>
        -NitrogenDioxide_tropospheric_NO2_column_number_density<br>
        -NitrogenDioxide_stratospheric_NO2_column_number_density<br>
        -NitrogenDioxide_NO2_slant_column_number_density<br>
        -NitrogenDioxide_tropopause_pressure<br>
        -NitrogenDioxide_absorbing_aerosol_index<br>
        -NitrogenDioxide_cloud_fraction<br>
        -NitrogenDioxide_sensor_altitude<br>
        -NitrogenDioxide_sensor_azimuth_angle<br>
        -NitrogenDioxide_sensor_zenith_angle<br>
        -NitrogenDioxide_solar_azimuth_angle<br>
        -NitrogenDioxide_solar_zenith_angle<br>

-**Formaldehyde:**
        -Formaldehyde_tropospheric_HCHO_column_number_density<br>
        -Formaldehyde_tropospheric_HCHO_column_number_density_amf<br>
        -Formaldehyde_HCHO_slant_column_number_density<br>
        -Formaldehyde_cloud_fraction<br>
        -Formaldehyde_solar_zenith_angle<br>
        -Formaldehyde_solar_azimuth_angle<br>
        -Formaldehyde_sensor_zenith_angle<br>
        -Formaldehyde_sensor_azimuth_angle<br>

-**UvAerosolIndex:**
        -UvAerosolIndex_absorbing_aerosol_index<br>
        -UvAerosolIndex_sensor_altitude<br>
        -UvAerosolIndex_sensor_azimuth_angle<br>
        -UvAerosolIndex_sensor_zenith_angle<br>
        -UvAerosolIndex_solar_azimuth_angle<br>
        -UvAerosolIndex_solar_zenith_angle<br>

-**Ozone:**
        -Ozone_O3_column_number_density<br>
        -Ozone_O3_column_number_density_amf<br>
        -Ozone_O3_slant_column_number_density<br>
        -Ozone_O3_effective_temperature<br>
        -Ozone_cloud_fraction<br>
        -Ozone_sensor_azimuth_angle<br>
        -Ozone_sensor_zenith_angle<br>
        -Ozone_solar_azimuth_angle<br>
        -Ozone_solar_zenith_angle<br>

-**UvAerosolLayerHeight:**
        -UvAerosolLayerHeight_aerosol_height<br>
        -UvAerosolLayerHeight_aerosol_pressure<br>
        -UvAerosolLayerHeight_aerosol_optical_depth<br>
        -UvAerosolLayerHeight_sensor_zenith_angle<br>
        -UvAerosolLayerHeight_sensor_azimuth_angle<br>
        -UvAerosolLayerHeight_solar_azimuth_angle<br>
        -UvAerosolLayerHeight_solar_zenith_angle<br>

-**Cloud:**
        -Cloud_cloud_fraction<br>
        -Cloud_cloud_top_pressure<br>
        -Cloud_cloud_top_height<br>
        -Cloud_cloud_base_pressure<br>
        -Cloud_cloud_base_height<br>
        -Cloud_cloud_optical_depth<br>
        -Cloud_surface_albedo<br>
        -Cloud_sensor_azimuth_angle<br>
        -Cloud_sensor_zenith_angle<br>
        -Cloud_solar_azimuth_angle<br>
        -Cloud_solar_zenith_angle<br>

-**Target:**
        -emission



#### start Programm: 
python -m streamlit run app.py