import pandas as pd


#region RAW DATA EXPLANATION
# This section loads the original metadata and weather datasets.
#
# The variables ending in "_raw" represent the original source data.
# These should not be modified directly. Cleaned working copies are
# created later in the script.
#endregion

metadata_raw = pd.read_csv(
    "building-data-genome-project-2-official/data/metadata/metadata.csv"
)

weather_raw = pd.read_csv(
    "building-data-genome-project-2-official/data/weather/weather.csv"
)


#region COLUMN SELECTION EXPLANATION
# Only relevant columns are selected from the original datasets.
#
# Metadata contains information about each individual building.
# Weather contains time-series measurements for each site.
#
# Some additional weather variables are kept so they remain available
# for plotting and further analysis, even though Task 1 mainly requires
# the average outdoor air temperature.
#endregion

metadata_columns = [
    "building_id",             # Unique ID for each building
    "site_id",                 # Site/location ID
    "sqm",                     # Building area in square meters
    "timezone",                # Local timezone
    "yearbuilt",               # Construction year
    "numberoffloors",          # Number of floors
    "occupants",               # Number of occupants
    "rating",                  # Building rating/certification
    "primaryspaceusage",       # Main building use
    "sub_primaryspaceusage"    # More specific building use
]

weather_columns = [
    "timestamp",               # Time of measurement
    "site_id",                 # Site/location ID
    "airTemperature",          # Outdoor temperature
    "dewTemperature",          # Dew point temperature
    "cloudCoverage",           # Cloud coverage
    "seaLvlPressure",          # Atmospheric pressure
    "windDirection",           # Wind direction
    "windSpeed"                # Wind speed
]

metadata = metadata_raw[metadata_columns].copy()
weather = weather_raw[weather_columns].copy()


#region DATA CLEANING EXPLANATION
# The metadata and weather datasets are prepared for further analysis.
#
# building_id is used as the index because it uniquely identifies
# each building.
#
# Buildings without floor area (sqm) are removed because floor area
# is required later when calculating energy use per square meter.
#
# Weather timestamps are converted to datetime so they can be used
# for time-series analysis, filtering and plotting.
#endregion

metadata = metadata.set_index("building_id")
metadata = metadata.dropna(subset=["sqm"])

weather["timestamp"] = pd.to_datetime(weather["timestamp"])


#region SITE WEATHER EXPLANATION
# The weather dataset contains many hourly observations for each site.
#
# For Task 1, the average outdoor air temperature is calculated for
# every site. This creates a smaller dataset with one row per site.
#
# The original weather DataFrame is kept unchanged so it can still
# be used for time-series plots and other analyses.
#endregion

weather_site = (
    weather.groupby("site_id")[["airTemperature"]]
    .mean()
    .rename(columns={"airTemperature": "mean_air_temperature"})
)


#region DATA MERGING EXPLANATION
# Each building belongs to a site through the site_id column.
#
# The site-level average air temperature is therefore joined with
# the building metadata using site_id.
#
# metadata_with_weather becomes the prepared building-level dataset
# containing both building information and average outdoor temperature.
#endregion

metadata_with_weather = metadata.join(
    weather_site,
    on="site_id"
)


#region HOW TO IMPORT THE PREPARED DATA
#
# Example:
#
#     from Pandas_data import metadata_with_weather
#
# The user can then access the prepared building dataset directly:
#
#     print(metadata_with_weather.head())
#
# Several datasets can also be imported at the same time:
#
#     from Pandas_data import metadata, weather, weather_site
#
# Available prepared datasets:
#
# metadata
#     Cleaned building metadata.
#
# weather
#     Cleaned hourly weather data. Useful for time-series analysis
#     and plotting.
#
# weather_site
#     Average outdoor air temperature for each site.
#
# metadata_with_weather
#     Building metadata combined with average site temperature.
#     This is currently the main prepared dataset for Task 1.
#
# Important:
# Importing Pandas_data.py will run the data preparation in this file,
# so the original BDG2 data files must be available at the paths used
# above.
#endregion