import pandas as pd

metadata_raw = pd.read_csv(
    "AI/building-data-genome-project-2-official/data/metadata/metadata.csv"
)

weather_raw = pd.read_csv(
    "AI/building-data-genome-project-2-official/data/weather/weather.csv"
)


metadata_columns = [
    "building_id",
    "site_id",
    "sqm",
    "lat",
    "lng",
    "timezone",
    "yearbuilt",
    "numberoffloors",
    "occupants",
    "rating",
    "primaryspaceusage",
    "sub_primaryspaceusage"
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

metadata = metadata.set_index("building_id")
metadata = metadata.dropna(subset=["sqm"])

weather["timestamp"] = pd.to_datetime(weather["timestamp"])


meter_path = "AI/building-data-genome-project-2-official/data/meters/raw/"

electricity  = pd.read_csv(meter_path + "electricity.csv")
chilledwater = pd.read_csv(meter_path + "chilledwater.csv")
gas          = pd.read_csv(meter_path + "gas.csv")
hotwater     = pd.read_csv(meter_path + "hotwater.csv")
irrigation   = pd.read_csv(meter_path + "irrigation.csv")
solar        = pd.read_csv(meter_path + "solar.csv")
steam        = pd.read_csv(meter_path + "steam.csv")
water        = pd.read_csv(meter_path + "water.csv")

electricity["timestamp"]  = pd.to_datetime(electricity["timestamp"])
chilledwater["timestamp"] = pd.to_datetime(chilledwater["timestamp"])
gas["timestamp"]          = pd.to_datetime(gas["timestamp"])
hotwater["timestamp"]     = pd.to_datetime(hotwater["timestamp"])
irrigation["timestamp"]   = pd.to_datetime(irrigation["timestamp"])
solar["timestamp"]        = pd.to_datetime(solar["timestamp"])
steam["timestamp"]        = pd.to_datetime(steam["timestamp"])
water["timestamp"]        = pd.to_datetime(water["timestamp"])

