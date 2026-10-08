import Data_distribution as dd
import Pandas_data as pdd
import matplotlib.pyplot as plt
import pandas as pd


# Større tekst på alle grafer
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# METADATA BUILDING AMOUNT
buildings_per_site = pdd.metadata["site_id"].value_counts()

plt.figure(figsize=(12, 6))

buildings_per_site.plot(kind="bar")

plt.xlabel("Site")
plt.ylabel("Number of buildings")
plt.title("Number of buildings per site")

plt.xticks(rotation=45)
plt.tight_layout()



# METADATA AVAILABILITY
metadata_summary = pd.DataFrame({
    "valid_percent": (
        pdd.metadata.notna().mean() * 100
    ).round(1),

    "missing_percent": (
        pdd.metadata.isna().mean() * 100
    ).round(1)
})


metadata_summary[
    ["valid_percent", "missing_percent"]
].plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Metadata column")
plt.ylabel("Percent [%]")
plt.title("Metadata availability - all sites")

plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()



# WEATHER AVAILABILITY
weather_data = pdd.weather.drop(
    columns=["timestamp", "site_id"],
    errors="ignore"
)

weather_summary = pd.DataFrame({
    "valid_percent": (
        weather_data.notna().mean() * 100
    ).round(1),

    "missing_percent": (
        weather_data.isna().mean() * 100
    ).round(1)
})


weather_summary[
    ["valid_percent", "missing_percent"]
].plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Weather variable")
plt.ylabel("Percent [%]")
plt.title("Weather data availability - all sites")

plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()


plt.show()