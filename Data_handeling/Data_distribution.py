import Pandas_data as pdd
import pandas as pd


def site_statistics(site_name, show=True):

    site_metadata = pdd.metadata[pdd.metadata["site_id"] == site_name]
    site_weather = pdd.weather[pdd.weather["site_id"] == site_name]

    site_buildings = site_metadata.index


    primary_use = site_metadata["primaryspaceusage"].value_counts()
    sub_use = site_metadata["sub_primaryspaceusage"].value_counts()


    numeric_metadata = site_metadata.select_dtypes(include="number")

    metadata_stats = numeric_metadata.agg(
        ["mean", "median", "min", "max", "std"]
    )

    metadata_stats.loc["Q1"] = numeric_metadata.quantile(0.25)
    metadata_stats.loc["Q3"] = numeric_metadata.quantile(0.75)


    numeric_weather = site_weather.select_dtypes(include="number")

    weather_stats = numeric_weather.agg(
        ["mean", "median", "min", "max", "std"]
    )

    weather_stats.loc["Q1"] = numeric_weather.quantile(0.25)
    weather_stats.loc["Q3"] = numeric_weather.quantile(0.75)


    meter_types = {
        "Electricity": pdd.electricity,
        "Chilled water": pdd.chilledwater,
        "Gas": pdd.gas,
        "Hot water": pdd.hotwater,
        "Irrigation": pdd.irrigation,
        "Solar": pdd.solar,
        "Steam": pdd.steam,
        "Water": pdd.water
    }

    meter_counts = {}

    for meter_name, meter_data in meter_types.items():

        buildings = [
            building
            for building in site_buildings
            if building in meter_data.columns
        ]

        meter_counts[meter_name] = len(buildings)


    missing_summary = pd.DataFrame({
        "missing": site_metadata.isna().sum(),
        "valid": site_metadata.notna().sum(),
        "valid_percent": (
            site_metadata.notna().mean() * 100
        ).round(1)
    })


    if show:

        print("METADATA:")
        print(metadata_stats)

        print("\nPRIMARY SPACE USAGE:")
        print(primary_use)

        print("\nSUB PRIMARY SPACE USAGE:")
        print(sub_use)

        print("\nWEATHER:")
        print(weather_stats)

        print("\nMETER TYPES USED:")
        print(meter_counts)

        print("\nMISSING VALUES:")
        print(site_metadata.isna().sum())

        print("\nDATA SUMMARY:")
        print(missing_summary)

    return {
        "metadata": site_metadata,
        "weather": site_weather,
        "metadata_stats": metadata_stats,
        "weather_stats": weather_stats,
        "primary_use": primary_use,
        "sub_use": sub_use,
        "meter_counts": meter_counts,
        "missing_summary": missing_summary
    }


if __name__ == "__main__":
    site_statistics("Rat", show=True)