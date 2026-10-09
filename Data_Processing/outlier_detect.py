import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "Cleaned_data"

electricity = pd.read_csv(
    DATA_DIR / "electricity_2016_cleaned.csv"
)



#===========================FUNKSJON FOR OUTLIER DETECTION===========================

def detect_outliers(data, method="iqr", threshold=None):

    #Beholder kun numeriske kolonner
    numeric_data = data.select_dtypes(include=[np.number])

    if method == "iqr":

        if threshold is None:
            threshold = 1.5

        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - threshold * IQR
        upper_limit = Q3 + threshold * IQR

    elif method == "zscore":

        if threshold is None:
            threshold = 3.0

        mean = numeric_data.mean()
        std = numeric_data.std()

        lower_limit = mean - threshold * std
        upper_limit = mean + threshold * std

    else:
        raise ValueError("Velg 'iqr' eller 'zscore' som metode.")

    #True betyr mulig outlier, False betyr ikke outlier
    outliers = (
        (numeric_data < lower_limit) |
        (numeric_data > upper_limit)
    )

    return outliers

def clean_outliers(data, window=48, threshold=8.0, max_change=3.0):

    cleaned = data.copy()

    numeric_columns = data.select_dtypes(include="number").columns

    corrected_count = 0
    rejected_buildings = 0

    for building in numeric_columns:

        values = data[building].copy()

        # Lokal median
        local_median = values.rolling(
            window=window,
            center=True,
            min_periods=12
        ).median()

        # Median Absolute Deviation (MAD)
        deviation = (values - local_median).abs()

        local_mad = deviation.rolling(
            window=window,
            center=True,
            min_periods=12
        ).median()

        # Robust estimat av standardavvik
        robust_std = 1.4826 * local_mad

        # Finn ekstreme topper
        outliers = (
            (values > local_median) &
            (values - local_median > threshold * robust_std) &
            (robust_std > 0)
        )

        # Lag midlertidig korrigert dataserie
        corrected = values.copy()
        corrected.loc[outliers] = np.nan

        corrected = corrected.interpolate(
            method="linear",
            limit_direction="both"
        )

        # Kontroller endring i årsforbruk
        original_total = values.sum()
        corrected_total = corrected.sum()

        if original_total <= 0:
            continue

        change_percent = abs(
            corrected_total - original_total
        ) / original_total * 100

        # Godkjenn bare dersom årsforbruket endres maks 3 %
        if change_percent <= max_change:

            cleaned[building] = corrected
            corrected_count += int(outliers.sum())

        else:
            rejected_buildings += 1

    print("Antall korrigerte outliers:", corrected_count)
    print("Bygninger der korrigering ble avvist:", rejected_buildings)

    return cleaned

#========================MATPLOTLIB FUNKSJON FOR OUTLIER VISUALISERING========================

def plot_outliers(data, building):

    # Hent strømdata og tidsstempler
    time = pd.to_datetime(data["timestamp"])
    values = data[building]

    # Finn outliers med begge metodene
    iqr = detect_outliers(data[[building]], method="iqr")
    zscore = detect_outliers(data[[building]], method="zscore")

    plt.figure(figsize=(15, 6))

    # Originalt strømforbruk
    plt.plot(time, values, color="gray",
             linewidth=0.7, label="Electricity")

    # IQR outliers
    plt.scatter(
        time[iqr[building]],
        values[iqr[building]],
        color="red",
        s=5,
        label="IQR"
    )

    # Z-score outliers
    plt.scatter(
        time[zscore[building]],
        values[zscore[building]],
        color="blue",
        s=10,
        label="Z-score"
    )

    plt.title(f"Outlier Detection - {building}")
    plt.xlabel("Time")
    plt.ylabel("Electricity consumption")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



#======================KJØRER FUNKSJONER==============================================

outliers = detect_outliers(electricity, method="iqr")

print("Total antall mulige outliers:", outliers.sum().sum())

print("\nBygningene med flest outliers:")
print(outliers.sum().sort_values(ascending=False).head(10))

#======================================================================================

outliers_zscore = detect_outliers(electricity, method="zscore")

print("\nTotalt antall outliers med Z-score:",
      outliers_zscore.sum().sum())

print("\nBygningene med flest outliers (Z-score):")
print(outliers_zscore.sum().sort_values(ascending=False).head(10))



#==============MATPLOTLIB CALLING=========================

plot_outliers(electricity, "Fox_public_Belle")

#=========================================================

# Les originale strømdata
RAW_DIR = Path(__file__).resolve().parent.parent / "building-data-genome-project-2-official" / "data" / "meters" / "raw"

raw_data = pd.read_csv(RAW_DIR / "electricity.csv")

building = "Fox_public_Belle"

# Velg 2016 fra rådataene
raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
raw_2016 = raw_data[raw_data["timestamp"].dt.year == 2016]

# Sammenlign originale og behandlede målinger
plt.figure(figsize=(14, 5))

plt.plot(raw_2016["timestamp"], raw_2016[building],
         label="Original data", alpha=0.7)

#plt.plot(pd.to_datetime(electricity["timestamp"]), electricity[building],
         #label="Interpolated data", alpha=0.7)

plt.legend()
plt.grid()
plt.show()

#=================================================================
#kontroller negative strømverdier
negative_values = (electricity.select_dtypes(include="number") < 0)

print("Antall negative strømverdier:",
      negative_values.sum().sum())

#finn mulige outliners med z-score
outliners_zscore = detect_outliers(electricity, method="zscore")

print("Antall mulige outliers (Z-score):",
      outliers_zscore.sum().sum())

#==============LAGRE NY DATASETT UTEN OUTLIERS========================

# Korriger outliers
electricity_cleaned = clean_outliers(electricity)

# Lagre som ny fil
electricity_cleaned.to_csv(
    DATA_DIR / "electricity_2016_final.csv",
    index=False
)

print("Ferdig renset datasett er lagret.")