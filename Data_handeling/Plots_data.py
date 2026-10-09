import Data_distribution as dd  # Beholdt fra ditt opprinnelige skript
import Pandas_data as pdd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path

# Filene lagres i samme mappe som rapportens andre figurer.
OUTPUT_DIR = Path("Images/Visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SITES = ["Rat", "Robin", "Bull", "Hog"]

# Større tekst på alle grafer (samme innstillinger som før)
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")


# -----------------------------------------------------------------------------
# 1. METADATA BUILDING AMOUNT (din opprinnelige graf)
# -----------------------------------------------------------------------------
buildings_per_site = pdd.metadata["site_id"].value_counts()

plt.figure(figsize=(12, 6))
buildings_per_site.plot(kind="bar")

plt.xlabel("Site")
plt.ylabel("Number of buildings")
plt.title("Number of buildings per site")
plt.xticks(rotation=45, ha="right")
save_plot("01_buildings_per_site.png")


# -----------------------------------------------------------------------------
# 2. METADATA AVAILABILITY (din opprinnelige graf)
# -----------------------------------------------------------------------------
metadata_summary = pd.DataFrame({
    "valid_percent": (pdd.metadata.notna().mean() * 100).round(1),
    "missing_percent": (pdd.metadata.isna().mean() * 100).round(1)
})

metadata_summary[["valid_percent", "missing_percent"]].plot(
    kind="bar", stacked=True, figsize=(10, 6)
)

plt.xlabel("Metadata column")
plt.ylabel("Percent [%]")
plt.title("Metadata availability - all sites")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")
save_plot("02_metadata_availability.png")


# -----------------------------------------------------------------------------
# 3. WEATHER AVAILABILITY (din opprinnelige graf)
# -----------------------------------------------------------------------------
weather_data = pdd.weather.drop(
    columns=["timestamp", "site_id"], errors="ignore"
)

weather_summary = pd.DataFrame({
    "valid_percent": (weather_data.notna().mean() * 100).round(1),
    "missing_percent": (weather_data.isna().mean() * 100).round(1)
})

weather_summary[["valid_percent", "missing_percent"]].plot(
    kind="bar", stacked=True, figsize=(10, 6)
)

plt.xlabel("Weather variable")
plt.ylabel("Percent [%]")
plt.title("Weather data availability - all sites")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right")
save_plot("03_weather_availability.png")


# -----------------------------------------------------------------------------
# 4. NEW: BUILDING AREA DISTRIBUTION FOR THE FOUR SELECTED SITES
# Boksplott krever de faktiske bygningene, ikke bare oppsummert statistikk.
# -----------------------------------------------------------------------------
selected_metadata = pdd.metadata.loc[
    pdd.metadata["site_id"].isin(SELECTED_SITES)
].copy()

missing_sites = set(SELECTED_SITES) - set(selected_metadata["site_id"].unique())
if missing_sites:
    raise ValueError(f"Disse områdene mangler i metadata: {sorted(missing_sites)}")

area_by_site = [
    pd.to_numeric(
        selected_metadata.loc[
            selected_metadata["site_id"] == site, "sqm"
        ], errors="coerce"
    ).dropna()
    for site in SELECTED_SITES
]

plt.figure(figsize=(12, 6))
plt.boxplot(area_by_site, tick_labels=SELECTED_SITES)
plt.xlabel("Site")
plt.ylabel("Building area [m$^2$]")
plt.title("Building area distribution - selected sites")
save_plot("04_selected_area_boxplot.png")


# -----------------------------------------------------------------------------
# 5. NEW: PERCENTAGE DISTRIBUTION OF BUILDING TYPES
# Resten av bygningstypene samles i "Other" for en ryddig graf.
# -----------------------------------------------------------------------------
usage_counts = pd.crosstab(
    selected_metadata["site_id"],
    selected_metadata["primaryspaceusage"].fillna("Unknown")
).reindex(SELECTED_SITES)

main_categories = [
    "Education",
    "Office",
    "Public services",
    "Entertainment/public assembly",
    "Lodging/residential"
]

usage_main = usage_counts.reindex(
    columns=main_categories, fill_value=0
).copy()
usage_main["Other"] = usage_counts.sum(axis=1) - usage_main.sum(axis=1)

usage_percent = usage_main.div(usage_counts.sum(axis=1), axis=0) * 100
usage_percent.plot(kind="bar", stacked=True, figsize=(12, 6), width=0.7)

plt.xlabel("Site")
plt.ylabel("Share of buildings [%]")
plt.title("Building use distribution - selected sites")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(
    title="Primary space usage",
    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2
)
save_plot("05_selected_usage.png")


# -----------------------------------------------------------------------------
# 6. NEW: AIR TEMPERATURE DISTRIBUTION FOR THE FOUR SITES
# Denne er valgfri i rapporten dersom 3-sidersgrensen blir for trang.
# -----------------------------------------------------------------------------
selected_weather = pdd.weather.loc[
    pdd.weather["site_id"].isin(SELECTED_SITES)
].copy()

temperature_by_site = [
    pd.to_numeric(
        selected_weather.loc[
            selected_weather["site_id"] == site, "airTemperature"
        ], errors="coerce"
    ).dropna()
    for site in SELECTED_SITES
]

if any(values.empty for values in temperature_by_site):
    raise ValueError("Mangler temperaturdata for ett eller flere valgte områder.")

plt.figure(figsize=(12, 6))
plt.boxplot(temperature_by_site, tick_labels=SELECTED_SITES)
plt.xlabel("Site")
plt.ylabel("Air temperature [°C]")
plt.title("Air temperature distribution - selected sites")
save_plot("06_selected_temperature.png")


# -----------------------------------------------------------------------------
# EKSTRA KONTROLL: tall som kan brukes mot tabellene i rapporten
# -----------------------------------------------------------------------------
#print("\nBUILDINGS PER SELECTED SITE:")
#print(selected_metadata["site_id"].value_counts().reindex(SELECTED_SITES))

#print("\nBUILDING AREA STATISTICS:")
#print(
#    selected_metadata.groupby("site_id")["sqm"]
#    .agg(["count", "mean", "median", "std", "min", "max"])
#    .reindex(SELECTED_SITES).round(1)
#)

#print("\nMETADATA COMPLETENESS [%]:")
#fields = ["sqm", "yearbuilt", "numberoffloors", "occupants", "rating"]
#print(
#    selected_metadata.groupby("site_id")[fields]
#    .agg(lambda col: col.notna().mean() * 100)
#    .reindex(SELECTED_SITES).round(1)
#)

#print("\nBUILDING USAGE DISTRIBUTION [%]:")
#print(usage_percent.round(1))

#print("\nAIR TEMPERATURE STATISTICS:")
#print(
#    selected_weather.groupby("site_id")["airTemperature"]
#    .agg(["mean", "std", "min", "max"])
#    .reindex(SELECTED_SITES).round(2)
#)

# -----------------------------------------------------------------------------
# 7. ELECTRICITY: RAW VS CLEANED (SAME BUILDING, ONLY 2016)
# -----------------------------------------------------------------------------
BUILDING_ID = "Bear_education_Liliana"  # Bytt til en annen building_id ved behov
YEAR = 2016  # CSV-filen inneholder 2016, ikke 2026

start = pd.Timestamp(f"{YEAR}-01-01")
end = pd.Timestamp(f"{YEAR + 1}-01-01")

# Bruker de eksisterende DataFrame-variablene fra Pandas_data.py.
# Merk: electricity_clean_2026 er et misvisende variabelnavn for 2016-filen.
raw_df = pdd.electricity
clean_df = pdd.electricity_clean_2026

if BUILDING_ID not in raw_df.columns:
    raise ValueError(f"{BUILDING_ID} finnes ikke i rådataene.")
if BUILDING_ID not in clean_df.columns:
    raise ValueError(f"{BUILDING_ID} finnes ikke i cleaned-dataene.")

# >= 2016-01-01 og < 2017-01-01: ingen timer fra 2017 slipper inn.
raw_2016 = raw_df.loc[
    (raw_df["timestamp"] >= start) & (raw_df["timestamp"] < end),
    ["timestamp", BUILDING_ID]
].sort_values("timestamp")

clean_2016 = clean_df.loc[
    (clean_df["timestamp"] >= start) & (clean_df["timestamp"] < end),
    ["timestamp", BUILDING_ID]
].sort_values("timestamp")

if raw_2016.empty or clean_2016.empty:
    raise ValueError(f"Mangler data for {BUILDING_ID} i {YEAR} i ett av datasettene.")

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, sharey=True)

axes[0].plot(raw_2016["timestamp"], raw_2016[BUILDING_ID],
             label="Raw electricity", linewidth=0.7)
axes[0].set_title(f"Raw electricity – {BUILDING_ID} ({YEAR})")
axes[0].set_ylabel("Electricity [kWh]")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(clean_2016["timestamp"], clean_2016[BUILDING_ID],
             label="Cleaned electricity", linewidth=0.7)
axes[1].set_title(f"Cleaned electricity – {BUILDING_ID} ({YEAR})")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Electricity [kWh]")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].set_xlim(start, end)

save_plot(f"07_electricity_raw_vs_cleaned_{BUILDING_ID}_{YEAR}.png")

print(f"\nRådata i {YEAR}: {len(raw_2016)} rader")
print(f"Vasket data i {YEAR}: {len(clean_2016)} rader")

print(f"\nFigurene er lagret i: {OUTPUT_DIR.resolve()}")
plt.show()