import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "Data_handeling"))

from Pandas_data import electricity, metadata, weather



#======================FUNKSJONER FOR Å ANALYSERE DATA======================================

def data_quality(missing_values, longest_gaps, total_hours):

    #Prosentvis datadekning
    coverage = (1-(missing_values / total_hours)) * 100

    #lengste sammenhengende periode med manglende verdier
    max_gap = longest_gaps

    #Andel timer som ligger i det lengste gapet
    gap_ratio = max_gap / missing_values.replace(0, float("nan"))

    #samle resultater i en DataFrame
    quality = pd.DataFrame({
        "coverage": coverage,
        "max_gap": max_gap,
        "gap_ratio": gap_ratio
    })

    return quality


def test_interpolation(electricity_2016, missing_values, n_tests=100):

    complete_buildings = missing_values[missing_values == 0].index
    rng = np.random.default_rng(42)

    results = []

    for building in complete_buildings:

        original = electricity_2016[building].copy()
        original_total = original.sum()

        if original_total <= 0:
            continue

        for test in range(n_tests):

            test_data = original.copy()

            # Tilfeldig andel manglende data (5–50 %)
            missing_fraction = rng.uniform(0.05, 0.50)

            # Tilfeldig lengde på sammenhengende gap (1–168 timer)
            gap_length = int(rng.integers(1, 169))

            # Tilfeldig plassering av gap
            gap_start = int(
                rng.integers(0, len(test_data) - gap_length + 1)
            )

            test_data.iloc[
                gap_start:gap_start + gap_length
            ] = np.nan

            # Fjern ytterligere tilfeldige målinger
            remaining = np.flatnonzero(test_data.notna().to_numpy())

            target_missing = int(len(test_data) * missing_fraction)
            additional_missing = max(0, target_missing - gap_length)
            additional_missing = min(additional_missing, len(remaining))

            missing_indices = rng.choice(
                remaining,
                size=additional_missing,
                replace=False
            )

            test_data.iloc[missing_indices] = np.nan

            # Interpolasjon
            reconstructed = test_data.interpolate(
                method="linear",
                limit_direction="both"
            )

            estimated_total = reconstructed.sum()

            error_percent = (
                abs(estimated_total - original_total)
                / original_total * 100
            )

            # Faktisk datakvalitet i testen
            actual_coverage = test_data.notna().mean() * 100
            total_missing = test_data.isna().sum()
            gap_ratio = gap_length / total_missing

            results.append({
                "building": building,
                "test": test,
                "coverage": actual_coverage,
                "max_gap": gap_length,
                "gap_ratio": gap_ratio,
                "error_percent": error_percent
            })

    return pd.DataFrame(results)


def filter_data_quality(quality, results, error_limit=3.0):

    quality = quality.copy()

    # Standardverdi dersom datakvaliteten ikke kan vurderes
    quality["estimated_error"] = float("nan")
    quality["status"] = "Usikker"

    for building in quality.index:

        coverage = quality.loc[building, "coverage"]
        max_gap = quality.loc[building, "max_gap"]

        # Ingen målinger
        if coverage == 0:
            quality.loc[building, "status"] = "Ekskludert"
            continue

        # Ingen manglende målinger
        if coverage == 100:
            quality.loc[building, "estimated_error"] = 0.0
            quality.loc[building, "status"] = "Godkjent"
            continue

        #Nesten komplette målinger
        if coverage >= 99 and max_gap <= 9:
            quality.loc[building, "status"] = "Godkjent"
            continue

        # Mangler nødvendige kvalitetsdata
        if pd.isna(max_gap):
            continue

        # Finn simuleringer med lignende datakvalitet
        similar = results[
            (results["coverage"].between(coverage - 5, coverage + 5)) &
            (results["max_gap"].between(max_gap - 24, max_gap + 24))
        ]

        # For lite testgrunnlag
        if len(similar) < 30:
            continue

        # Bruk 95-persentilen som usikkerhetsmål
        estimated_error = similar["error_percent"].quantile(0.95)

        quality.loc[building, "estimated_error"] = estimated_error

        if estimated_error <= error_limit:
            quality.loc[building, "status"] = "Godkjent"
        else:
            quality.loc[building, "status"] = "Ekskludert"
            

    return quality


#==========================FUNKSJON FOR GODKJENTEBYGG/INTERPOLERING AV MÅLINGER========================================

def interpolate_approved(electricity_2016, filtered_quality):

    #Henter ID til godkjente bygg
    approved_buildings = filtered_quality[
        filtered_quality["status"] == "Godkjent"
    ].index

    #kopierer kun godkjente bygg
    cleaned_data = electricity_2016[
        ["timestamp"] + list(approved_buildings)
    ].copy()

    #linær interpolasjon for å fylle inn manglende målinger
    cleaned_data[approved_buildings] = cleaned_data[
        approved_buildings
    ].interpolate(
        method="linear",
        limit_direction="both"
    )

    return cleaned_data

#===========================================================================================

#==========================ANALYSERER MANGEL PÅ DATA========================================


#Velger kun data fra 2016
electricity_2016 = electricity[
    electricity["timestamp"].dt.year == 2016
    ].copy()

#Tar bort bygninger som ikke finnes med areal i metadata
electricity_2016 = electricity_2016[
    ["timestamp"] + [col for col in electricity_2016.columns if col in metadata.index]
]


#teller manglende verdier for hver bygning
missing_values = electricity_2016.drop(
    columns=["timestamp"]
    ).isna().sum()


#Finn lengste sammenhengende periode med manglende verdier for hver bygning
longest_gap = {}
#Behold bygninger med minst en registrert strømmåling
buildings_with_data = missing_values[missing_values < len(electricity_2016)].index

for building in buildings_with_data:
    missing = electricity_2016[building].isna()

    groups = missing.ne(missing.shift()).cumsum()

    gap_lengths = missing.groupby(groups).sum()

    longest_gap[building] = gap_lengths.max()

#Sorterer bygningene etter lengste sammenhengende periode med manglende verdier
longest_gaps = pd.Series(longest_gap).sort_values(ascending=False)

print("Bygningene med lengste sammenhengende periode med manglende verdier:")
print(longest_gaps.head(20))

#Printer de 10 bygningene med flest manglende verdier
#print(missing_values.sort_values(ascending=False).head(10))


#==================================STATISTIKK========================================================
#Printer statestikk på hva som er manglende 
print("Totalt antall bygninger:", len(missing_values))
print("Bygninger uten målinger:", (missing_values == 8784).sum())
print("bygninger med komplette målinger:", (missing_values == 0).sum())
print("Bygninger med delvise målinger:", 
      ((missing_values > 0) & (missing_values < 8784)).sum()
)

#=====================================================================================================

#Beregn prosentvis dekning for hvert bygg
total_hours = len(electricity_2016)

data_coverage = (1-(missing_values / total_hours)) * 100

#sorterer bygninene etter prosentvis dekning
print(data_coverage.describe())

#Teller bygnigner med minst 80 % dekning av data
print("Bygninger som har minst 80 % dekning av data:", (data_coverage >= 80).sum())

quality = data_quality(
    missing_values,
    longest_gaps,
    len(electricity_2016)
)

print("Kvalitet på data for bygningene:", quality[quality["coverage"] > 0].sort_values("coverage").head(20))


#=========================TEST==============================================================

# Velg en bygning med komplette data
building = missing_values[missing_values == 0].index[0]

# Kopier de originale målingene
original = electricity_2016[building].copy()

# Fjern kunstig hver femte måling (20 %)
test_data = original.copy()
test_data.iloc[::5] = float("nan")

# Rekonstruer manglende målinger med lineær interpolasjon
reconstructed = test_data.interpolate(method="linear")

# Beregn avvik i estimert årsforbruk
original_total = original.sum()
estimated_total = reconstructed.sum()

error_percent = abs(
    estimated_total - original_total
) / original_total * 100

print("Bygning:", building)
print("Opprinnelig årsforbruk:", original_total)
print("Estimert årsforbruk:", estimated_total)
print("Avvik (%):", error_percent)


results = test_interpolation(electricity_2016, missing_values)

print("\nResultater fra 100 simuleringer per bygning:")
print(results["error_percent"].describe(percentiles=[0.95]))

print("\nGjennomsnittlig feil:", results["error_percent"].mean())
print("95-persentil:", results["error_percent"].quantile(0.95))
print("Maksimal feil:", results["error_percent"].max())

#=================================filter data quality========================================================

filtered_quality = filter_data_quality(
    quality,
    results,
    error_limit=3.0
)

print("\nResultat fra dynamisk kvalitetsvurdering:")
print(filtered_quality["status"].value_counts())

#================================ANALYSERER USIKRE BYGNINGER========================================================
uncertain = filtered_quality[
    filtered_quality["status"] == "Usikker"
]

print("\nDatadekning for usikre bygninger:")
print(uncertain["coverage"].describe())

print("Over 99 % dekning:",
      (uncertain["coverage"] >= 99).sum())

print("Mellom 95 og 99 %:",
      uncertain["coverage"].between(95, 99, inclusive="left").sum())

print("Under 95 %:",
      (uncertain["coverage"] < 95).sum())


high_coverage = uncertain[uncertain["coverage"] >= 99]

print("\nLengste gap for bygninger med over 99 % dekning:")
print(high_coverage["max_gap"].describe())

print("\nDe 10 største gapene:")
print(high_coverage["max_gap"].nlargest(10))

#==========================================

print("\nGaplengder for usikre bygninger:")
print(uncertain["max_gap"].describe())

print("\nDe 20 største gapene:")
print(uncertain["max_gap"].nlargest(20))

#=======================INTERPOLERING AV GODKJENTE BYGG============================================

cleaned_data = interpolate_approved(
    electricity_2016,
    filtered_quality
)

print("\nAntall bygninger etter interpolering:", 
      len(cleaned_data.columns) - 1)

print("Manglende verdier etter interpolering:", 
      cleaned_data.isna().sum().sum())


#===========================LAGRING AV RENSKET DATA========================================================

OUTPUT_DIR = Path('_file_').resolve().parent.parent / "Cleaned_data"
OUTPUT_DIR.mkdir(exist_ok=True)

cleaned_data.to_csv(
    OUTPUT_DIR / "electricity_2016_cleaned.csv",
    index=False
)

filtered_quality.to_csv(
    OUTPUT_DIR / "data_quality_2016.csv"
)

print("Behandle data og kvalitetsrapport lagret")