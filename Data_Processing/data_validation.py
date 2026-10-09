import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Filstier
DATA_DIR = Path(__file__).resolve().parent.parent / "Cleaned_data"

# Les data før og etter outlier-korrigering
original = pd.read_csv(DATA_DIR / "electricity_2016_cleaned.csv")
cleaned = pd.read_csv(DATA_DIR / "electricity_2016_final.csv")


def validate_data(original, cleaned):

    # Fjern timestamp fra beregningene
    original_values = original.drop(columns=["timestamp"])
    cleaned_values = cleaned.drop(columns=["timestamp"])

    # Kontroller at datasett har samme struktur
    if not original["timestamp"].equals(cleaned["timestamp"]):
        raise ValueError("Datasett har ulike tidsstempler!")

    if not original_values.columns.equals(cleaned_values.columns):
        raise ValueError("Datasett har ulike bygninger!")

    print("\n--- DATA VALIDATION ---")

    # Antall bygninger
    print("Antall bygninger:", len(cleaned_values.columns))

    # Manglende verdier
    missing = cleaned_values.isna().sum().sum()
    print("Manglende verdier:", missing)

    # Negative verdier
    negative = (cleaned_values < 0).sum().sum()
    print("Negative verdier:", negative)

    # Uendelige verdier
    infinite = np.isinf(cleaned_values.to_numpy()).sum()
    print("Uendelige verdier:", infinite)

    # Beregn årsforbruk før og etter korrigering
    original_total = original_values.sum()
    cleaned_total = cleaned_values.sum()

    # Prosentvis endring i årsforbruk
    change_percent = (
        (cleaned_total - original_total)
        / original_total.replace(0, np.nan)
    ) * 100

    # Samle resultatene
    validation = pd.DataFrame({
        "original_kWh": original_total,
        "cleaned_kWh": cleaned_total,
        "change_percent": change_percent,
        "absolute_change": change_percent.abs()
    })

    # Tell bygninger med betydelige endringer
    print("\n--- ENDRINGER I ÅRSFORBRUK ---")

    print("Gjennomsnittlig absolutt endring (%):",
          validation["absolute_change"].mean())

    print("Største endring (%):",
          validation["absolute_change"].max())

    print("Bygninger med over 1 % endring:",
          (validation["absolute_change"] > 1).sum())

    print("Bygninger med over 3 % endring:",
          (validation["absolute_change"] > 3).sum())

    print("\nDe 10 bygningene med størst endring:")
    print(
        validation.sort_values(
            "absolute_change",
            ascending=False
        ).head(10)
    )

    return validation


#===================MATPLOTLIB VISUALISERING===================
def plot_outlier_correction(original, cleaned, building):

    time = pd.to_datetime(original["timestamp"])

    plt.figure(figsize=(15, 6))

    # Før outlier-korrigering
    plt.plot(
        time,
        original[building],
        label="Before correction",
        color="red",
        alpha=0.7
    )

    # Etter outlier-korrigering
    plt.plot(
        time,
        cleaned[building],
        label="After correction",
        color="blue",
        alpha=0.7
    )

    # Marker punktene som ble endret
    changed = ~np.isclose(
        original[building],
        cleaned[building],
        rtol=1e-9,
        atol=1e-9
    )

    plt.scatter(
        time[changed],
        original.loc[changed, building],
        color="orange",
        s=25,
        label="Corrected outliers",
        zorder=5
    )

    plt.title(f"Outlier correction - {building}")
    plt.xlabel("Time")
    plt.ylabel("Electricity consumption (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#==========================================================================


# Kjør kontrollen
validation = validate_data(original, cleaned)

# Lagre kvalitetsrapport
validation.to_csv(
    DATA_DIR / "validation_2016.csv",
    index_label="building_id"
)

print("\nValideringsrapport lagret.")

#===============MATPLOTLIB VISUALISERING AV OUTLIER-KORRIGERING===================

plot_outlier_correction(
    original,
    cleaned,
    "Bull_education_Jeffery"
)