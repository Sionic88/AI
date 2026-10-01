"""
Profilering av værdata for site "Robin" i Building Data Genome Project 2.

Skriver ut hvilke variabler som finnes, dekningsgrad, statistikk og
korrelasjon mot elektrisitetsforbruk. Lager to figurer.

Bruk:
    python vaer_profil.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "building-data-genome-project-2-official" / "data")
SITE = "Robin"
OUT_DIR = Path("figures").resolve()
OUT_DIR.mkdir(exist_ok=True)


def lagre(fig, navn):
    """Lagrer figuren via en apen filhandle. Unngar Errno 22 pa Windows."""
    sti = OUT_DIR / navn
    try:
        with open(sti, "wb") as f:
            fig.savefig(f, format="png")
        print(f"  lagret {navn}")
    except Exception as e:
        print(f"  KUNNE IKKE lagre {navn}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def last_vaer():
    w = pd.read_csv(DATA_DIR / "weather" / "weather.csv",
                    parse_dates=["timestamp"])
    w = w[w["site_id"] == SITE].copy()
    w = w.set_index("timestamp").sort_index()
    return w.drop(columns=["site_id"])


def spearman(a, b):
    """Spearman uten scipy: Pearson på rangeringene."""
    return a.rank().corr(b.rank())


def last_elektrisitet():
    e = pd.read_csv(DATA_DIR / "meters" / "raw" / "electricity.csv",
                    parse_dates=["timestamp"], index_col="timestamp")
    cols = [c for c in e.columns if c.startswith(SITE + "_")]
    return e[cols]


def main():
    w = last_vaer()
    print(f"{SITE}: {len(w):,} timesrader, "
          f"{w.index.min()} til {w.index.max()}\n")

    profil = pd.DataFrame({
        "dekning_%": 100 * w.notna().mean(),
        "min": w.min(),
        "median": w.median(),
        "max": w.max(),
        "mean": w.mean(),
        "std": w.std(),
        "unike": w.nunique(),
    })
    print("Værvariabler:")
    print(profil.to_string(float_format=lambda x: f"{x:,.2f}"))
    profil.to_csv(OUT_DIR / "vaer_profil.csv")

    print()
    korrelasjon(w)

    # Boksplott per variabel, standardisert slik at alt får plass i én figur.
    # Kolonner uten variasjon kan ikke standardiseres og vises ikke.
    brukbare = [c for c in w.columns
                if w[c].notna().sum() > 100 and w[c].std() > 0]
    z = (w[brukbare] - w[brukbare].mean()) / w[brukbare].std()
    fig, ax = plt.subplots(figsize=(1.3 * len(brukbare) + 2, 4.5))
    ax.boxplot([z[c].dropna().to_numpy() for c in brukbare],
               tick_labels=brukbare,
               flierprops=dict(marker=".", markersize=2, alpha=0.25),
               medianprops=dict(color="crimson"))
    ax.set_ylabel("Standardisert verdi (z)")
    ax.set_title(f"{SITE}: værvariabler, standardisert")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    lagre(fig, "10_vaer_boksplott.png")

    # Lufttemperatur per måned
    t = w["airTemperature"].dropna()
    mnd = t.index.to_period("M").astype(str)
    order = sorted(set(mnd))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot([t[mnd == m].to_numpy() for m in order], tick_labels=order,
               flierprops=dict(marker=".", markersize=2, alpha=0.25),
               medianprops=dict(color="crimson"))
    ax.set_ylabel("Lufttemperatur (°C)")
    ax.set_title(f"{SITE}: lufttemperatur per måned")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    lagre(fig, "11_vaer_temperatur_maaned.png")

    print(f"\nSkrevet til {OUT_DIR}")


def korrelasjon(w):
    e = last_elektrisitet()
    felles = w.index.intersection(e.index)
    temp = w.loc[felles, "airTemperature"]
    rho = {}
    for c in e.columns:
        x = e.loc[felles, c]
        m = x.notna() & temp.notna()
        if m.sum() > 1000:
            rho[c] = spearman(x[m], temp[m])
    rho = pd.Series(rho).sort_values()
    print("\nSpearman-korrelasjon mellom forbruk og lufttemperatur:")
    print(f"  antall bygg: {len(rho)}")
    print(f"  median:      {rho.median(): .3f}")
    print(f"  laveste:     {rho.min(): .3f}  ({rho.idxmin()})")
    print(f"  høyeste:     {rho.max(): .3f}  ({rho.idxmax()})")
    rho.to_csv(OUT_DIR / "vaer_korrelasjon.csv", header=["spearman_rho"])


if __name__ == "__main__":
    main()
