"""
Vindrose for en valgfri site i BDG2.

Lager polare stablede søylediagram som viser hvor vinden kommer fra
(16 sektorer) og hvor sterk den er (fargelagte fartsintervaller).

Bruk:
    python vindrose.py           # spør "Skriv inn site" ved oppstart
    python vindrose.py Panther   # site direkte som argument

Figurer havner i .\\figures\\<site>\\:
    12_vindrose_hele_perioden.png
    13_vindrose_per_aar.png
    14_vindrose_per_sesong.png
Tabell: figures\\<site>\\vindrose_tabell.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import weather

SITE = "Robin"                     # standard når du bare trykker Enter
OUT_ROOT = Path("figures").resolve()
OUT_DIR = OUT_ROOT / SITE          # settes på nytt når site er valgt

N_SEKTORER = 16                               # 16 x 22.5 grader
FART_KANTER = [0, 2, 4, 6, 8, 10, np.inf]     # m/s
FART_FARGER = ["#d9e8f5", "#9dc3e0", "#5d9ec7",
               "#2f76a8", "#1b4f79", "#0d2b45"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def lagre(fig, navn):
    sti = OUT_DIR / navn
    try:
        with open(sti, "wb") as f:
            fig.savefig(f, format="png")
        print(f"  lagret {navn}")
    except Exception as e:
        print(f"  KUNNE IKKE lagre {navn}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)


def last_vaer():
    """Vinddata for alle sites i datasettet."""
    return weather[["timestamp", "site_id", "windDirection", "windSpeed"]]


def velg_site(sites):
    """Spør etter site til et gyldig navn er skrevet inn.

    Godtar navnet (uavhengig av store og små bokstaver), nummeret i listen,
    eller tom linje for standard. Navnet kan også gis som argument.
    """
    oppslag = {s.lower(): s for s in sites}
    if len(sys.argv) > 1:
        svar = sys.argv[1].strip()
        if svar.lower() in oppslag:
            return oppslag[svar.lower()]
        print(f"Ukjent site '{svar}'.")

    print("Tilgjengelige sites:")
    for i, s in enumerate(sites, start=1):
        print(f"  {i:2d}. {s}")
    while True:
        svar = input(f"\nSkriv inn site (navn eller nummer, Enter = {SITE}): ")
        svar = svar.strip()
        if not svar:
            return SITE
        if svar.isdigit() and 1 <= int(svar) <= len(sites):
            return sites[int(svar) - 1]
        if svar.lower() in oppslag:
            return oppslag[svar.lower()]
        print(f"Fant ikke '{svar}'. Prøv igjen.")


def skill_ut_stille(w):
    """Skiller ut stille timer, som ikke har noen retning å tegne.

    ISD koder stille vær som retning 0 sammen med fart 0. Slike timer hører
    ikke hjemme i rosen, men andelen rapporteres. Timer uten måling telles
    ikke med i noen av delene.
    """
    n_start = len(w)
    w = w.dropna(subset=["windDirection", "windSpeed"])
    stille = w["windSpeed"] <= 0.0
    andel_stille = 100 * stille.mean()
    w = w[~stille]

    print(f"  {n_start:,} timer totalt")
    print(f"  {n_start - len(w) :,} uten retning (manglende eller stille)")
    print(f"  stille timer: {andel_stille:.1f} % av målte timer")
    return w, andel_stille


def sektorer():
    """Sentervinkler og etiketter for kompassektorene."""
    bredde = 360 / N_SEKTORER
    sentre = np.arange(0, 360, bredde)
    if N_SEKTORER == 16:
        navn = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    else:
        navn = [f"{int(s)}" for s in sentre]
    return sentre, bredde, navn


def krysstabell(w):
    """Andel timer per sektor og fartsintervall, i prosent."""
    sentre, bredde, navn = sektorer()

    # skift en halv sektor slik at N dekker 348.75 til 11.25 grader
    sektor = np.floor(((w["windDirection"] + bredde / 2) % 360) / bredde)
    sektor = sektor.astype(int)

    fart = pd.cut(w["windSpeed"], bins=FART_KANTER, right=False)

    tab = pd.crosstab(sektor, fart)
    tab = tab.reindex(range(N_SEKTORER), fill_value=0)
    tab.index = navn
    return 100 * tab / len(w)


def tegn_rose(ax, tab, tittel):
    sentre, bredde, _ = sektorer()
    theta = np.deg2rad(sentre)
    bunn = np.zeros(N_SEKTORER)

    for i, kol in enumerate(tab.columns):
        verdier = tab[kol].to_numpy()
        lo = FART_KANTER[i]
        hi = FART_KANTER[i + 1]
        merke = f"{lo:.0f}+ m/s" if np.isinf(hi) else f"{lo:.0f}-{hi:.0f} m/s"
        ax.bar(theta, verdier, width=np.deg2rad(bredde) * 0.92, bottom=bunn,
               color=FART_FARGER[i % len(FART_FARGER)], edgecolor="white",
               linewidth=0.4, label=merke)
        bunn += verdier

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(sentre))
    ax.set_xticklabels(tab.index, fontsize=7)
    ax.set_title(tittel, pad=14)
    ax.grid(alpha=0.3)
    ticks = [t for t in ax.get_yticks() if t > 0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g} %" for t in ticks], fontsize=6)


def figur_enkel(w, andel_stille):
    tab = krysstabell(w)
    tab.to_csv(OUT_DIR / "vindrose_tabell.csv")

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="polar")
    periode = f"{w.index.min().year}-{w.index.max().year}"
    tegn_rose(ax, tab, f"{SITE}: wind rose {periode}\n"
                       f"(direction the wind blows from, "
                       f"{andel_stille:.1f} % calm hours excluded)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=False,
              title="Wind speed")
    fig.tight_layout()
    lagre(fig, "12_vindrose_hele_perioden.png")

    print("\nAndel timer per sektor og fartsintervall (%):")
    print(tab.to_string(float_format=lambda x: f"{x:5.2f}"))
    print(f"\nHyppigste retning: {tab.sum(axis=1).idxmax()} "
          f"({tab.sum(axis=1).max():.1f} % av timene)")
    print(f"Median vindstyrke:  {w['windSpeed'].median():.1f} m/s")
    print(f"Maks vindstyrke:    {w['windSpeed'].max():.1f} m/s")


def figur_panel(w, grupper, filnavn, tittel_mal):
    n = len(grupper)
    fig = plt.figure(figsize=(5.2 * n, 5.8))
    for i, (navn, del_w) in enumerate(grupper, start=1):
        ax = fig.add_subplot(1, n, i, projection="polar")
        if len(del_w) < 50:
            ax.set_title(f"{navn}\n(too little data)")
            continue
        tegn_rose(ax, krysstabell(del_w), tittel_mal.format(navn=navn))
        if i == n:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05),
                      frameon=False, title="Wind speed", fontsize=7)
    fig.tight_layout()
    lagre(fig, filnavn)


def main():
    global SITE, OUT_DIR
    alle = last_vaer()
    SITE = velg_site(sorted(alle["site_id"].dropna().unique()))
    OUT_DIR = OUT_ROOT / SITE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{SITE}: vindrose\n")
    w = (alle[alle["site_id"] == SITE]
         .set_index("timestamp").sort_index()
         [["windDirection", "windSpeed"]])
    w, andel_stille = skill_ut_stille(w)
    if len(w) < 50:
        print(f"\n{SITE} har for lite vinddata til å lage vindrose.")
        return

    figur_enkel(w, andel_stille)

    aar = [(str(a), d) for a, d in w.groupby(w.index.year)]
    figur_panel(w, aar, "13_vindrose_per_aar.png", "{navn}")

    sesong_navn = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}
    s = w.assign(sesong=[sesong_navn[m] for m in w.index.month])
    rekkefolge = ["Winter", "Spring", "Summer", "Autumn"]
    sesonger = [(navn, s[s["sesong"] == navn].drop(columns="sesong"))
                for navn in rekkefolge]
    figur_panel(w, sesonger, "14_vindrose_per_sesong.png", "{navn}")

    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
