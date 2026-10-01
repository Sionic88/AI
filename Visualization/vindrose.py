"""
Vindrose for site "Robin" (University College London) i BDG2.

Lager polare stablede søylediagram som viser hvor vinden kommer fra
(16 sektorer) og hvor sterk den er (fargelagte fartsintervaller).

Bruk:
    python vindrose.py

Figurer havner i .\\figures\\:
    12_vindrose_hele_perioden.png
    13_vindrose_per_aar.png
    14_vindrose_per_sesong.png
Tabell: figures\\vindrose_tabell.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "building-data-genome-project-2-official" / "data")
SITE = "Robin"
OUT_DIR = Path("figures").resolve()
OUT_DIR.mkdir(exist_ok=True)

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
    w = pd.read_csv(DATA_DIR / "weather" / "weather.csv",
                    parse_dates=["timestamp"])
    w = w[w["site_id"] == SITE].copy()
    w = w.set_index("timestamp").sort_index()
    return w[["windDirection", "windSpeed"]]


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
        navn = ["N", "NNØ", "NØ", "ØNØ", "Ø", "ØSØ", "SØ", "SSØ",
                "S", "SSV", "SV", "VSV", "V", "VNV", "NV", "NNV"]
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
    tegn_rose(ax, tab, f"{SITE}: vindrose 2016-2017\n"
                       f"(retningen vinden kommer fra, "
                       f"{andel_stille:.1f} % stille timer utelatt)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=False,
              title="Vindstyrke")
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
            ax.set_title(f"{navn}\n(for få data)")
            continue
        tegn_rose(ax, krysstabell(del_w), tittel_mal.format(navn=navn))
        if i == n:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05),
                      frameon=False, title="Vindstyrke", fontsize=7)
    fig.tight_layout()
    lagre(fig, filnavn)


def main():
    print(f"{SITE}: vindrose\n")
    w = last_vaer()
    w, andel_stille = skill_ut_stille(w)

    figur_enkel(w, andel_stille)

    aar = [(str(a), d) for a, d in w.groupby(w.index.year)]
    figur_panel(w, aar, "13_vindrose_per_aar.png", "{navn}")

    sesong_navn = {12: "Vinter", 1: "Vinter", 2: "Vinter",
                   3: "Vår", 4: "Vår", 5: "Vår",
                   6: "Sommer", 7: "Sommer", 8: "Sommer",
                   9: "Høst", 10: "Høst", 11: "Høst"}
    s = w.assign(sesong=[sesong_navn[m] for m in w.index.month])
    rekkefolge = ["Vinter", "Vår", "Sommer", "Høst"]
    sesonger = [(navn, s[s["sesong"] == navn].drop(columns="sesong"))
                for navn in rekkefolge]
    figur_panel(w, sesonger, "14_vindrose_per_sesong.png", "{navn}")

    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
