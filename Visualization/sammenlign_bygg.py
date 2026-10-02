"""
Sammenligner fire like undervisningsbygg (College Classroom, ca. 10 000-12 000 m2)
i fire ulike klima, fra Building Data Genome Project 2.

    Ottawa, Canada         kaldt innlandsklima      Crow_education_Omer
    Cardiff, Storbritannia mildt kystklima          Lamb_education_Emery
    Austin, Texas, USA     fuktig subtropisk        Bull_education_Miranda
    Tempe, Arizona, USA    varm ørken               Fox_education_Nilda

Strøm er i kWh på alle sites og sammenlignes på nivå (kWh/m2).
Varme- og kjølemålerne har ulik enhet fra site til site i rådataene, så de
sammenlignes bare på form: hver måned som andel av byggets eget årsforbruk.

Skriptet vasker ikke dataene; det gjøres et annet sted.

Bruk:
    python sammenlign_bygg.py

Figurer havner i .\\figures\\sammenligning\\.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "building-data-genome-project-2-official" / "data")
OUT_DIR = Path("figures").resolve() / "sammenligning"

# Rekkefølge fra kaldest til varmest. Fargen følger bygget i alle figurer.
BYGG = {
    "Crow_education_Omer":    {"sted": "Ottawa",  "klima": "kaldt innland",
                               "varme": "hotwater", "kjol": "chilledwater"},
    "Lamb_education_Emery":   {"sted": "Cardiff", "klima": "mildt kystklima",
                               "varme": "gas",      "kjol": None},
    "Bull_education_Miranda": {"sted": "Austin",  "klima": "fuktig subtropisk",
                               "varme": "steam",    "kjol": "chilledwater"},
    "Fox_education_Nilda":    {"sted": "Tempe",   "klima": "varm ørken",
                               "varme": "hotwater", "kjol": "chilledwater"},
}
FARGER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fast kategorisk rekkefølge
for (b, info), farge in zip(BYGG.items(), FARGER):
    info["farge"] = farge

TEKST = "#1a1a19"
TEKST_SEKUNDAER = "#5f5e5a"
RUTENETT = "#e4e3dc"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9,
    "axes.edgecolor": RUTENETT, "axes.labelcolor": TEKST_SEKUNDAER,
    "axes.titlecolor": TEKST, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RUTENETT, "grid.linewidth": 0.6,
    "xtick.color": TEKST_SEKUNDAER, "ytick.color": TEKST_SEKUNDAER,
    "legend.frameon": False, "lines.linewidth": 2,
})


def lagre(fig, navn):
    """Lagrer figuren via en åpen filhandle. Unngår Errno 22 på Windows."""
    try:
        with open(OUT_DIR / navn, "wb") as f:
            fig.savefig(f, format="png")
        print(f"  lagret {navn}")
    except Exception as e:
        print(f"  KUNNE IKKE lagre {navn}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)


def navn(b):
    return f"{BYGG[b]['sted']} ({b.split('_')[-1]})"


# ----------------------------------------------------------------------
# Innlasting
# ----------------------------------------------------------------------
def last_metadata():
    md = pd.read_csv(DATA_DIR / "metadata" / "metadata.csv")
    return md.set_index("building_id").loc[list(BYGG)]


def last_maler(meter):
    """Kolonnene for de valgte byggene i en målerfil (tom tabell om ingen)."""
    e = pd.read_csv(DATA_DIR / "meters" / "raw" / f"{meter}.csv",
                    parse_dates=["timestamp"], index_col="timestamp")
    return e[[b for b in BYGG if b in e.columns]]


def last_temperatur(meta):
    """Utetemperatur per bygg, hentet fra byggets site."""
    w = pd.read_csv(DATA_DIR / "weather" / "weather.csv",
                    usecols=["timestamp", "site_id", "airTemperature"],
                    parse_dates=["timestamp"])
    t = w.pivot_table(index="timestamp", columns="site_id",
                      values="airTemperature")
    return pd.DataFrame({b: t[meta.loc[b, "site_id"]] for b in BYGG})


# ----------------------------------------------------------------------
# Hjelpere
# ----------------------------------------------------------------------
def maanedlig_sum(df):
    """Månedssum estimert som snitt per time ganger timer i måneden.

    Gir et rettferdig tall også for måneder med hull i måleserien.
    """
    snitt = df.resample("MS").mean()
    timer = snitt.index.days_in_month * 24
    return snitt.mul(timer, axis=0)


def direkte_etikett(ax, x, y, tekst, farge):
    ax.annotate(tekst, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=TEKST)
    ax.plot([x], [y], "o", ms=4, color=farge)


# ----------------------------------------------------------------------
# Figurer
# ----------------------------------------------------------------------
def fig_klima(temp):
    """Månedlig snittemperatur: hvor ulike klimaene faktisk er."""
    mnd = temp.resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for b in BYGG:
        ax.plot(mnd.index, mnd[b], color=BYGG[b]["farge"], label=navn(b))
        siste = mnd[b].last_valid_index()
        direkte_etikett(ax, siste, mnd[b][siste],
                        BYGG[b]["sted"], BYGG[b]["farge"])
    ax.axhline(0, color=TEKST_SEKUNDAER, lw=0.8)
    ax.set_ylabel("Utetemperatur, månedssnitt (°C)")
    ax.set_title("Klima: månedlig snittemperatur 2016-2017")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=8)
    ax.set_xlim(right=mnd.index[-1] + pd.Timedelta(days=75))
    fig.tight_layout()
    lagre(fig, "01_klima.png")


def fig_strom_aar(el, meta):
    """Årlig strømforbruk per m2, ett tall per bygg."""
    aar = maanedlig_sum(el).sum() / 2 / meta["sqm"]
    rekkefolge = list(BYGG)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(rekkefolge))[::-1]
    ax.barh(y, [aar[b] for b in rekkefolge], height=0.55,
            color=[BYGG[b]["farge"] for b in rekkefolge])
    ax.set_yticks(y, [f"{navn(b)}\n{BYGG[b]['klima']}" for b in rekkefolge])
    for yi, b in zip(y, rekkefolge):
        ax.text(aar[b], yi, f"  {aar[b]:.0f}", va="center", fontsize=8,
                color=TEKST)
    ax.set_xlabel("Strøm (kWh/m² per år, snitt 2016-2017)")
    ax.set_title("Årlig strømforbruk per kvadratmeter")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=aar.max() * 1.15)
    fig.tight_layout()
    lagre(fig, "02_strom_per_m2_aar.png")
    return aar


def fig_strom_maaned(el, meta):
    """Strøm per m2 per måned: sesongvariasjon i hvert klima."""
    mnd = maanedlig_sum(el).div(meta["sqm"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for b in BYGG:
        ax.plot(mnd.index, mnd[b], color=BYGG[b]["farge"], label=navn(b))
        siste = mnd[b].last_valid_index()
        direkte_etikett(ax, siste, mnd[b][siste],
                        BYGG[b]["sted"], BYGG[b]["farge"])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Strøm (kWh/m² per måned)")
    ax.set_title("Strøm per kvadratmeter, per måned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=8)
    ax.set_xlim(right=mnd.index[-1] + pd.Timedelta(days=75))
    fig.tight_layout()
    lagre(fig, "03_strom_per_m2_maaned.png")


def fig_strom_mot_temp(el, temp, meta):
    """Døgnforbruk mot døgntemperatur, ett panel per bygg, felles akser."""
    d_el = el.resample("D").mean().mul(24).div(meta["sqm"]) * 1000  # Wh/m2
    d_t = temp.resample("D").mean()
    fig, axes = plt.subplots(1, len(BYGG), figsize=(13, 3.8),
                             sharex=True, sharey=True)
    for ax, b in zip(axes, BYGG):
        m = d_el[b].notna() & d_t[b].notna()
        hverdag = d_el.index.dayofweek < 5
        ax.scatter(d_t.loc[m & hverdag, b], d_el.loc[m & hverdag, b], s=9,
                   color=BYGG[b]["farge"], alpha=0.55, linewidths=0,
                   label="hverdag")
        ax.scatter(d_t.loc[m & ~hverdag, b], d_el.loc[m & ~hverdag, b], s=9,
                   facecolors="none", edgecolors=BYGG[b]["farge"],
                   alpha=0.6, linewidths=0.7, label="helg")
        ax.set_title(f"{navn(b)}\n{BYGG[b]['klima']}", fontsize=9)
        ax.set_xlabel("Utetemperatur, døgnsnitt (°C)")
    axes[0].set_ylabel("Strøm (Wh/m² per døgn)")
    axes[0].set_ylim(bottom=0)
    axes[-1].legend(loc="upper right", fontsize=8, markerscale=1.5)
    fig.suptitle("Strøm mot utetemperatur, ett punkt per døgn",
                 color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "04_strom_mot_temperatur.png")


def fig_doegnprofil(el, meta):
    """Gjennomsnittlig time-for-time-profil, hverdag og helg."""
    per_m2 = el.div(meta["sqm"]) * 1000                                # Wh/m2
    hverdag = per_m2.index.dayofweek < 5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (maske, tittel) in zip(axes, [(hverdag, "Hverdag"),
                                          (~hverdag, "Helg")]):
        prof = per_m2[maske].groupby(per_m2[maske].index.hour).mean()
        for b in BYGG:
            ax.plot(prof.index, prof[b], color=BYGG[b]["farge"],
                    label=navn(b))
        ax.set_title(tittel)
        ax.set_xlabel("Time på døgnet")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Strøm (Wh/m² per time)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Døgnprofil for strøm, snitt over 2016-2017",
                 color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "05_strom_doegnprofil.png")


def fig_termisk_form(varme, kjol):
    """Varme og kjøling som andel av eget årsforbruk per måned.

    Enhetene for termiske målere er ulike mellom sites, så bare formen
    sammenlignes: hvilke måneder bygget bruker varme og kjøling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (data, tittel, nokkel) in zip(axes, [(varme, "Varme", "varme"),
                                                 (kjol, "Kjøling", "kjol")]):
        for b in BYGG:
            if b not in data:
                continue
            mnd = maanedlig_sum(data[[b]])[b]
            prof = mnd.groupby(mnd.index.month).mean()
            andel = 100 * prof / prof.sum()
            ax.plot(andel.index, andel, color=BYGG[b]["farge"],
                    marker="o", ms=4, label=f"{navn(b)}: {BYGG[b][nokkel]}")
        mangler = [BYGG[b]["sted"] for b in BYGG if b not in data]
        if mangler:
            ax.text(0.99, 0.97, "Ingen måler: " + ", ".join(mangler),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=TEKST_SEKUNDAER)
        ax.set_title(tittel)
        ax.set_xticks(range(1, 13),
                      ["J", "F", "M", "A", "M", "J",
                       "J", "A", "S", "O", "N", "D"])
        ax.set_xlabel("Måned")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                  fontsize=7.5)
    axes[0].set_ylabel("Andel av byggets årsforbruk (%)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("Når på året brukes varme og kjøling? "
                 "(form, ikke nivå: enhetene er ulike mellom sites)",
                 color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "06_varme_kjoling_sesong.png")


# ----------------------------------------------------------------------
# Hovedløp
# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = last_metadata()
    print("Sammenligner:")
    for b in BYGG:
        print(f"  {b:24s} {BYGG[b]['sted']:8s} {meta.loc[b, 'sqm']:8,.0f} m2")
    print()

    el = last_maler("electricity")
    temp = last_temperatur(meta)

    maler = {}
    for m in {i[k] for i in BYGG.values() for k in ("varme", "kjol") if i[k]}:
        maler[m] = last_maler(m)
    varme = pd.DataFrame({b: maler[i["varme"]][b] for b, i in BYGG.items()})
    kjol = pd.DataFrame({b: maler[i["kjol"]][b]
                         for b, i in BYGG.items() if i["kjol"]})

    fig_klima(temp)
    aar = fig_strom_aar(el, meta)
    fig_strom_maaned(el, meta)
    fig_strom_mot_temp(el, temp, meta)
    fig_doegnprofil(el, meta)
    fig_termisk_form(varme, kjol)

    tab = pd.DataFrame({
        "sted": [BYGG[b]["sted"] for b in BYGG],
        "klima": [BYGG[b]["klima"] for b in BYGG],
        "sqm": meta["sqm"],
        "strom_kWh_m2_aar": aar.round(0),
        "strom_dekning_%": (100 * el.notna().mean()).round(0),
        "snittemp_C": temp.mean().round(1),
        "varmemaler": [BYGG[b]["varme"] for b in BYGG],
        "kjolemaler": [BYGG[b]["kjol"] or "" for b in BYGG],
    }, index=list(BYGG))
    tab.index.name = "building_id"
    tab.to_csv(OUT_DIR / "sammenligning.csv", sep=";", encoding="utf-8-sig")
    print("\n" + tab.to_string())
    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
