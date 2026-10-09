"""
Tidsserier, fordelinger og korrelasjonsmatrise for en valgfri site i BDG2.

Dekker de delene av "Data understanding" som de andre skriptene ikke viser
direkte:

  19  Tidsserie       daglig forbruk per målertype over hele 2016-2017, med
                      glidende 30-dagers snitt og utetemperaturen øverst
  20  År mot år       2016 lagt oppå 2017, så trend og avvik mellom årene synes
  21  Bygg over tid   ett bygg per rad, én kolonne per dag: viser hull, døde
                      målere og nivåskift for hvert bygg
  22  Histogram       timesverdier per målertype, lineær og log-akse
  23  Histogram vær   fordelingen av hver værvariabel
  24  Korrelasjon     korrelasjonsmatrise (Spearman, døgnverdier) mellom alle
                      værvariabler og målertyper på siten

Nøkkeltall (n, min, Q1, median, Q3, maks, snitt, std, skjevhet) for alle
målere og værvariabler skrives til sammendrag.csv og til konsollen.

Skriptet vasker ikke dataene; det fjerner bare negative målerverdier. Det er
med vilje, slik at feil i dataene blir synlige i figurene.

Bruk:
    python tidsserie_fordeling.py           # spør "Skriv inn site" ved oppstart
    python tidsserie_fordeling.py Panther   # site direkte som argument

Figurer og tabeller havner i .\\figures\\<site>\\.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import METER_TYPES, load_meter, weather_raw

SITE = "Robin"                     # standard når du bare trykker Enter
OUT_ROOT = Path("figures").resolve()
OUT_DIR = OUT_ROOT / SITE          # settes på nytt når site er valgt

MIN_TIMER_PER_DOGN = 20            # døgn med færre timer regnes som manglende
MIN_DEKNING_VAER = 0.5             # værvariabler med lavere dekning droppes

VAER = {
    "airTemperature": "Air temperature (°C)",
    "dewTemperature": "Dew point (°C)",
    "windSpeed": "Wind speed (m/s)",
    "seaLvlPressure": "Sea level pressure (hPa)",
    "cloudCoverage": "Cloud cover (oktas)",
    "precipDepth1HR": "Precipitation (mm/h)",
}

# Fast farge per målertype i alle figurer
MAALER_FARGE = {
    "electricity": "#2a78d6", "hotwater": "#eb6834", "chilledwater": "#1baf7a",
    "steam": "#eda100", "gas": "#e87ba4", "water": "#4a3aa7",
    "irrigation": "#5f5e5a", "solar": "#e34948",
}
AAR_FARGE = {2016: "#2a78d6", 2017: "#eb6834"}
TEKST = "#1a1a19"
TEKST_SEKUNDAER = "#5f5e5a"
RUTENETT = "#e4e3dc"
NOYTRAL = "#b4b2a9"
DIVERGERENDE = LinearSegmentedColormap.from_list(
    "div", ["#2a78d6", "#f0efec", "#e34948"])

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
            fig.savefig(f, format="png", bbox_inches="tight")
        print(f"  lagret {navn}")
    except Exception as e:
        print(f"  KUNNE IKKE lagre {navn}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)


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


# ----------------------------------------------------------------------
# Innlasting
# ----------------------------------------------------------------------
def last_maalere():
    """Timesverdier per målertype for siten, negative verdier fjernet."""
    maalere = {}
    for m in METER_TYPES:
        df = load_meter(m, site=SITE)
        if df is None or df.shape[1] == 0 or df.notna().sum().sum() == 0:
            continue
        maalere[m] = df.where(df >= 0)
    return maalere


def last_vaer(indeks):
    """Timesvær for siten. Variabler med for lav dekning eller uten
    variasjon tas ut, og det skrives hvorfor."""
    w = weather_raw[weather_raw["site_id"] == SITE].copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    w = (w.drop(columns=["site_id"]).groupby("timestamp").mean()
         .reindex(indeks))
    w["precipDepth1HR"] = w["precipDepth1HR"].clip(lower=0)  # -1 = "spor" i ISD
    beholdt = []
    for c in VAER:
        dekning = w[c].notna().mean()
        if dekning < MIN_DEKNING_VAER:
            print(f"  {c}: tatt ut, bare {dekning:.0%} av timene har verdi")
        elif w[c].nunique() < 3:
            print(f"  {c}: tatt ut, bare {w[c].nunique()} ulike verdier")
        else:
            beholdt.append(c)
    return w[beholdt]


def daglig(df, hvordan="sum"):
    """Døgnverdi per kolonne. Døgn med for få timer settes til manglende."""
    r = df.resample("D")
    verdi = r.sum(min_count=1) if hvordan == "sum" else r.mean()
    return verdi.where(r.count() >= MIN_TIMER_PER_DOGN)


# ----------------------------------------------------------------------
# Nøkkeltall
# ----------------------------------------------------------------------
def beskriv(v):
    v = pd.Series(v).dropna()
    return {"n": len(v), "min": v.min(), "Q1": v.quantile(0.25),
            "median": v.median(), "Q3": v.quantile(0.75), "max": v.max(),
            "mean": v.mean(), "std": v.std(), "skew": v.skew(),
            "andel_null_%": 100 * (v == 0).mean()}


def sammendrag(maalere, vaer):
    rader = {f"meter: {m}": beskriv(df.to_numpy().ravel())
             for m, df in maalere.items()}
    rader.update({f"weather: {c}": beskriv(vaer[c]) for c in vaer.columns})
    tab = pd.DataFrame(rader).T
    tab["n"] = tab["n"].astype(int)
    print("\nNøkkeltall (timesverdier):")
    print(tab.to_string(float_format=lambda x: f"{x:,.2f}"))
    tab.to_csv(OUT_DIR / "sammendrag.csv", sep=";", encoding="utf-8-sig")
    return tab


# ----------------------------------------------------------------------
# Figurer
# ----------------------------------------------------------------------
def datoakse(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def fig_tidsserie(dag_per_bygg, temp_dag):
    """Daglig snitt per bygg for hver målertype, med 30-dagers glidende snitt.

    Snitt per bygg (ikke sum) gjør kurven uavhengig av hvor mange bygg som
    rapporterer den dagen.
    """
    typer = list(dag_per_bygg)
    fig, akser = plt.subplots(len(typer) + 1, 1, sharex=True,
                              figsize=(12, 1.9 * (len(typer) + 1) + 0.6))
    akser = np.atleast_1d(akser)

    ax = akser[0]
    ax.plot(temp_dag.index, temp_dag, color=NOYTRAL, linewidth=0.8)
    ax.plot(temp_dag.index, temp_dag.rolling(30, center=True, min_periods=15).mean(),
            color=TEKST, label="30-day mean")
    ax.set_ylabel("°C")
    ax.set_title("Outdoor air temperature, daily mean", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    for ax, m in zip(akser[1:], typer):
        s = dag_per_bygg[m].mean(axis=1)
        farge = MAALER_FARGE[m]
        ax.plot(s.index, s, color=farge, linewidth=0.7, alpha=0.45,
                label="daily")
        ax.plot(s.index, s.rolling(30, center=True, min_periods=15).mean(),
                color=farge, label="30-day mean")
        topp = s.idxmax()
        ax.scatter([topp], [s[topp]], s=36, color=farge, zorder=3,
                   edgecolor="white", linewidth=1.5)
        # Etiketten legges på den siden av punktet der det er plass
        venstre = topp > s.index[0] + (s.index[-1] - s.index[0]) * 0.75
        ax.annotate(f"max {s[topp]:,.0f} ({topp:%d.%m.%Y})", (topp, s[topp]),
                    xytext=(-6 if venstre else 6, -2), textcoords="offset points",
                    fontsize=7.5, color=TEKST, va="top",
                    ha="right" if venstre else "left")
        n = dag_per_bygg[m].notna().sum(axis=1)
        ax.set_title(f"{m}: daily use per building (raw units), "
                     f"{n.median():.0f} of {dag_per_bygg[m].shape[1]} buildings "
                     f"reporting on a typical day", loc="left")
        ax.set_ylim(0, s.max() * 1.15)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    datoakse(akser[-1])
    fig.suptitle(f"{SITE}: time series 2016-2017", x=0.01, ha="left",
                 color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "19_tidsserie.png")


def fig_aar_mot_aar(dag_per_bygg, temp_dag):
    """30-dagers glidende snitt for 2016 og 2017 på felles x-akse (dag i året)."""
    serier = {"Air temperature (°C)": temp_dag}
    serier.update({f"{m} (per building, raw units)": d.mean(axis=1)
                   for m, d in dag_per_bygg.items()})
    fig, akser = plt.subplots(len(serier), 1, sharex=True,
                              figsize=(10, 1.9 * len(serier) + 0.6))
    akser = np.atleast_1d(akser)
    for ax, (tittel, s) in zip(akser, serier.items()):
        glatt = s.rolling(30, center=True, min_periods=15).mean()
        for aar, farge in AAR_FARGE.items():
            g = glatt[glatt.index.year == aar]
            ax.plot(g.index.dayofyear, g.to_numpy(), color=farge, label=str(aar))
        ax.set_title(tittel, loc="left")
        # Endring fra 2016 til 2017 for hele året
        a, b = s[s.index.year == 2016].mean(), s[s.index.year == 2017].mean()
        if pd.notna(a) and pd.notna(b) and a != 0 and "temperature" not in tittel:
            ax.text(0.99, 0.95, f"2017 vs 2016: {100 * (b - a) / abs(a):+.0f} %",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=TEKST_SEKUNDAER)
    akser[0].legend(loc="upper left", fontsize=8, ncol=2)
    starter = pd.date_range("2017-01-01", periods=12, freq="MS")
    akser[-1].set_xticks(starter.dayofyear, starter.strftime("%b"))
    akser[-1].set_xlabel("Month (30-day running mean)")
    fig.suptitle(f"{SITE}: 2016 compared with 2017", x=0.01, ha="left",
                 color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "20_aar_mot_aar.png")


def fig_bygg_over_tid(dag_per_bygg):
    """Én rad per bygg, én kolonne per dag: døgnforbruk delt på byggets
    median. Blått = lavere enn vanlig, rødt = høyere, grått = mangler."""
    m = max(dag_per_bygg, key=lambda k: dag_per_bygg[k].shape[1])
    d = dag_per_bygg[m]
    rel = (d / d.median()).T.sort_index()
    rel.index = [b.replace(SITE + "_", "") for b in rel.index]
    cmap = DIVERGERENDE.copy()
    cmap.set_bad(NOYTRAL)
    fig, ax = plt.subplots(figsize=(12, 0.13 * len(rel) + 1.8))
    im = ax.imshow(np.ma.masked_invalid(rel.to_numpy(dtype=float)),
                   aspect="auto", cmap=cmap, vmin=0, vmax=2,
                   interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(len(rel)), rel.index, fontsize=5 if len(rel) > 40 else 7)
    maaneder = pd.date_range(d.index.min(), d.index.max(), freq="QS")
    ax.set_xticks([d.index.get_loc(t) for t in maaneder],
                  maaneder.strftime("%b %Y"))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                      ticks=[0, 0.5, 1, 1.5, 2])
    cb.set_label("Daily use / building's own median")
    cb.ax.set_yticklabels(["0", "0.5", "1 (normal)", "1.5", "≥2"])
    cb.outline.set_visible(False)
    ax.set_title(f"{SITE}: {m} per building and day. Blue = below normal, "
                 "red = above, grey = missing", loc="left")
    lagre(fig, "21_bygg_over_tid.png")


def fig_histogram_maalere(maalere):
    """Lineær akse til venstre, log10 av positive verdier til høyre."""
    fig, akser = plt.subplots(len(maalere), 2, figsize=(11, 2.3 * len(maalere) + 0.6),
                              squeeze=False)
    for (ax_l, ax_g), (m, df) in zip(akser, maalere.items()):
        v = pd.Series(df.to_numpy().ravel()).dropna()
        farge = MAALER_FARGE[m]
        ax_l.hist(v, bins=80, color=farge, edgecolor="white", linewidth=0.3)
        for verdi, stil, navn in [(v.mean(), "-", "mean"),
                                  (v.median(), "--", "median")]:
            ax_l.axvline(verdi, color=TEKST, linestyle=stil, linewidth=1,
                         label=f"{navn} {verdi:,.0f}")
        ax_l.set_yscale("log")
        ax_l.set_ylabel("Hours (log)")
        ax_l.set_title(f"{m}: linear axis, skewness {v.skew():.1f}", loc="left")
        ax_l.legend(fontsize=7.5)

        pos = v[v > 0]
        ax_g.hist(np.log10(pos), bins=80, color=farge, edgecolor="white",
                  linewidth=0.3)
        ax_g.set_ylabel("Hours")
        ax_g.set_title(f"{m}: log10 of positive values, skewness "
                       f"{np.log10(pos).skew():.1f}  ({100 * (v == 0).mean():.0f} % "
                       "zeros left out)", loc="left")
        ax_g.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{10 ** x:g}"))
        ax_l.set_xlabel("Hourly value (raw units)")
        ax_g.set_xlabel("Hourly value (raw units, log scale)")
    fig.suptitle(f"{SITE}: distribution of hourly meter values", x=0.01,
                 ha="left", color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "22_histogram_maalere.png")


def fig_histogram_vaer(vaer):
    kol = list(vaer.columns)
    ncol = 3
    nrad = int(np.ceil(len(kol) / ncol))
    fig, akser = plt.subplots(nrad, ncol, figsize=(12, 2.9 * nrad + 0.5),
                              squeeze=False)
    for ax, c in zip(akser.flat, kol):
        v = vaer[c].dropna()
        ax.hist(v, bins=50, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.axvline(v.mean(), color=TEKST, linewidth=1, label=f"mean {v.mean():.1f}")
        ax.axvline(v.median(), color=TEKST, linewidth=1, linestyle="--",
                   label=f"median {v.median():.1f}")
        ax.set_title(f"{VAER[c]}, skewness {v.skew():.1f}", loc="left")
        ax.set_ylabel("Hours")
        ax.legend(fontsize=7.5)
    for ax in list(akser.flat)[len(kol):]:
        ax.set_visible(False)
    fig.suptitle(f"{SITE}: distribution of hourly weather variables", x=0.01,
                 ha="left", color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "23_histogram_vaer.png")


def fig_korrelasjon(dag_per_bygg, vaer):
    """Spearman-korrelasjon mellom døgnverdier. Spearman fordi flere av
    variablene er sterkt skjeve (se figur 22)."""
    d = pd.DataFrame({VAER[c].split(" (")[0]: daglig(vaer[[c]], "mean")[c]
                      for c in vaer.columns if c != "precipDepth1HR"})
    if "precipDepth1HR" in vaer:
        d["Precipitation"] = daglig(vaer[["precipDepth1HR"]], "sum")["precipDepth1HR"]
    for m, df in dag_per_bygg.items():
        d[m] = df.mean(axis=1)
    d["Weekend"] = (d.index.dayofweek >= 5).astype(float)
    k = d.rank().corr()
    k.to_csv(OUT_DIR / "korrelasjonsmatrise.csv", sep=";", encoding="utf-8-sig")

    # Nedre trekant uten diagonalen: rad 2..n mot kolonne 1..n-1
    vis = k.iloc[1:, :-1]
    n = len(vis)
    maske = np.triu(np.ones((n, n), dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(0.75 * n + 2.5, 0.65 * n + 1.8))
    im = ax.imshow(np.ma.masked_array(vis.to_numpy(), maske), cmap=DIVERGERENDE,
                   vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), vis.columns, rotation=40, ha="right")
    ax.set_yticks(range(n), vis.index)
    for i in range(n):
        for j in range(i + 1):
            v = vis.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 else TEKST)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman ρ")
    cb.outline.set_visible(False)
    ax.set_title(f"{SITE}: correlation matrix of daily values\n"
                 "(meters as mean per building, precipitation as daily sum)",
                 loc="left")
    lagre(fig, "24_korrelasjonsmatrise.png")
    return k


# ----------------------------------------------------------------------
def main():
    global SITE, OUT_DIR
    SITE = velg_site(sorted(weather_raw["site_id"].dropna().unique()))
    OUT_DIR = OUT_ROOT / SITE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLaster målere for {SITE} ...")
    maalere = last_maalere()
    if not maalere:
        print("Ingen målerdata for denne siten.")
        return
    for m, df in maalere.items():
        print(f"  {m:13s} {df.shape[1]:4d} bygg")
    indeks = next(iter(maalere.values())).index

    print("Laster vær ...")
    vaer = last_vaer(indeks)

    sammendrag(maalere, vaer)
    dag_per_bygg = {m: daglig(df) for m, df in maalere.items()}
    temp_dag = daglig(vaer[["airTemperature"]], "mean")["airTemperature"]

    print("\nFigurer:")
    fig_tidsserie(dag_per_bygg, temp_dag)
    fig_aar_mot_aar(dag_per_bygg, temp_dag)
    fig_bygg_over_tid(dag_per_bygg)
    fig_histogram_maalere(maalere)
    fig_histogram_vaer(vaer)
    k = fig_korrelasjon(dag_per_bygg, vaer)

    # Sterkeste par i matrisen
    par = (k.where(np.tril(np.ones(k.shape, dtype=bool), k=-1)).stack()
           .sort_values(key=abs, ascending=False))
    print("\nSterkeste korrelasjoner (Spearman, døgnverdier):")
    for (a, b), v in par.head(6).items():
        print(f"  {a:22s} - {b:22s} {v:+.2f}")
    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
