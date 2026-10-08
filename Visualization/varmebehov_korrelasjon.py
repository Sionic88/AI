"""
Hva henger sammen med varmebehovet? Building Data Genome Project 2.

Varmebehov måles her med byggenes varmemålere (hotwater, steam, gas). Hver
bygg-måler-kombinasjon er én serie. Enhetene er ulike fra site til site, så
alle serier gjøres relative: forbruk delt på seriens eget snitt (1 = vanlig dag).

Skriptet ser på mer enn de åpenbare faktorene temperatur og vind:

  1  Rå korrelasjon        daglig varme mot 17 faktorer (fukt, skydekke, nedbør,
                           lufttrykk og trykkendring, døgnamplitude, dagslengde,
                           helg, strømforbruk i samme bygg ...)
  2  Temperaturkorrigert   samme faktorer, men mot det som er igjen etter at
                           temperaturen er trukket fra. Viser hva som forklarer
                           varmebehovet UTOVER temperaturen.
  3  Tidsforsinkelse       hvor mange timer henger varmen etter temperaturen, og
                           hvor lang "termisk hukommelse" har byggene?
  4  Samspill              endres temperaturkurven når det blåser, er fuktig,
                           overskyet, helg, vår kontra høst eller høyt strømforbruk?
  5  Driftsmønster         time x ukedag, vinter mot sommer
  6  Byggegenskaper        endringspunktmodell per serie (balansetemperatur,
                           følsomhet, grunnlast) korrelert mot byggeår, areal,
                           etasjer, brukere, klima og bygningstype

Korrelasjonene er Spearman (rangkorrelasjon), regnet per serie og oppsummert
med median. Skriptet vasker ikke dataene; det fjerner bare negative verdier og
åpenbare spikes (mer enn 10 ganger seriens 99-persentil).

Bruk:
    python varmebehov_korrelasjon.py            # spør etter site, Enter = alle
    python varmebehov_korrelasjon.py alle
    python varmebehov_korrelasjon.py Robin

Figurer og tabeller havner i .\\figures\\varmebehov\\<site eller alle>\\.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import load_meter, metadata_raw, weather_raw

VARME = ["hotwater", "steam", "gas"]
OUT_ROOT = Path("figures").resolve() / "varmebehov"
OUT_DIR = OUT_ROOT / "alle"          # settes på nytt når site er valgt

MIN_DAGER = 120                      # minste antall gyldige dager per serie
T_BIN = 1.5                          # bredde (°C) på temperaturbøttene i korrigeringen

# Faktorene som testes, med etikett til figurene
FAKTORER = {
    "T_mean":    "Air temperature",
    "T_7d":      "7-day mean temperature",
    "T_change":  "Temp. change since yesterday",
    "T_range":   "Diurnal temperature range",
    "dew":       "Dew point",
    "rh":        "Relative humidity",
    "abs_hum":   "Absolute humidity",
    "cloud":     "Cloud cover",
    "precip":    "Precipitation",
    "pressure":  "Sea level pressure",
    "dpressure": "Pressure change since yesterday",
    "wind":      "Wind speed",
    "wind_max":  "Max wind speed",
    "windchill": "Wind x cold",
    "daylength": "Day length",
    "spring":    "Spring half (Jan-Jun)",
    "weekend":   "Weekend",
    "elec":      "Electricity, same building",
}
# Disse er bare temperatur i annen form og tas ut av den korrigerte analysen
TEMPERATURFAKTORER = ["T_mean"]

# Fast fargeoppsett (samme kategoriske rekkefølge som sammenlign_bygg.py)
FARGER = ["#2a78d6", "#eb6834", "#1baf7a"]
TEKST = "#1a1a19"
TEKST_SEKUNDAER = "#5f5e5a"
RUTENETT = "#e4e3dc"
NOYTRAL = "#b4b2a9"
DIVERGERENDE = LinearSegmentedColormap.from_list(
    "div", ["#2a78d6", "#f0efec", "#e34948"])
SEKVENSIELL = LinearSegmentedColormap.from_list(
    "seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#104281"])

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


def lagre_csv(df, navn, **kw):
    df.to_csv(OUT_DIR / navn, **kw)
    print(f"  lagret {navn}")


def velg_site(sites):
    """Site fra argument eller spørsmål. Tom linje eller 'alle' gir alle sites."""
    oppslag = {s.lower(): s for s in sites}
    if len(sys.argv) > 1:
        svar = sys.argv[1].strip().lower()
        if svar in ("alle", "all"):
            return None
        if svar in oppslag:
            return oppslag[svar]
        print(f"Ukjent site '{sys.argv[1]}'.")

    print("Sites med varmemålere:")
    for i, s in enumerate(sites, start=1):
        print(f"  {i:2d}. {s}")
    while True:
        svar = input("\nSkriv inn site (navn eller nummer, Enter = alle): ")
        svar = svar.strip().lower()
        if svar in ("", "alle", "all"):
            return None
        if svar.isdigit() and 1 <= int(svar) <= len(sites):
            return sites[int(svar) - 1]
        if svar in oppslag:
            return oppslag[svar]
        print(f"Fant ikke '{svar}'. Prøv igjen.")


def rangkorr(a, b, minimum=60):
    """Spearman uten scipy: Pearson på rangeringene, bare felles gyldige rader."""
    m = a.notna() & b.notna()
    if m.sum() < minimum or a[m].nunique() < 2 or b[m].nunique() < 2:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def korr_np(a, b, minimum=1000):
    """Pearson på numpy-arrays med NaN. Brukes på ferdig rangerte timeserier."""
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < minimum:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def site_av(serie):
    return serie.split("|")[0].split("_")[0]


# ----------------------------------------------------------------------
# Innlasting
# ----------------------------------------------------------------------
def last_varme(site):
    """Alle varmeserier (timesverdier) som bygg|måler, valgfritt for én site."""
    serier = {}
    for m in VARME:
        df = load_meter(m, site=site)
        if df is None or df.empty:
            continue
        for b in df.columns:
            s = df[b].where(df[b] >= 0)
            g = s.dropna()
            if len(g) < 0.5 * len(s) or g.nunique() < 20 or g.mean() <= 0:
                continue
            tak = 10 * g.quantile(0.99)
            if tak > 0:
                s = s.where(s <= tak)
            serier[f"{b}|{m}"] = s
    return pd.DataFrame(serier)


def last_strom(bygg):
    e = load_meter("electricity", buildings=bygg)
    if e is None:
        return pd.DataFrame()
    return e.where(e >= 0)


def vaer_timer(site, indeks):
    """Timesvær for én site, avledede fuktvariabler, på målernes tidsakse."""
    w = weather_raw[weather_raw["site_id"] == site].copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    w = (w.drop(columns=["site_id"]).groupby("timestamp").mean()
         .reindex(indeks))
    T, Td = w["airTemperature"], w["dewTemperature"]
    # Magnus-formelen: metningstrykk og faktisk damptrykk (hPa)
    es = 6.112 * np.exp(17.62 * T / (243.12 + T))
    e = 6.112 * np.exp(17.62 * Td / (243.12 + Td))
    w["rh"] = (100 * e / es).clip(upper=100)
    w["abs_hum"] = 216.7 * e / (273.15 + T)          # g/m3
    w["windchill"] = w["windSpeed"] * (18 - T).clip(lower=0)
    w["precip"] = w["precipDepth1HR"].clip(lower=0)  # -1 betyr "spor" i ISD
    return w


def dagslengde(dager, lat):
    """Timer med dagslys fra breddegrad og dag i året."""
    if pd.isna(lat):
        return pd.Series(np.nan, index=dager)
    doy = dager.dayofyear.to_numpy()
    dekl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    x = np.clip(-np.tan(np.radians(lat)) * np.tan(dekl), -1, 1)
    return pd.Series(24 / np.pi * np.arccos(x), index=dager)


def vaer_daglig(w, lat):
    """Daglige værfaktorer. Dager med færre enn 18 temperaturtimer forkastes."""
    T = w["airTemperature"]
    dag = T.resample("D")
    gyldig = dag.count() >= 18
    d = pd.DataFrame({
        "T_mean": dag.mean(),
        "T_range": dag.max() - dag.min(),
        "dew": w["dewTemperature"].resample("D").mean(),
        "rh": w["rh"].resample("D").mean(),
        "abs_hum": w["abs_hum"].resample("D").mean(),
        "cloud": w["cloudCoverage"].resample("D").mean(),
        "precip": w["precip"].resample("D").sum(min_count=12),
        "pressure": w["seaLvlPressure"].resample("D").mean(),
        "wind": w["windSpeed"].resample("D").mean(),
        "wind_max": w["windSpeed"].resample("D").max(),
        "windchill": w["windchill"].resample("D").mean(),
    })
    d = d.where(gyldig)
    d["T_7d"] = d["T_mean"].rolling(7, min_periods=5).mean()
    d["T_change"] = d["T_mean"].diff()
    d["dpressure"] = d["pressure"].diff()
    d["daylength"] = dagslengde(d.index, lat)
    d["spring"] = (d.index.month <= 6).astype(float)
    d["weekend"] = (d.index.dayofweek >= 5).astype(float)
    return d


def daglig_relativ(df):
    """Døgnsnitt delt på seriens snitt. Dager med under 18 timer forkastes."""
    dag = df.resample("D").mean().where(df.resample("D").count() >= 18)
    return dag / dag.mean()


# ----------------------------------------------------------------------
# Analyser per serie
# ----------------------------------------------------------------------
def temperaturkorriger(y, T):
    """Trekker fra medianen i hver temperaturbøtte. Det som er igjen er
    varmebehov som temperaturen ikke forklarer."""
    b = np.floor(T / T_BIN)
    g = y.groupby(b)
    return (y - g.transform("median")).where(g.transform("size") >= 5)


def endringspunkt(T, y):
    """Varme = grunnlast + følsomhet * max(0, Tb - T). Søker Tb på et rutenett.

    Returnerer (Tb, grunnlast, følsomhet, R2) eller None.
    """
    T, y = T.to_numpy(), y.to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    beste = None
    for tb in np.arange(0, 24.5, 0.5):
        x = np.clip(tb - T, 0, None)
        if x.std() == 0:
            continue
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if coef[1] <= 0:
            continue
        r2 = 1 - ((y - X @ coef) ** 2).sum() / sst
        if beste is None or r2 > beste[3]:
            beste = (tb, coef[0], coef[1], r2)
    return beste


def analyser_serier(dag_varme, dag_vaer, dag_strom):
    """Rå og temperaturkorrigert korrelasjon, og endringspunktmodell, per serie."""
    raa, korr, modell, lang = {}, {}, {}, []
    for kol in dag_varme.columns:
        bygg, maaler = kol.split("|")
        d = dag_vaer[site_av(kol)].copy()
        d["elec"] = dag_strom[bygg] if bygg in dag_strom else np.nan
        d["y"] = dag_varme[kol]
        d = d.dropna(subset=["y", "T_mean"])
        if len(d) < MIN_DAGER:
            continue

        raa[kol] = {f: rangkorr(d["y"], d[f]) for f in FAKTORER}

        rest = temperaturkorriger(d["y"], d["T_mean"])
        korr[kol] = {f: rangkorr(rest, d[f]) for f in FAKTORER
                     if f not in TEMPERATURFAKTORER}
        m = rest.notna()
        r2_temp = 1 - rest[m].var() / d.loc[m, "y"].var()

        cp = endringspunkt(d["T_mean"], d["y"])
        if cp is not None:
            modell[kol] = {
                "building_id": bygg, "meter": maaler, "site_id": site_av(kol),
                "balanse_temp": cp[0],
                "foelsomhet_pst_per_grad": 100 * cp[2],
                "grunnlast_pst": 100 * cp[1],
                "r2_endringspunkt": cp[3],
                "r2_temperatur_ikkelineaer": r2_temp,
                "dager": len(d),
            }
        d["serie"] = kol
        d["site"] = site_av(kol)
        lang.append(d)

    return (pd.DataFrame(raa).T, pd.DataFrame(korr).T,
            pd.DataFrame(modell).T, pd.concat(lang))


def tidsforsinkelse(varme, temp_timer):
    """Spearman mellom timesvarme og temperatur forskjøvet bakover (lag), og
    mot glidende snitt av temperaturen over ulike vinduer."""
    lags = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72]
    vinduer = [1, 3, 6, 12, 24, 48, 72, 120, 168, 336]
    # Ranger temperaturvariantene én gang per site; da blir løkken rask
    ferdig = {}
    for site, T in temp_timer.items():
        ferdig[site] = (
            {L: T.shift(L).rank().to_numpy() for L in lags},
            {v: T.rolling(v, min_periods=v // 2 + 1).mean().rank().to_numpy()
             for v in vinduer},
        )
    lag_res, vindu_res = {}, {}
    for kol in varme.columns:
        h = varme[kol].rank().to_numpy()
        lagr, vindr = ferdig[site_av(kol)]
        lag_res[kol] = {L: -korr_np(h, lagr[L]) for L in lags}
        vindu_res[kol] = {v: -korr_np(h, vindr[v]) for v in vinduer}
    return pd.DataFrame(lag_res).T, pd.DataFrame(vindu_res).T


# ----------------------------------------------------------------------
# Figurer
# ----------------------------------------------------------------------
def varmekart(tabell, tittel, navn, n=None):
    """Rader = site eller serie, kolonner = faktorer, celle = median rho."""
    tabell = tabell.rename(columns=FAKTORER)
    rader, kolonner = tabell.shape
    vis_tall = rader <= 30
    fig, ax = plt.subplots(figsize=(0.62 * kolonner + 3, 0.32 * rader + 2.4))
    im = ax.imshow(tabell.to_numpy(dtype=float), cmap=DIVERGERENDE,
                   vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(kolonner), tabell.columns, rotation=40, ha="right")
    etiketter = [f"{r}  (n={n[r]})" if n is not None else r
                 for r in tabell.index]
    ax.set_yticks(range(rader), etiketter, fontsize=8 if rader <= 30 else 5)
    if vis_tall:
        for i in range(rader):
            for j in range(kolonner):
                v = tabell.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > 0.6 else TEKST)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ (median across series)")
    cb.outline.set_visible(False)
    ax.set_title(tittel, loc="left")
    lagre(fig, navn)


def oppsummer_per_site(tabell):
    """Median per site pluss en rad for alle samlet."""
    s = tabell.groupby(tabell.index.map(site_av)).median()
    n = tabell.groupby(tabell.index.map(site_av)).size()
    if len(s) > 1:
        s.loc["All sites"] = tabell.median()
        n.loc["All sites"] = len(tabell)
    return s, n


def rangering(korr, navn, tittel):
    """Median og kvartilbredde per faktor, sortert etter styrke."""
    med = korr.median()
    orden = med.abs().sort_values().index
    q1, q3 = korr.quantile(0.25)[orden], korr.quantile(0.75)[orden]
    y = np.arange(len(orden))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(orden) + 1.5))
    ax.axvline(0, color=NOYTRAL, linewidth=1)
    ax.hlines(y, q1, q3, color=FARGER[0], alpha=0.35, linewidth=6)
    farge = [FARGER[1] if v > 0 else FARGER[0] for v in med[orden]]
    ax.scatter(med[orden], y, s=36, color=farge, zorder=3,
               edgecolor="white", linewidth=1.5)
    for yi, f in zip(y, orden):
        ax.text(q3[f] + 0.02, yi, f"{med[f]:+.2f}", va="center",
                fontsize=7.5, color=TEKST_SEKUNDAER)
    ax.set_yticks(y, [FAKTORER[f] for f in orden])
    ax.set_xlabel("Spearman ρ with heating left after temperature is removed\n"
                  "(dot = median, bar = middle 50 % of series)")
    ax.set_title(tittel, loc="left")
    ax.grid(axis="y", visible=False)
    lagre(fig, navn)


def figur_tidsforsinkelse(lag, vindu, navn):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tab, xlab, tittel in [
        (a1, lag, "Lag (hours the temperature is shifted back)",
         "Heating vs. temperature L hours earlier"),
        (a2, vindu, "Window for running mean temperature (hours, log)",
         "Heating vs. running mean temperature"),
    ]:
        x = np.array(tab.columns, dtype=float)
        for site, g in tab.groupby(tab.index.map(site_av)):
            ax.plot(x, g.median(), color=NOYTRAL, linewidth=0.8, alpha=0.8)
        med = tab.median()
        ax.fill_between(x, tab.quantile(0.25), tab.quantile(0.75),
                        color=FARGER[0], alpha=0.15, linewidth=0)
        ax.plot(x, med, color=FARGER[0], label="Median, all series")
        topp = med.idxmax()
        ax.scatter([topp], [med[topp]], s=40, color=FARGER[0], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"peak at {topp} h", (topp, med[topp]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=TEKST)
        ax.set_xlabel(xlab)
        ax.set_title(tittel, loc="left")
    a2.set_xscale("log")
    a1.set_ylabel("−Spearman ρ (higher = stronger heating response)")
    a1.plot([], [], color=NOYTRAL, linewidth=0.8, label="Median per site")
    a1.legend(loc="lower left")
    fig.tight_layout()
    lagre(fig, navn)


def figur_samspill(lang, navn):
    """Temperaturkurven delt etter en tredje faktor. Hver serie får like mye
    vekt: snitt per serie og temperaturbøtte, så median på tvers av seriene."""
    lang = lang.copy()
    lang["tbin"] = np.floor(lang["T_mean"] / 2) * 2 + 1
    paneler = [
        ("wind", "Wind speed", "tercile"),
        ("rh", "Relative humidity", "tercile"),
        ("cloud", "Cloud cover", "tercile"),
        ("elec", "Electricity use, same building", "tercile"),
        ("weekend", "Weekday / weekend", {0: "Weekday", 1: "Weekend"}),
        ("spring", "Half of year", {0: "Jul-Dec (autumn)", 1: "Jan-Jun (spring)"}),
    ]
    fig, akser = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, (f, tittel, grupper) in zip(akser.flat, paneler):
        d = lang.dropna(subset=[f])
        if d.empty:
            ax.set_title(f"{tittel}: no data", loc="left")
            continue
        if grupper == "tercile":
            pct = d.groupby("serie")[f].rank(pct=True)
            d = d.assign(gruppe=np.ceil(pct * 3).clip(1, 3) - 1)
            grupper = {0: "Lowest third", 1: "Middle third", 2: "Highest third"}
        else:
            d = d.assign(gruppe=d[f])
        per_serie = d.groupby(["gruppe", "tbin", "serie"])["y"].agg(["mean", "size"])
        per_serie = per_serie[per_serie["size"] >= 3]["mean"]
        kurve = per_serie.groupby(["gruppe", "tbin"]).agg(["median", "size"])
        kurve = kurve[kurve["size"] >= 5]["median"]
        for i, (g, etikett) in enumerate(grupper.items()):
            if g in kurve.index.get_level_values(0):
                k = kurve.loc[g]
                ax.plot(k.index, k.values, color=FARGER[i], label=etikett,
                        marker="o", markersize=3)
        ax.set_title(tittel, loc="left")
        ax.legend(fontsize=7.5)
    for ax in akser[1]:
        ax.set_xlabel("Daily mean air temperature (°C)")
    for ax in akser[:, 0]:
        ax.set_ylabel("Heating, relative to series mean")
    fig.suptitle("Does the temperature response change with a third factor?",
                 x=0.01, ha="left", color=TEKST)
    fig.tight_layout()
    lagre(fig, navn)


def figur_drift(varme_rel, navn):
    """Time x ukedag, median relativ varme på tvers av seriene."""
    idx = varme_rel.index
    perioder = {"Winter (Dec-Feb)": idx.month.isin([12, 1, 2]),
                "Summer (Jun-Aug)": idx.month.isin([6, 7, 8])}
    kart = {}
    for p, m in perioder.items():
        sub = varme_rel[m]
        g = sub.groupby([sub.index.dayofweek, sub.index.hour]).mean()
        kart[p] = g.median(axis=1).unstack()
    vmax = max(k.to_numpy().max() for k in kart.values())
    fig, akser = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, (p, k) in zip(akser, kart.items()):
        im = ax.imshow(k.to_numpy(), cmap=SEKVENSIELL, vmin=0, vmax=vmax,
                       aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(0, 24, 3), range(0, 24, 3))
        ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("Hour of day")
        ax.set_title(p, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=akser, fraction=0.025, pad=0.02)
    cb.set_label("Heating, relative to series mean")
    cb.outline.set_visible(False)
    lagre(fig, navn)


def figur_forklart(modell, navn):
    """Hvor mye av den daglige variasjonen forklarer temperaturen alene?"""
    r2 = modell["r2_temperatur_ikkelineaer"].astype(float).dropna()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(r2, bins=np.linspace(min(0, r2.min()), 1, 26), color=FARGER[0],
            edgecolor="white", linewidth=1.5)
    ax.axvline(r2.median(), color=TEKST, linewidth=1, linestyle="--")
    ax.annotate(f"median {r2.median():.2f}", (r2.median(), ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Share of daily variation explained by temperature (R²)")
    ax.set_ylabel("Number of series")
    ax.set_title("What is left for other factors to explain?", loc="left")
    ax.grid(axis="x", visible=False)
    lagre(fig, navn)


def metadata_tabell(modell, temp_site):
    md = metadata_raw.set_index("building_id")
    m = modell.join(md[["sqm", "yearbuilt", "numberoffloors", "occupants",
                        "lat", "primaryspaceusage", "heatingtype",
                        "energystarscore"]], on="building_id")
    m["sqm_per_floor"] = m["sqm"] / m["numberoffloors"]
    m["sqm_per_occupant"] = m["sqm"] / m["occupants"]
    m["site_mean_temp"] = m["site_id"].map(temp_site)
    for c in ["balanse_temp", "foelsomhet_pst_per_grad", "grunnlast_pst",
              "r2_endringspunkt", "r2_temperatur_ikkelineaer", "sqm",
              "yearbuilt", "numberoffloors", "occupants", "lat",
              "energystarscore"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    # Tb på kanten av søkeområdet betyr at modellen ikke fant noe knekkpunkt
    m["knekkpunkt_funnet"] = m["balanse_temp"].between(0.5, 23.5)
    return m


PARAMETRE = {
    "balanse_temp": "Balance temperature",
    "foelsomhet_pst_per_grad": "Sensitivity (% per °C)",
    "grunnlast_pst": "Base load (% of mean)",
    "r2_temperatur_ikkelineaer": "R² temperature",
}
EGENSKAPER = {
    "site_mean_temp": "Site mean temp.",
    "lat": "Latitude",
    "yearbuilt": "Year built",
    "sqm": "Floor area",
    "numberoffloors": "Floors",
    "sqm_per_floor": "Area per floor",
    "occupants": "Occupants",
    "sqm_per_occupant": "Area per occupant",
    "energystarscore": "Energy Star score",
}


def figur_metadata(m, navn):
    rho = pd.DataFrame(index=list(PARAMETRE), columns=list(EGENSKAPER), dtype=float)
    n = rho.copy()
    for p in PARAMETRE:
        for e in EGENSKAPER:
            ok = m[p].notna() & m[e].notna()
            n.loc[p, e] = ok.sum()
            rho.loc[p, e] = rangkorr(m[p], m[e], minimum=10)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(rho.to_numpy(dtype=float), cmap=DIVERGERENDE, vmin=-1,
                   vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(EGENSKAPER)), EGENSKAPER.values(), rotation=30,
                  ha="right")
    ax.set_yticks(range(len(PARAMETRE)), PARAMETRE.values())
    for i, p in enumerate(PARAMETRE):
        for j, e in enumerate(EGENSKAPER):
            v = rho.loc[p, e]
            tekst = f"{v:.2f}\nn={int(n.loc[p, e])}" if pd.notna(v) else "–"
            ax.text(j, i, tekst, ha="center", va="center", fontsize=6.5,
                    color="white" if pd.notna(v) and abs(v) > 0.6 else TEKST)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ across series")
    cb.outline.set_visible(False)
    ax.set_title("Heating behaviour (change-point model) vs. building properties",
                 loc="left")
    lagre(fig, navn)
    rho.to_csv(OUT_DIR / "metadata_korrelasjon.csv")
    return rho, n


def figur_kategori(m, kolonne, navn, tittel):
    """Balansetemperatur og følsomhet per kategori (minst 5 serier)."""
    d = m.dropna(subset=[kolonne])
    antall = d[kolonne].value_counts()
    kat = antall[antall >= 5].index
    if len(kat) == 0:
        return
    d = d[d[kolonne].isin(kat)]
    orden = d.groupby(kolonne)["balanse_temp"].median().sort_values().index
    rng = np.random.default_rng(0)
    fig, akser = plt.subplots(1, 2, figsize=(11, 0.4 * len(orden) + 1.8),
                              sharey=True)
    for ax, p in zip(akser, ["balanse_temp", "foelsomhet_pst_per_grad"]):
        for i, k in enumerate(orden):
            v = d.loc[d[kolonne] == k, p].dropna()
            ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10,
                       color=FARGER[0], alpha=0.45, linewidth=0)
            ax.scatter([v.median()], [i], s=60, marker="|", color=TEKST,
                       linewidth=2, zorder=3)
        ax.set_xlabel(PARAMETRE[p])
        ax.grid(axis="y", visible=False)
    # Noen få ekstreme følsomheter ville ellers presse sammen aksen
    tak = d["foelsomhet_pst_per_grad"].quantile(0.98)
    utenfor = (d["foelsomhet_pst_per_grad"] > tak).sum()
    akser[1].set_xlim(0, tak * 1.05)
    if utenfor:
        akser[1].set_xlabel(f"{PARAMETRE['foelsomhet_pst_per_grad']}  "
                            f"({utenfor} series above {tak:.0f} not shown)")
    akser[0].set_xlabel("Balance temperature (°C)")
    akser[0].set_yticks(range(len(orden)),
                        [f"{k}  (n={antall[k]})" for k in orden])
    akser[0].set_title(tittel, loc="left")
    fig.tight_layout()
    lagre(fig, navn)


# ----------------------------------------------------------------------
def main():
    global OUT_DIR
    md = metadata_raw
    har_varme = md[(md[VARME] == "Yes").any(axis=1)]["site_id"]
    site = velg_site(sorted(har_varme.dropna().unique()))
    OUT_DIR = OUT_ROOT / (site or "alle")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLaster varmemålere for {site or 'alle sites'} ...")
    varme = last_varme(site)
    if varme.empty:
        print("Ingen brukbare varmeserier.")
        return
    sites = sorted({site_av(k) for k in varme.columns})
    print(f"  {varme.shape[1]} serier på {len(sites)} site(s)")

    print("Laster strøm og vær ...")
    bygg = sorted({k.split("|")[0] for k in varme.columns})
    strom = last_strom(bygg).reindex(varme.index)
    lat = md.groupby("site_id")["lat"].median()
    vaer_t = {s: vaer_timer(s, varme.index) for s in sites}
    dag_vaer = {s: vaer_daglig(w, lat.get(s)) for s, w in vaer_t.items()}
    temp_site = pd.Series({s: w["airTemperature"].mean()
                           for s, w in vaer_t.items()})

    print("Regner korrelasjoner per serie ...")
    dag_varme = daglig_relativ(varme)
    dag_strom = daglig_relativ(strom) if not strom.empty else pd.DataFrame()
    raa, korr, modell, lang = analyser_serier(dag_varme, dag_vaer, dag_strom)
    print(f"  {len(raa)} serier med minst {MIN_DAGER} gyldige dager")
    if raa.empty:
        return

    print("Regner tidsforsinkelse på timesdata ...")
    varme_rel = (varme / varme.mean())[raa.index]
    lag, vindu = tidsforsinkelse(
        varme_rel, {s: w["airTemperature"] for s, w in vaer_t.items()})

    print("\nFigurer og tabeller:")
    lagre_csv(raa, "korrelasjon_raa.csv")
    lagre_csv(korr, "korrelasjon_temperaturkorrigert.csv")
    lagre_csv(lag, "tidsforsinkelse_lag.csv")
    lagre_csv(vindu, "tidsforsinkelse_vindu.csv")

    # Én site: én rad per serie. Alle: median per site.
    if site:
        orden = raa["T_mean"].sort_values().index
        rad_raa, n_raa = raa.loc[orden], None
        rad_korr, n_korr = korr.loc[orden], None
    else:
        rad_raa, n_raa = oppsummer_per_site(raa)
        rad_korr, n_korr = oppsummer_per_site(korr)
    hvor = site or "all sites"
    varmekart(rad_raa, f"{hvor}: daily heating vs. factors (raw)",
              "01_korrelasjon_raa.png", n_raa)
    varmekart(rad_korr, f"{hvor}: heating left after temperature is removed "
              "vs. factors", "02_korrelasjon_temperaturkorrigert.png", n_korr)
    rangering(korr, "03_faktor_rangering.png",
              f"{hvor}: what drives heating beyond temperature?")
    figur_forklart(modell, "04_temperatur_forklart.png")
    figur_tidsforsinkelse(lag, vindu, "05_tidsforsinkelse.png")
    figur_samspill(lang, "06_samspill_temperatur.png")
    figur_drift(varme_rel, "07_driftsmoenster.png")

    m = metadata_tabell(modell, temp_site)
    lagre_csv(m, "endringspunkt_modell.csv")
    mk = m[m["knekkpunkt_funnet"]]
    print(f"  ({len(m) - len(mk)} serier uten tydelig knekkpunkt holdes "
          f"utenfor figur 08-10)")
    figur_metadata(mk, "08_byggegenskaper.png")
    figur_kategori(mk, "primaryspaceusage", "09_bygningstype.png",
                   "Balance temperature and sensitivity by building use")
    figur_kategori(mk, "meter", "10_maalertype.png",
                   "Balance temperature and sensitivity by meter type")

    # Kort oppsummering i konsollen
    print(f"\nOppsummering ({len(raa)} serier):")
    print(f"  temperatur forklarer i median "
          f"{m['r2_temperatur_ikkelineaer'].median():.0%} av den daglige variasjonen")
    print(f"  sterkeste lag mot timesvarme:   {lag.median().idxmax()} timer")
    print(f"  beste vindu for glidende snitt: {vindu.median().idxmax()} timer")
    print(f"  median balansetemperatur:       {mk['balanse_temp'].median():.1f} °C")
    print("\n  Faktorer utover temperatur (median rho, temperaturkorrigert):")
    med = korr.median()
    for f in med.abs().sort_values(ascending=False).index:
        print(f"    {FAKTORER[f]:34s} {med[f]:+.3f}")
    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
