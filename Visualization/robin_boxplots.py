"""
MEPA2315 Assignment 1 - Data understanding
Building Data Genome Project 2, enten en hel site (standard "Robin",
University College London) eller ett spesifikt bygg.

Laster målerdata for valgt site eller bygg og lager boksplott og en
statistikktabell. Skriptet vasker ikke dataene; det gjøres et annet sted.

Bruk:
    python robin_boxplots.py                              # spør "Skriv inn site"
    python robin_boxplots.py --site Panther               # annen site
    python robin_boxplots.py --bygg Robin_education_Kiera # ett bygg
    python robin_boxplots.py --bygg Kiera                 # kortform, --site legges foran
    python robin_boxplots.py --liste                      # list bygg på site

Figurer havner i figures\\<site eller bygg>\\, tabellen skrives til
konsollen og til statistikk.csv i samme mappe.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import METER_TYPES, load_meter, metadata

# ----------------------------------------------------------------------
# Konfigurasjon
# ----------------------------------------------------------------------
SITE = "Robin"              # standard site, kan overstyres med --site
BYGG = None                 # building_id når ett enkelt bygg analyseres
NAVN = SITE                 # brukes i titler og mappenavn
OUT_ROOT = Path("figures")
OUT_DIR = OUT_ROOT / NAVN

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


# ----------------------------------------------------------------------
# Innlasting
# ----------------------------------------------------------------------
def load_metadata():
    md = metadata.reset_index()
    if BYGG:
        return md[md["building_id"] == BYGG].copy()
    return md[md["site_id"] == SITE].copy()


def last_maler(meter_type):
    """Leser en målerfil, returnerer kolonnene for valgt site eller bygg."""
    if BYGG:
        df = load_meter(meter_type, buildings=[BYGG])
    else:
        df = load_meter(meter_type, site=SITE)
    if df is None or df.empty:
        return None
    return df


def to_long(df, meter_type):
    out = (df.stack(future_stack=True)
             .rename("value")
             .reset_index()
             .rename(columns={"level_1": "building_id"}))
    out["meter"] = meter_type
    return out


# ----------------------------------------------------------------------
# Statistikk
# ----------------------------------------------------------------------
def statistikk(long_df):
    rows = []
    for meter, g in long_df.groupby("meter"):
        x = g["value"].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        rows.append({
            "meter": meter,
            "n": x.size,
            "min": x.min(),
            "Q1": q1,
            "median": np.median(x),
            "Q3": q3,
            "max": x.max(),
            "mean": x.mean(),
            "std": x.std(),
            "skew": pd.Series(x).skew(),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ----------------------------------------------------------------------
# Plottehjelper
# ----------------------------------------------------------------------
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


def boxplot(ax, groups, labels, log=True):
    if log:                                 # log-akse kan ikke vise 0 og negative
        groups = [np.asarray(g)[np.asarray(g) > 0] for g in groups]
    ax.boxplot(groups, tick_labels=labels, showfliers=True,
               flierprops=dict(marker=".", markersize=2, alpha=0.25),
               medianprops=dict(color="crimson"))
    if log:
        ax.set_yscale("log")


def fig_per_meter_type(long_df, filnavn):
    order = (long_df.groupby("meter")["value"].median()
             .sort_values(ascending=False).index.tolist())
    groups, labels = [], []
    for m in order:
        v = long_df.loc[long_df["meter"] == m, "value"].dropna()
        v = v[v > 0]
        if len(v):
            groups.append(v.to_numpy())
            labels.append(f"{m}\n(n={len(v):,})")
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2, 4.5))
    boxplot(ax, groups, labels)
    ax.set_ylabel("Hourly value (kWh, litres for water/irrigation)")
    ax.set_title(f"{NAVN}: distribution per meter type")
    fig.tight_layout()
    lagre(fig, filnavn)


def fig_per_building(elec, filnavn, top=25):
    cols = elec.median().sort_values(ascending=False).index[:top].tolist()
    groups = [elec[c].dropna().to_numpy() for c in cols]
    labels = [c.replace(SITE + "_", "") for c in cols]

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cols)), 5))
    boxplot(ax, groups, labels)
    ax.set_ylabel("Electricity (kWh/hour)")
    ax.set_title(f"{NAVN}: electricity per building")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    lagre(fig, filnavn)


def fig_per_month(elec, filnavn):
    s = elec.stack(future_stack=True).rename("value").reset_index()
    s["month"] = s["timestamp"].dt.to_period("M").astype(str)
    order = sorted(s["month"].unique())
    groups = [s.loc[s["month"] == m, "value"].dropna().to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    boxplot(ax, groups, order)
    ax.set_ylabel("Electricity (kWh/hour)")
    ax.set_title(f"{NAVN}: electricity per month")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    lagre(fig, filnavn)


def fig_daily_profile(elec, filnavn):
    s = elec.stack(future_stack=True).rename("value").reset_index()
    s["hour"] = s["timestamp"].dt.hour
    s["weekend"] = s["timestamp"].dt.dayofweek >= 5

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (is_we, tittel) in zip(axes, [(False, "Weekday"), (True, "Weekend")]):
        sub = s[s["weekend"] == is_we]
        groups = [sub.loc[sub["hour"] == h, "value"].dropna().to_numpy()
                  for h in range(24)]
        boxplot(ax, groups, [str(h) for h in range(24)])
        ax.set_title(f"{NAVN}: {tittel}")
        ax.set_xlabel("Hour of day")
    axes[0].set_ylabel("Electricity (kWh/hour)")
    fig.tight_layout()
    lagre(fig, filnavn)


def fig_normalised(elec, meta, filnavn):
    sqm = meta.set_index("building_id")["sqm"]
    usage = meta.set_index("building_id")["primaryspaceusage"]
    cols = [c for c in elec.columns if c in sqm.index and sqm[c] > 0]
    if not cols:
        return
    norm = elec[cols].divide(sqm[cols], axis=1)

    s = norm.stack(future_stack=True).rename("value").reset_index()
    s = s.rename(columns={"level_1": "building_id"})
    s["usage"] = s["building_id"].map(usage)
    s = s.dropna(subset=["usage", "value"])
    s = s[s["value"] > 0]
    if s.empty:
        return

    order = (s.groupby("usage")["value"].median()
             .sort_values(ascending=False).index.tolist())
    groups = [s.loc[s["usage"] == u, "value"].to_numpy() for u in order]

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(order)), 4.5))
    boxplot(ax, groups, order)
    ax.set_ylabel("Electricity (kWh/hour/m2)")
    ax.set_title(f"{NAVN}: area-normalised electricity")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    lagre(fig, filnavn)


def fig_raw_vs_log(elec, filnavn):
    v = elec.stack(future_stack=True).dropna()
    v = v[v > 0].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].boxplot([v], tick_labels=["raw"],
                    flierprops=dict(marker=".", markersize=2, alpha=0.2),
                    medianprops=dict(color="crimson"))
    axes[0].set_title("Linear axis")
    axes[0].set_ylabel("kWh/hour")
    axes[1].boxplot([np.log1p(v)], tick_labels=["log1p"],
                    flierprops=dict(marker=".", markersize=2, alpha=0.2),
                    medianprops=dict(color="crimson"))
    axes[1].set_title("After log(1+x)")
    axes[1].set_ylabel("log(1 + kWh/hour)")
    fig.suptitle(f"{NAVN}: effect of log transform on skewness")
    fig.tight_layout()
    lagre(fig, filnavn)


# ----------------------------------------------------------------------
# Hovedløp
# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Boksplott for en BDG2-site eller ett bygg.")
    ap.add_argument("--site",
                    help="site_id. Utelates den, spør skriptet ved oppstart.")
    ap.add_argument("--bygg",
                    help="building_id, f.eks. Robin_education_Kiera. Kortform "
                         "uten site-prefiks (Kiera) slås opp på --site.")
    ap.add_argument("--liste", action="store_true",
                    help="list bygg på valgt site og avslutt")
    ap.add_argument("--ut", type=Path, default=OUT_ROOT,
                    help=f"rotmappe for figurer (standard: {OUT_ROOT})")
    return ap.parse_args()


def velg_site(sites, forslag=None):
    """Spør etter site til et gyldig navn er skrevet inn.

    Godtar navnet (uavhengig av store og små bokstaver), nummeret i listen,
    eller tom linje for standard. Er forslag (fra --site) gyldig, brukes det
    uten å spørre.
    """
    oppslag = {s.lower(): s for s in sites}
    if forslag:
        if forslag.strip().lower() in oppslag:
            return oppslag[forslag.strip().lower()]
        print(f"Ukjent site '{forslag}'.")

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


def konfigurer(args):
    """Setter globale SITE, BYGG, NAVN og OUT_DIR ut fra argumentene."""
    global SITE, BYGG, NAVN, OUT_DIR
    SITE = args.site
    BYGG = None
    if args.bygg:
        BYGG = args.bygg
        if "_" not in BYGG:                 # kortform: bare kallenavnet
            md = metadata.reset_index()
            treff = md.loc[(md["site_id"] == SITE)
                           & md["building_id"].str.endswith("_" + BYGG),
                           "building_id"].tolist()
            if len(treff) != 1:
                raise SystemExit(f"Fant {len(treff)} bygg som matcher "
                                 f"'{BYGG}' på {SITE}: {treff}")
            BYGG = treff[0]
        SITE = BYGG.split("_")[0]
    NAVN = BYGG or SITE
    OUT_DIR = args.ut / NAVN
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    # Fullt building_id gir siten selv; ellers velges site (spør om nødvendig)
    if not (args.bygg and "_" in args.bygg):
        args.site = velg_site(sorted(metadata["site_id"].dropna().unique()),
                              args.site)
    if args.liste:
        md = metadata.reset_index()
        md = md[md["site_id"] == args.site]
        print(md[["building_id", "primaryspaceusage", "sqm"]]
              .to_string(index=False))
        return
    konfigurer(args)

    meta = load_metadata()
    if BYGG and meta.empty:
        raise SystemExit(f"Bygget {BYGG} finnes ikke i metadata")
    print(f"{NAVN}: {len(meta)} bygg i metadata\n")

    data = {}
    for m in METER_TYPES:
        df = last_maler(m)
        if df is None:
            continue
        data[m] = df
        print(f"  {m:14s} {df.shape[1]:3d} malere")

    if not data:
        print("Fant ingen malerdata for", NAVN)
        return

    long_df = pd.concat([to_long(d, m) for m, d in data.items()],
                        ignore_index=True)

    stat = statistikk(long_df)
    print("\nStatistikk:")
    print(stat.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    stat.to_csv(OUT_DIR / "statistikk.csv", index=False)

    fig_per_meter_type(long_df, "01_per_maalertype.png")

    if "electricity" in data:
        elec = data["electricity"]
        fig_raw_vs_log(elec, "02_lineaer_vs_log.png")
        if not BYGG:                        # gir ingen mening for ett bygg
            fig_per_building(elec, "03_per_bygg.png")
        fig_per_month(elec, "04_per_maaned.png")
        fig_daily_profile(elec, "05_doegnprofil.png")
        fig_normalised(elec, meta, "06_arealnormalisert.png")

    print(f"\nFigurer og tabell skrevet til {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
