"""
Datakompletthet for samtlige bygg i Building Data Genome Project 2.

Lager et stripediagram med én rad per bygg og én kolonne per datafelt
(metadata og målere). Feltene ordnes grådig:

  1. byggeår (yearbuilt) først
  2. deretter feltet som flest av de gjenværende byggene har
  3. osv., til alle felt er brukt

Byggene sorteres i samme rekkefølge, slik at de mest komplette byggene
havner øverst. En trakt over diagrammet viser hvor mange bygg som har
alle felt fram til og med hver kolonne. Byggene som overlever lengst i
trakten er "idealbyggene".

En måler regnes som tilstede når den har verdi i minst MIN_DEKNING av timene.

Bruk:
    python datakompletthet.py

Figur og tabeller havner i .\\figures\\datakompletthet\\.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import METER_TYPES, load_meter, metadata_raw

OUT_DIR = Path("figures").resolve() / "datakompletthet"

FORSTE_FELT = "yearbuilt"
MIN_DEKNING = 0.5          # andel timer med verdi før en måler teller som data

METADATA_FELT = ["yearbuilt", "sqm", "primaryspaceusage", "sub_primaryspaceusage",
                 "lat", "heatingtype", "date_opened",
                 "numberoffloors", "occupants", "energystarscore", "eui",
                 "site_eui", "source_eui", "leed_level"]
# Tatt ut fordi de er lite relevante og nesten bare fylt ut for britiske bygg,
# slik at trakten ellers bare sitter igjen med bygg fra Storbritannia:
#   "industry", "subindustry", "rating"

FARGE_DATA = "#2a78d6"
FARGE_MANGLER = "#e4e3dc"
TEKST = "#1a1a19"
TEKST_SEKUNDAER = "#5f5e5a"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9,
    "axes.edgecolor": FARGE_MANGLER, "axes.labelcolor": TEKST_SEKUNDAER,
    "axes.titlecolor": TEKST, "xtick.color": TEKST_SEKUNDAER,
    "ytick.color": TEKST_SEKUNDAER,
    "axes.spines.top": False, "axes.spines.right": False,
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


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def tilstede_tabell():
    """Bygg x felt, True der bygget har data i feltet."""
    # rå metadata: analysen handler nettopp om hvilke felt som mangler
    md = metadata_raw.set_index("building_id")
    tab = md[METADATA_FELT].notna()

    for m in METER_TYPES:
        df = load_meter(m)
        if df is None:
            continue
        dekning = df.notna().mean()
        tab[m] = (dekning.reindex(tab.index).fillna(0) >= MIN_DEKNING)
        print(f"  {m:13s} {int(tab[m].sum()):5d} bygg med minst "
              f"{MIN_DEKNING:.0%} dekning")
    return tab, md


def gradig_rekkefolge(tab):
    """Ordner feltene: FORSTE_FELT, så det som flest gjenværende bygg har.

    Når ingen bygg er igjen, ordnes resten etter total dekning.
    Returnerer rekkefølgen og antall bygg igjen etter hvert felt.
    """
    rekkefolge = [FORSTE_FELT]
    igjen = tab[FORSTE_FELT]
    antall = [int(igjen.sum())]
    rest = [f for f in tab.columns if f != FORSTE_FELT]

    while rest:
        if igjen.any():
            treff = tab.loc[igjen, rest].sum()
            # uavgjort brytes på total dekning i hele datasettet
            neste = max(rest, key=lambda f: (treff[f], tab[f].sum()))
        else:
            neste = max(rest, key=lambda f: tab[f].sum())
        rekkefolge.append(neste)
        rest.remove(neste)
        igjen = igjen & tab[neste]
        antall.append(int(igjen.sum()))
    return rekkefolge, antall


def sorter_bygg(tab, rekkefolge):
    """Leksikografisk sortering: data i første felt først, så neste, osv.

    Innenfor like mønstre kommer bygg med flest felt totalt først.
    """
    nokler = [-tab[f].to_numpy(int) for f in reversed(rekkefolge)]
    nokler = [-tab.sum(axis=1).to_numpy()] + nokler    # minst viktige nøkkel
    idx = np.lexsort(nokler)
    return tab.iloc[idx][rekkefolge]


# ----------------------------------------------------------------------
# Figur
# ----------------------------------------------------------------------
def figur(sortert, antall):
    felt = list(sortert.columns)
    n_bygg, n_felt = sortert.shape
    x = np.arange(n_felt)

    fig, (ax_t, ax_m) = plt.subplots(
        2, 1, figsize=(12, 15), sharex=True,
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.04})

    # Trakt: bygg som har alle felt fram til og med denne kolonnen
    ax_t.bar(x, antall, width=0.8, color=FARGE_DATA)
    for xi, a in zip(x, antall):
        ax_t.text(xi, a, f"{a}", ha="center", va="bottom", fontsize=7,
                  color=TEKST)
    ax_t.set_ylabel("Buildings with all fields\nup to this column")
    ax_t.set_ylim(0, n_bygg * 1.12)
    ax_t.grid(axis="y", color=FARGE_MANGLER, lw=0.6)
    ax_t.set_axisbelow(True)
    ax_t.set_title(f"Data completeness for all {n_bygg} buildings in BDG2: "
                   f"fields ordered by year built, then by most remaining buildings",
                   fontsize=11)

    # Stripediagram
    farger = matplotlib.colors.ListedColormap([FARGE_MANGLER, FARGE_DATA])
    ax_m.imshow(sortert.to_numpy(int), aspect="auto", cmap=farger,
                interpolation="nearest", vmin=0, vmax=1)
    ax_m.set_xticks(x, felt, rotation=60, ha="right")
    ax_m.set_ylabel(f"Buildings (sorted, {n_bygg} rows)")
    ax_m.set_yticks([0, n_bygg - 1], ["most complete", "least complete"])

    # Skille mellom metadata og målere i kolonneetikettene
    for lbl in ax_m.get_xticklabels():
        if lbl.get_text() in METER_TYPES:
            lbl.set_fontweight("bold")

    ax_m.legend(handles=[
        matplotlib.patches.Patch(color=FARGE_DATA, label="has data"),
        matplotlib.patches.Patch(color=FARGE_MANGLER, label="missing"),
    ], loc="lower right", frameon=True, fontsize=8)
    ax_m.text(0, -0.13, f"Bold = meter (counts at least "
              f"{MIN_DEKNING:.0%} hourly coverage). Regular = metadata.",
              transform=ax_m.transAxes, fontsize=8, color=TEKST_SEKUNDAER,
              va="top")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.1)
    lagre(fig, "datakompletthet.png")


LAND = {"Lamb": "United Kingdom", "Mouse": "United Kingdom",
        "Robin": "United Kingdom", "Shrew": "United Kingdom",
        "Crow": "Canada", "Moose": "Canada", "Wolf": "Ireland"}  # resten: USA
KATEGORI_FARGER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FARGE_ANDRE = "#a8a79f"


def fig_fordeling(tab, rekkefolge, antall, md):
    """Hvordan byggene som er igjen fordeler seg på land og bruk, steg for steg."""
    steg = [i for i, a in enumerate(antall) if a > 0]
    land = md["site_id"].map(LAND).fillna("USA")
    topp_bruk = md["primaryspaceusage"].value_counts().index[:4].tolist()
    bruk = md["primaryspaceusage"].where(
        md["primaryspaceusage"].isin(topp_bruk), "Other")

    paneler = [
        (land, ["USA", "United Kingdom", "Canada", "Ireland"], "country"),
        (bruk, topp_bruk + ["Other"], "use (primaryspaceusage)"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(steg))
    for ax, (gruppe, kategorier, tittel) in zip(axes, paneler):
        igjen = pd.Series(True, index=tab.index)
        tellinger = []
        for i in steg:
            igjen = igjen & tab[rekkefolge[i]]
            tellinger.append(gruppe[igjen[igjen].index].value_counts())
        t = pd.DataFrame(tellinger).reindex(columns=kategorier).fillna(0)

        bunn = np.zeros(len(steg))
        for k, kat in enumerate(kategorier):
            farge = FARGE_ANDRE if kat == "Other" else KATEGORI_FARGER[k]
            ax.bar(x, t[kat], bottom=bunn, width=0.75, color=farge,
                   edgecolor="white", linewidth=1, label=kat)
            bunn += t[kat].to_numpy()
        for xi, total in zip(x, bunn):
            ax.text(xi, total, f"{int(total)}", ha="center", va="bottom",
                    fontsize=8, color=TEKST)
        ax.set_ylabel("Buildings remaining")
        ax.set_ylim(0, bunn.max() * 1.1)
        ax.set_title(f"Distribution by {tittel}", loc="left", fontsize=10)
        ax.grid(axis="y", color=FARGE_MANGLER, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].set_xticks(x, [("+ " if i else "") + rekkefolge[i] for i in steg],
                        rotation=30, ha="right")
    axes[-1].set_xlabel("Requirement added, step by step (each step requires "
                        "all fields to the left)")
    fig.suptitle("Who survives the funnel? Buildings with all fields "
                 "up to each step", color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "fordeling.png")


def fig_site_dekning(tab, rekkefolge, md):
    """Site x felt: andel av sitens bygg som har feltet.

    Viser hvilke sites som dekker hvilke aspekter. Sitene er gruppert på land
    og sortert etter hvor mye de dekker totalt. Returnerer tabellen i prosent.
    """
    site = md["site_id"].reindex(tab.index)
    andel = 100 * tab[rekkefolge].groupby(site).mean()
    n = site.value_counts()
    land = andel.index.map(lambda s: LAND.get(s, "USA"))
    orden = (pd.DataFrame({"land": land, "snitt": andel.mean(axis=1).to_numpy()},
                          index=andel.index)
             .assign(land_nr=lambda d: d["land"].map(
                 {"USA": 0, "Canada": 1, "United Kingdom": 2, "Ireland": 3}))
             .sort_values(["land_nr", "snitt"], ascending=[True, False]))
    andel = andel.loc[orden.index]

    n_site, n_felt = andel.shape
    fig, ax = plt.subplots(figsize=(13, 0.42 * n_site + 2.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dekning", ["#f4f3ee", "#9ec5f4", "#3987e5", "#1c5cab", "#104281"])
    im = ax.imshow(andel.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                   vmax=100, interpolation="nearest")

    # Prosent i cellene. Tom celle = ingen bygg på siten har feltet.
    for i in range(n_site):
        for j in range(n_felt):
            v = andel.iat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 55 else TEKST)

    # Skillelinjer mellom land
    grenser = np.flatnonzero(orden["land"].to_numpy()[1:]
                             != orden["land"].to_numpy()[:-1])
    for g in grenser:
        ax.axhline(g + 0.5, color=TEKST, linewidth=1.2)

    ax.set_xticks(range(n_felt), rekkefolge, rotation=60, ha="right")
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in METER_TYPES:
            lbl.set_fontweight("bold")
    ax.set_yticks(range(n_site),
                  [f"{s}  ({LAND.get(s, 'USA')}, n={n[s]})" for s in andel.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Share of the site's buildings with data (%)")
    cb.outline.set_visible(False)
    ax.set_title("Which sites cover which fields? Share of buildings per site "
                 "with data in each field", loc="left", fontsize=11, color=TEKST)
    ax.text(0, -0.25, f"Empty cell = no building on the site has the field. "
            f"Bold = meter (at least {MIN_DEKNING:.0%} hourly coverage). "
            "Columns in the same order as datakompletthet.png.",
            transform=ax.transAxes, fontsize=8, color=TEKST_SEKUNDAER, va="top")
    fig.subplots_adjust(left=0.2, right=0.9, top=0.94, bottom=0.2)
    lagre(fig, "site_dekning.png")
    return andel


# ----------------------------------------------------------------------
# Hovedløp
# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Leser metadata og målerdekning:")
    tab, md = tilstede_tabell()

    rekkefolge, antall = gradig_rekkefolge(tab)
    sortert = sorter_bygg(tab, rekkefolge)

    trakt = pd.DataFrame({"felt": rekkefolge, "bygg_igjen": antall,
                          "dekning_totalt": [int(tab[f].sum())
                                             for f in rekkefolge]})
    print("\nTrakt (bygg med alle felt t.o.m. raden):")
    print(trakt.to_string(index=False))
    trakt.to_csv(OUT_DIR / "trakt.csv", index=False, sep=";",
                 encoding="utf-8-sig")

    # Idealbygg: de som har lengst sammenhengende rekke fra venstre
    lengde = sortert.cumprod(axis=1).sum(axis=1)
    rangering = pd.DataFrame({
        "site_id": md["site_id"].reindex(sortert.index),
        "bruk": md["sub_primaryspaceusage"].reindex(sortert.index),
        "felt_i_rekke": lengde,
        "felt_totalt": sortert.sum(axis=1),
        "mangler": sortert.apply(lambda r: ", ".join(r.index[~r]), axis=1),
    })
    rangering.index.name = "building_id"
    rangering.to_csv(OUT_DIR / "bygg_rangert.csv", sep=";",
                     encoding="utf-8-sig")

    beste = lengde.max()
    ideal = rangering[rangering["felt_i_rekke"] == beste]
    print(f"\nIdealbygg: {len(ideal)} bygg har de {beste} første feltene "
          f"({', '.join(rekkefolge[:beste])})")
    print(ideal.groupby(["site_id", "bruk"]).size()
          .rename("antall").to_string())
    print("Full liste i bygg_rangert.csv")

    figur(sortert, antall)
    fig_fordeling(tab, rekkefolge, antall, md)
    site_andel = fig_site_dekning(tab, rekkefolge, md)
    site_andel.round(1).to_csv(OUT_DIR / "site_dekning.csv", sep=";",
                               encoding="utf-8-sig")
    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
