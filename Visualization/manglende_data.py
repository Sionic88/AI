"""
Manglende data for varme- og kjølekildene i BDG2.

Ser på to nivåer:
  1. Bygg:  hvor mange av de 1 636 byggene har måleren i det hele tatt
  2. Timer: hvor stor andel av timene har en verdi, for målerne som finnes

Timene deles i tre: verdi over null, null og manglende. Null kan være ekte
(anlegget er av) eller en manglende måling som er lagret som 0, så andelen
nuller tas med. Alle tall er fra rådataene.

Kategorier:
  Heating  hotwater, steam, gas
  Cooling  chilledwater
  Referanse electricity (alle bygg har strøm)

Metadatafeltet heatingtype (oppvarmingskilde) vises for seg.

Bruk:
    python manglende_data.py

Figurer og tabeller havner i .\\figures\\manglende_data\\.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import load_meter, metadata_raw

OUT_DIR = Path("figures").resolve() / "manglende_data"

KATEGORI = {
    "hotwater": "Heating", "steam": "Heating", "gas": "Heating",
    "chilledwater": "Cooling", "electricity": "Reference",
}
NAVN = {"hotwater": "Hot water", "steam": "Steam", "gas": "Gas",
        "chilledwater": "Chilled water", "electricity": "Electricity"}

# Grupper for de mange skrivemåtene i heatingtype
OPPVARMING = {
    "Gas": "Gas", "Gas Boilers": "Gas", "Boiler fed central heating": "Gas",
    "Heat network": "District heating", "District Heating": "District heating",
    "Heat network and steam": "District heating",
    "Heat network but not ours": "District heating",
    "Electricity": "Electric", "Electric": "Electric",
    "Steam": "Steam", "Oil": "Oil", "Biomass": "Biomass",
}

MIN_DEKNING = 0.5

FARGE_VERDI = "#2a78d6"
FARGE_NULL = "#9ec5f4"
FARGE_MANGLER = "#e4e3dc"
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
    "legend.frameon": False,
})


def lagre(fig, navn):
    """Lagrer figuren via en åpen filhandle. Unngår Errno 22 på Windows."""
    try:
        with open(OUT_DIR / navn, "wb") as f:
            fig.savefig(f, format="png", bbox_inches="tight")
        print(f"  saved {navn}")
    except Exception as e:
        print(f"  COULD NOT save {navn}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)


def lagre_csv(df, navn, **kw):
    df.to_csv(OUT_DIR / navn, sep=";", encoding="utf-8-sig", **kw)
    print(f"  saved {navn}")


# ----------------------------------------------------------------------
# Tabeller
# ----------------------------------------------------------------------
def kategoritabell(md, raa):
    """Én rad per målertype med dekning på bygg- og timenivå."""
    n_bygg = len(md)
    rader = {}
    for m, df in raa.items():
        v = df.to_numpy()
        har = ~np.isnan(v)
        dekning = df.notna().mean()
        rader[NAVN[m]] = {
            "Category": KATEGORI[m],
            "Buildings with meter": df.shape[1],
            "Share of all buildings (%)": 100 * df.shape[1] / n_bygg,
            "Sites": df.columns.str.split("_").str[0].nunique(),
            "Hours with value (%)": 100 * har.mean(),
            "Hours missing (%)": 100 * (1 - har.mean()),
            "Zeros among values (%)": 100 * (v[har] == 0).mean(),
            f"Meters with >= {MIN_DEKNING:.0%} coverage": int((dekning >= MIN_DEKNING).sum()),
            "Meters with >= 90% coverage": int((dekning >= 0.9).sum()),
            "Meters with no data at all": int((dekning == 0).sum()),
        }
    tab = pd.DataFrame(rader).T
    # Bygg med minst én varmemåler / kjølemåler
    varme = set().union(*[set(raa[m].columns) for m in raa if KATEGORI[m] == "Heating"])
    kjol = set(raa["chilledwater"].columns)
    sammen = {
        "Buildings with any heating meter": len(varme),
        "Buildings with a cooling meter": len(kjol),
        "Buildings with both": len(varme & kjol),
        "Buildings with neither": n_bygg - len(varme | kjol),
    }
    return tab, sammen


def sitetabell(raa):
    """Site x målertype: andel timer med verdi, og antall målere i parentes."""
    rader = {}
    for m, df in raa.items():
        site = df.columns.str.split("_").str[0]
        for s in sorted(site.unique()):
            sub = df.loc[:, site == s]
            rader.setdefault(s, {})[NAVN[m]] = (100 * sub.notna().to_numpy().mean(),
                                                 sub.shape[1])
    pst = pd.DataFrame({s: {k: v[0] for k, v in d.items()} for s, d in rader.items()}).T
    antall = pd.DataFrame({s: {k: v[1] for k, v in d.items()} for s, d in rader.items()}).T
    kol = [NAVN[m] for m in raa]
    return pst.reindex(columns=kol), antall.reindex(columns=kol)


def oppvarmingstabell(md):
    gruppe = md["heatingtype"].map(OPPVARMING)
    tab = pd.DataFrame({
        "Buildings": gruppe.value_counts(),
        "Share of buildings with heatingtype (%)":
            100 * gruppe.value_counts() / gruppe.notna().sum(),
    })
    tab.loc["Not recorded"] = [md["heatingtype"].isna().sum(), np.nan]
    land = md.loc[md["heatingtype"].notna(), "site_id"].value_counts()
    return tab, land


# ----------------------------------------------------------------------
# Figurer
# ----------------------------------------------------------------------
def fig_oversikt(tab, raa):
    """Venstre: andel av alle bygg som har måleren. Høyre: timene for de
    målerne som finnes, delt i verdi over null, null og manglende."""
    rekkefolge = [NAVN[m] for m in raa]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    y = np.arange(len(rekkefolge))[::-1]

    andel = tab.loc[rekkefolge, "Share of all buildings (%)"].astype(float)
    a1.barh(y, andel, color=FARGE_VERDI, height=0.6)
    for yi, v, n in zip(y, andel, tab.loc[rekkefolge, "Buildings with meter"]):
        a1.text(v + 1.5, yi, f"{v:.0f} % ({n:,} buildings)", va="center",
                fontsize=8, color=TEKST)
    a1.set_xlim(0, 125)
    a1.set_xlabel("Share of all 1,636 buildings (%)")
    a1.set_yticks(y, [f"{r}  ({tab.loc[r, 'Category']})" for r in rekkefolge])
    a1.set_title("Building level: which buildings have the meter", loc="left")
    a1.grid(axis="y", visible=False)

    for yi, m in zip(y, raa):
        v = raa[m].to_numpy()
        har = ~np.isnan(v)
        deler = [100 * (har & (v > 0)).mean(), 100 * (har & (v == 0)).mean(),
                 100 * (~har).mean()]
        venstre = 0
        for d, farge in zip(deler, [FARGE_VERDI, FARGE_NULL, FARGE_MANGLER]):
            a2.barh(yi, d, left=venstre, color=farge, height=0.6,
                    edgecolor="white", linewidth=1.5)
            if d >= 4:
                a2.text(venstre + d / 2, yi, f"{d:.0f} %", ha="center",
                        va="center", fontsize=8,
                        color="white" if farge == FARGE_VERDI else TEKST)
            venstre += d
    a2.set_xlim(0, 100)
    a2.set_xlabel("Share of hours for the meters that exist (%)")
    a2.set_title("Hour level: value, zero or missing", loc="left")
    a2.grid(axis="y", visible=False)
    a2.legend(handles=[
        matplotlib.patches.Patch(color=FARGE_VERDI, label="value above zero"),
        matplotlib.patches.Patch(color=FARGE_NULL, label="zero"),
        matplotlib.patches.Patch(color=FARGE_MANGLER, label="missing"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    fig.suptitle("Availability of heating and cooling data in BDG2, 2016–2017",
                 x=0.01, ha="left", color=TEKST, fontsize=11)
    fig.tight_layout()
    lagre(fig, "01_varme_kjoling_oversikt.png")


def fig_site(pst, antall):
    """Site x målertype: andel timer med verdi. Tom celle = ingen måler."""
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pst) + 1.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dek", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#104281"])
    cmap.set_bad("#ffffff")
    im = ax.imshow(np.ma.masked_invalid(pst.to_numpy(dtype=float)),
                   aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)
    for i in range(pst.shape[0]):
        for j in range(pst.shape[1]):
            v = pst.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f} %\n(n={int(antall.iat[i, j])})",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if v > 55 else TEKST)
    ax.set_xticks(range(pst.shape[1]), pst.columns)
    ax.set_yticks(range(pst.shape[0]), pst.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of hours with a value (%)")
    cb.outline.set_visible(False)
    ax.set_title("Hours with a value per site and meter type.\n"
                 "n = number of meters. Empty cell = no meter of this type.",
                 loc="left")
    lagre(fig, "02_dekning_per_site.png")


# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = metadata_raw.set_index("building_id")
    maalere = ["electricity", "hotwater", "steam", "gas", "chilledwater"]
    print("Loading meters ...")
    raa = {m: load_meter(m) for m in maalere}

    tab, sammen = kategoritabell(md, raa)
    pst, antall = sitetabell(raa)
    opp, opp_site = oppvarmingstabell(md)

    pd.set_option("display.width", 200)
    print("\nPer meter type:")
    print(tab.to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nHeating and cooling per building:")
    for k, v in sammen.items():
        print(f"  {k:34s} {v:5d}  ({100 * v / len(md):.1f} %)")
    print("\nHeating source from metadata (heatingtype):")
    print(opp.to_string(float_format=lambda x: f"{x:.1f}"))
    print("  recorded on sites:", opp_site.to_dict())

    print("\nFigures and tables:")
    lagre_csv(tab, "dekning_per_maalertype.csv")
    lagre_csv(pst.round(1), "dekning_per_site.csv")
    lagre_csv(opp, "oppvarmingskilde_metadata.csv")
    per_bygg = pd.DataFrame({"Buildings": pd.Series(sammen)})
    per_bygg["Share of all buildings (%)"] = 100 * per_bygg["Buildings"] / len(md)
    lagre_csv(per_bygg, "varme_kjoling_per_bygg.csv")
    fig_oversikt(tab, raa)
    fig_site(pst, antall)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
