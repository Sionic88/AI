"""
Retningsavhengig vindfølsomhet per bygg, site "Robin" (UCL) i BDG2.

Ideen: fjern først alt forbruk som kan forklares av temperatur, vindstyrke,
tid på døgnet, ukedag og måned. Det som blir igjen (residualet) grupperes
etter vindretning. Bygg med stort utslag mellom retninger er kandidater
for befaring.

Viktig: alle bygg på siten deler samme værserie, så en felles retningseffekt
over hele porteføljen kan ikke skilles fra at enkelte vindretninger bringer
kaldere luft. Skriptet rapporterer derfor BÅDE rå sektoreffekt og avvik fra
porteføljens gjennomsnitt. Det siste er det som peker på enkeltbygg.

Bruk:
    python vindretning_analyse.py

Krever kun numpy, pandas og matplotlib.
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

N_SEKTORER = 8                 # 8 x 45 grader gir nok dager per sektor
BASISTEMP_VARME = 15.5         # °C, grensetemperatur for oppvarmingsbehov
BASISTEMP_KJOL = 18.0          # °C
KUN_FYRINGSSESONG = True       # kun døgn med snittemperatur under 12 °C
MIN_DAGER_PER_SEKTOR = 15      # sektorer med færre døgn rapporteres ikke
MIN_TIMER_PER_BYGG = 24 * 180  # bygg med mindre data hoppes over
TOPP_N = 6                     # antall bygg i figuren

SEKTORNAVN_8 = ["N", "NØ", "Ø", "SØ", "S", "SV", "V", "NV"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def lagre(fig, navn):
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
def last_vaer():
    w = pd.read_csv(DATA_DIR / "weather" / "weather.csv",
                    parse_dates=["timestamp"])
    w = w[w["site_id"] == SITE].set_index("timestamp").sort_index()
    return w[["airTemperature", "windSpeed", "windDirection"]]


def last_elektrisitet():
    e = pd.read_csv(DATA_DIR / "meters" / "raw" / "electricity.csv",
                    parse_dates=["timestamp"], index_col="timestamp")
    cols = [c for c in e.columns if c.startswith(SITE + "_")]
    return e[cols]


def last_metadata():
    md = pd.read_csv(DATA_DIR / "metadata" / "metadata.csv")
    return md[md["site_id"] == SITE].set_index("building_id")


# ----------------------------------------------------------------------
# Modell
# ----------------------------------------------------------------------
def designmatrise(w):
    """Bygger forklaringsvariablene som skal ut av forbruket først.

    Inneholder:
      konstantledd
      graddifferanse for oppvarming og kjøling
      vindstyrke, og vindstyrke ganget med oppvarmingsbehov (infiltrasjon)
      dummy for hver time på døgnet, delt på ukedag og helg
      dummy for hver måned
    """
    dT_varme = np.clip(BASISTEMP_VARME - w["airTemperature"], 0, None)
    dT_kjol = np.clip(w["airTemperature"] - BASISTEMP_KJOL, 0, None)
    vind = w["windSpeed"]

    deler = [
        pd.Series(1.0, index=w.index, name="konst"),
        dT_varme.rename("dT_varme"),
        dT_kjol.rename("dT_kjol"),
        vind.rename("vind"),
        (dT_varme * vind).rename("dT_varme_x_vind"),
    ]

    helg = (w.index.dayofweek >= 5).astype(int)
    time = w.index.hour
    for h in range(1, 24):                       # time 0 er referanse
        deler.append(pd.Series(((time == h) & (helg == 0)).astype(float),
                               index=w.index, name=f"uke_t{h}"))
    for h in range(24):
        deler.append(pd.Series(((time == h) & (helg == 1)).astype(float),
                               index=w.index, name=f"helg_t{h}"))
    for m in range(2, 13):                       # januar er referanse
        deler.append(pd.Series((w.index.month == m).astype(float),
                               index=w.index, name=f"mnd{m}"))

    return pd.concat(deler, axis=1)


def residualer(y, X):
    """Minste kvadraters metode med numpy. Returnerer residual som
    andel av byggets gjennomsnittsforbruk."""
    m = y.notna() & X.notna().all(axis=1)
    if m.sum() < MIN_TIMER_PER_BYGG:
        return None, np.nan
    Xm = X[m].to_numpy(dtype=float)
    ym = y[m].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    pred = Xm @ beta
    snitt = ym.mean()
    if snitt <= 0:
        return None, np.nan

    ss_tot = ((ym - snitt) ** 2).sum()
    r2 = 1 - ((ym - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

    res = pd.Series(100 * (ym - pred) / snitt, index=y.index[m])
    return res, r2


# ----------------------------------------------------------------------
# Retning per døgn
# ----------------------------------------------------------------------
def dognretning(w):
    """Vektormiddel av vindretningen per døgn, vektet med vindstyrke.

    Returnerer en tabell med dato, sektor, døgnmiddeltemperatur og
    døgnmiddel vindstyrke. Aggregering til døgn er bevisst: timesverdier
    er sterkt autokorrelerte, og døgn gir ærligere usikkerhet.
    """
    d = w.dropna(subset=["windDirection", "windSpeed"]).copy()
    d = d[d["windSpeed"] > 0]
    rad = np.deg2rad(d["windDirection"] % 360)
    d["u"] = d["windSpeed"] * np.sin(rad)
    d["v"] = d["windSpeed"] * np.cos(rad)

    g = d.groupby(d.index.date)
    ut = pd.DataFrame({
        "u": g["u"].mean(),
        "v": g["v"].mean(),
        "vind": g["windSpeed"].mean(),
        "timer": g["windSpeed"].size(),
    })
    ut = ut[ut["timer"] >= 18]                   # krev et rimelig komplett døgn

    retning = (np.degrees(np.arctan2(ut["u"], ut["v"])) + 360) % 360
    bredde = 360 / N_SEKTORER
    ut["sektor"] = np.floor(((retning + bredde / 2) % 360) / bredde).astype(int)
    ut["retning"] = retning

    temp = w["airTemperature"].groupby(w.index.date).mean()
    ut["temp"] = temp.reindex(ut.index)
    ut.index = pd.to_datetime(ut.index)
    return ut


# ----------------------------------------------------------------------
# Figurer
# ----------------------------------------------------------------------
def fig_datagrunnlag(n_dager, r2):
    """Hvor mange døgn ligger bak hver sektor, og hvor godt treffer modellen."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    farger = ["#2166ac" if n >= MIN_DAGER_PER_SEKTOR else "#cccccc"
              for n in n_dager]
    ax.bar(n_dager.index, n_dager.to_numpy(), color=farger)
    ax.axhline(MIN_DAGER_PER_SEKTOR, color="crimson", ls="--", lw=1,
               label=f"terskel {MIN_DAGER_PER_SEKTOR} døgn")
    ax.set_ylabel("Antall døgn")
    ax.set_title("Datagrunnlag per vindsektor")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(n_dager.to_numpy()):
        ax.text(i, n, str(int(n)), ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    ax.hist(r2.dropna().to_numpy(), bins=20, color="#2166ac",
            edgecolor="white")
    ax.axvline(r2.median(), color="crimson", lw=1.5,
               label=f"median {r2.median():.2f}")
    ax.set_xlabel("Modellens R2 per bygg")
    ax.set_ylabel("Antall bygg")
    ax.set_title("Hvor mye basismodellen forklarer")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    lagre(fig, "16_datagrunnlag.png")


def fig_heatmap(avvik, spenn):
    """Alle bygg mot alle sektorer. Rødt = merforbruk, blått = mindre."""
    d = avvik.loc[spenn.index]
    if d.empty:
        return
    grense = np.nanmax(np.abs(d.to_numpy())) or 1.0

    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(d) + 1.5)))
    bilde = ax.imshow(d.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto",
                      vmin=-grense, vmax=grense)
    ax.set_xticks(range(d.shape[1]))
    ax.set_xticklabels(d.columns)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([b.replace(SITE + "_", "") for b in d.index],
                       fontsize=6)
    ax.set_title(f"{SITE}: merforbruk per vindretning\n"
                 "avvik fra porteføljen (%), sortert etter utslag")
    fig.colorbar(bilde, ax=ax, label="% av byggets snittforbruk", shrink=0.7)
    fig.tight_layout()
    lagre(fig, "17_retning_heatmap.png")


def fig_rangering(ut, topp=20):
    """Befaringslisten som stolpediagram."""
    d = ut.head(topp).iloc[::-1]
    if d.empty:
        return
    navn = [f"{b.replace(SITE + '_', '')}  [{s}]"
            for b, s in zip(d.index, d["verste_sektor"])]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(d) + 1.5)))
    ax.barh(range(len(d)), d["spenn_%"].to_numpy(dtype=float),
            color="#b2182b")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(navn, fontsize=7)
    ax.set_xlabel("Forskjell mellom beste og verste vindretning (%)")
    ax.set_title(f"{SITE}: kandidater for befaring\n"
                 "verste retning i klammer")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(d["spenn_%"].to_numpy(dtype=float)):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=7)
    fig.tight_layout()
    lagre(fig, "18_retning_rangering.png")


# ----------------------------------------------------------------------
# Hovedløp
# ----------------------------------------------------------------------
def main():
    print(f"{SITE}: retningsavhengig vindfølsomhet\n")
    w = last_vaer()
    e = last_elektrisitet()
    meta = last_metadata()
    print(f"  {e.shape[1]} bygg med elektrisitetsmåler")

    dogn = dognretning(w)
    if KUN_FYRINGSSESONG:
        dogn = dogn[dogn["temp"] < 12]
    print(f"  {len(dogn)} døgn i analysen"
          f"{' (kun fyringssesong)' if KUN_FYRINGSSESONG else ''}")

    X = designmatrise(w).reindex(e.index)   # koble vær og forbruk på tidsstempel

    sektor_tab, r2_tab = {}, {}
    for bygg in e.columns:
        res, r2 = residualer(e[bygg], X)
        if res is None:
            continue
        r2_tab[bygg] = r2
        dagsres = res.groupby(res.index.date).mean()
        dagsres.index = pd.to_datetime(dagsres.index)
        df = dogn.join(dagsres.rename("res"), how="inner").dropna(subset=["res"])

        rad = {}
        for s in range(N_SEKTORER):
            v = df.loc[df["sektor"] == s, "res"]
            rad[SEKTORNAVN_8[s]] = v.mean() if len(v) >= MIN_DAGER_PER_SEKTOR \
                else np.nan
        sektor_tab[bygg] = rad

    tab = pd.DataFrame(sektor_tab).T
    n_dager = dogn["sektor"].value_counts().reindex(range(N_SEKTORER),
                                                    fill_value=0)
    n_dager.index = SEKTORNAVN_8

    print("\nAntall døgn per sektor:")
    print(n_dager.to_string())

    print("\nPorteføljens gjennomsnittlige sektoreffekt "
          "(% avvik fra forventet forbruk):")
    portefolje = tab.mean()
    print(portefolje.to_string(float_format=lambda x: f"{x:6.2f}"))
    print("\n  Denne raden er IKKE bevis for fasadeeffekt. Den kan like gjerne")
    print("  skyldes at enkelte vindretninger bringer luft med andre egenskaper")
    print("  enn temperaturen alene fanger opp.")

    avvik = tab.subtract(portefolje, axis=1)
    spenn = (avvik.max(axis=1) - avvik.min(axis=1)).sort_values(ascending=False)

    ut = pd.DataFrame({
        "spenn_%": spenn,
        "verste_sektor": avvik.idxmax(axis=1),
        "beste_sektor": avvik.idxmin(axis=1),
        "modell_r2": pd.Series(r2_tab),
        "sqm": meta["sqm"].reindex(spenn.index),
        "bruk": meta["primaryspaceusage"].reindex(spenn.index),
    }).loc[spenn.index]

    print("\nRangering, bygg med størst retningsforskjell "
          "målt som avvik fra porteføljen:")
    print(ut.head(15).to_string(float_format=lambda x: f"{x:,.2f}"))

    # figurer i stedet for tabellfiler
    fig_datagrunnlag(n_dager, pd.Series(r2_tab))
    fig_heatmap(avvik, spenn)
    fig_rangering(ut)

    # figur: avvik per sektor for de mest utslagsgivende byggene
    topp = [b for b in spenn.index[:TOPP_N]]
    n = len(topp)
    if n:
        fig = plt.figure(figsize=(3.4 * min(n, 3), 4.0 * int(np.ceil(n / 3))))
        vinkler = np.deg2rad(np.arange(0, 360, 360 / N_SEKTORER))
        for i, bygg in enumerate(topp, start=1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), min(n, 3), i,
                                 projection="polar")
            verdier = avvik.loc[bygg].to_numpy(dtype=float)
            farger = ["#b2182b" if v > 0 else "#2166ac" for v in
                      np.nan_to_num(verdier)]
            ax.bar(vinkler, np.nan_to_num(verdier),
                   width=np.deg2rad(360 / N_SEKTORER) * 0.9,
                   color=farger, edgecolor="white", linewidth=0.5)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(vinkler)
            ax.set_xticklabels(SEKTORNAVN_8, fontsize=7)
            ax.set_title(bygg.replace(SITE + "_", ""), fontsize=8, pad=14)
            ax.tick_params(labelsize=6)
            # nullringen er referansen: innenfor = mindre, utenfor = mer
            ax.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200),
                    color="black", lw=0.8)
        fig.suptitle(f"{SITE}: merforbruk per vindretning, avvik fra "
                     f"porteføljen (%)\nrødt = mer enn forventet, "
                     f"blått = mindre", fontsize=10)
        fig.tight_layout(h_pad=2.5)
        lagre(fig, "15_retningsavvik_per_bygg.png")

    print(f"\nSkrevet til {OUT_DIR}")


if __name__ == "__main__":
    main()
