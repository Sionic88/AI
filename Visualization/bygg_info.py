"""
Skriver ut alt datasettet inneholder om ett enkelt bygg.

Bruk:
    python bygg_info.py              # spør "Skriv inn bygg" ved oppstart
    python bygg_info.py Lashandra
    python bygg_info.py Robin_Education_Lashandra

Søket er delvis og ufølsomt for store og små bokstaver.

Utskriften lagres også som tekstfil, én per bygg, i
Visualization\\figures\\bygg_info\\<building_id>.txt.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pandas_data import METER_TYPES, load_meter, metadata_raw

# Fast plassering ved siden av skriptet, uansett hvilken mappe det kjøres fra
UT_DIR = Path(__file__).resolve().parent / "figures" / "bygg_info"


class Tee:
    """Skriver til flere strømmer samtidig (konsoll og tekstfil)."""

    def __init__(self, *strommer):
        self.strommer = strommer

    def write(self, tekst):
        for s in self.strommer:
            s.write(tekst)

    def flush(self):
        for s in self.strommer:
            s.flush()


def main():
    if len(sys.argv) > 1:
        navn = sys.argv[1]
    else:
        navn = input("Skriv inn bygg (f.eks. Lashandra): ").strip()
        if not navn:
            return
    sok = navn.lower()

    md = metadata_raw     # alle felt og alle bygg, også de uten areal
    treff = md[md["building_id"].str.lower().str.contains(sok, na=False)]

    if treff.empty:
        print(f"Fant ingen bygg som matcher '{navn}'.")
        like = md[md["building_id"].str.lower().str.contains(
            sok[:5], na=False)]["building_id"].head(10).tolist()
        if like:
            print("Nærmeste navn:", ", ".join(like))
        return
    if len(treff) > 1:
        print("Flere treff:", ", ".join(treff["building_id"]))
        print()

    UT_DIR.mkdir(parents=True, exist_ok=True)
    for _, rad in treff.iterrows():
        buffer = io.StringIO()
        with redirect_stdout(Tee(sys.stdout, buffer)):
            vis_bygg(rad)
        fil = UT_DIR / f"{rad['building_id']}.txt"
        fil.write_text(buffer.getvalue(), encoding="utf-8")
        print(f"Lagret til {fil}\n")


def vis_bygg(rad):
    """Skriver metadata og målerstatistikk for ett bygg."""
    bygg = rad["building_id"]
    print("=" * 70)
    print(bygg)
    print("=" * 70)

    print("\nMetadata (kun felter med verdi):")
    for felt, verdi in rad.items():
        if pd.notna(verdi) and str(verdi).strip() != "":
            print(f"  {felt:24s} {verdi}")

    tomme = [f for f, v in rad.items() if pd.isna(v)]
    if tomme:
        print(f"\n  Felter uten verdi ({len(tomme)}): {', '.join(tomme)}")

    print("\nMålerdata (rå):")
    funnet = False
    for m in METER_TYPES:
        df = load_meter(m, buildings=[bygg])
        if df is None or bygg not in df.columns:
            continue
        funnet = True
        s = df[bygg]
        g = s.dropna()
        print(f"\n  {m}")
        print(f"    dekning      {100 * len(g) / len(s):6.2f} %"
              f"  ({len(g):,} av {len(s):,} timer)")
        if g.empty:
            continue
        print(f"    min          {g.min():12,.2f}")
        print(f"    Q1           {g.quantile(0.25):12,.2f}")
        print(f"    median       {g.median():12,.2f}")
        print(f"    Q3           {g.quantile(0.75):12,.2f}")
        print(f"    maks         {g.max():12,.2f}")
        print(f"    gjennomsnitt {g.mean():12,.2f}")
        print(f"    std          {g.std():12,.2f}")
        print(f"    skjevhet     {g.skew():12,.2f}")
        print(f"    sum per år   {g.sum() / 2:12,.0f}")
        if pd.notna(rad.get("sqm")) and rad["sqm"] > 0:
            print(f"    EUI          {g.sum() / 2 / rad['sqm']:12,.1f}"
                  f"  (kWh/m2/år, ufylte hull)")
    if not funnet:
        print("  ingen målere funnet for dette bygget")
    print()


if __name__ == "__main__":
    main()
