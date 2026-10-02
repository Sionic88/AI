"""
Skriver ut alt datasettet inneholder om ett enkelt bygg.

Bruk:
    python bygg_info.py              # spør "Skriv inn bygg" ved oppstart
    python bygg_info.py Lashandra
    python bygg_info.py Robin_Education_Lashandra

Søket er delvis og ufølsomt for store og små bokstaver.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "building-data-genome-project-2-official" / "data")
METER_TYPES = ["electricity", "hotwater", "chilledwater", "steam",
               "gas", "water", "irrigation", "solar"]


def main():
    if len(sys.argv) > 1:
        navn = sys.argv[1]
    else:
        navn = input("Skriv inn bygg (f.eks. Lashandra): ").strip()
        if not navn:
            return
    sok = navn.lower()

    md = pd.read_csv(DATA_DIR / "metadata" / "metadata.csv")
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

    for _, rad in treff.iterrows():
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
            sti = DATA_DIR / "meters" / "raw" / f"{m}.csv"
            if not sti.exists():
                continue
            kols = pd.read_csv(sti, nrows=0).columns
            if bygg not in kols:
                continue
            funnet = True
            s = pd.read_csv(sti, usecols=["timestamp", bygg],
                            parse_dates=["timestamp"],
                            index_col="timestamp")[bygg]
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
