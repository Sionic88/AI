from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures even without an interactive display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = (BASE_DIR / "building-data-genome-project-2-official"
            / "data/meters/raw/electricity.csv")
METADATA_FILE = (BASE_DIR / "building-data-genome-project-2-official"
                 / "data/metadata/metadata.csv")
BEFORE_FILE = BASE_DIR / "Cleaned_data/electricity_2016_cleaned.csv"
AFTER_FILE = BASE_DIR / "Cleaned_data/electricity_2016_final.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "Grafer"
PREFERRED_BUILDING = "Bull_education_Jeffery"
YEAR = 2016
EXPECTED_INDEX = pd.date_range("2016-01-01", "2017-01-01", freq="h", inclusive="left")

plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.22, "figure.dpi": 110})


def require_real_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Mangler fil: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"{path.name} er bare en Git LFS-peker. Kjor 'git lfs pull'.")


def read_processed(path):
    require_real_csv(path)
    data = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in data.columns:
        raise ValueError(f"Mangler timestamp i {path.name}")
    data = data.set_index("timestamp")
    if not data.index.equals(EXPECTED_INDEX):
        raise ValueError(f"{path.name}: forventet 8784 unike timer i riktig rekkefolge fra 2016.")
    if not all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise ValueError(f"{path.name}: enkelte bygningskolonner er ikke numeriske.")
    if not np.isfinite(data.to_numpy(copy=False)).all():
        raise ValueError(f"{path.name}: fant NaN eller uendelige verdier.")
    return data


def save(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=250, bbox_inches="tight")
    plt.close()
    print("  Lagret:", filename)


def read_original_coverage():
    """Read 2016 ORIGINAL meter values, not post-interpolation percentages."""
    require_real_csv(RAW_FILE)
    require_real_csv(METADATA_FILE)
    metadata = pd.read_csv(METADATA_FILE, usecols=["building_id", "sqm"])
    valid = metadata.loc[pd.to_numeric(metadata["sqm"], errors="coerce") > 0,
                         "building_id"].dropna().astype(str)

    raw_columns = list(pd.read_csv(RAW_FILE, nrows=0).columns)
    if "timestamp" not in raw_columns:
        raise ValueError("Rafilen mangler timestamp.")
    buildings = [name for name in raw_columns if name != "timestamp" and name in set(valid)]
    if not buildings:
        raise ValueError("Ingen bygningskolonner overlapper med gyldig sqm i metadata.")

    count = pd.Series(0, index=buildings, dtype="int64")
    timestamps = []
    for chunk in pd.read_csv(RAW_FILE, chunksize=1000):
        times = pd.to_datetime(chunk["timestamp"], errors="raise")
        year_mask = times.dt.year.eq(YEAR)
        if not year_mask.any():
            continue
        timestamps.extend(times.loc[year_mask].tolist())
        measured = chunk.loc[year_mask, buildings]
        # Convert explicitly so a nonnumeric string cannot count as a valid reading.
        measured = measured.apply(pd.to_numeric, errors="coerce")
        count += measured.notna().sum().astype("int64")

    idx = pd.DatetimeIndex(timestamps)
    if len(idx) != len(EXPECTED_INDEX) or not idx.is_unique or not idx.sort_values().equals(EXPECTED_INDEX):
        raise ValueError("Rafilen har ikke nøyaktig 8784 unike 2016-timer; kan ikke vise dekning mot 8784.")

    complete = int((count == len(idx)).sum())
    partial = int(((count > 0) & (count < len(idx))).sum())
    empty = int((count == 0).sum())
    print(f"Original 2016: {len(count)} bygninger | komplette={complete} | delvise={partial} | uten data={empty}")
    return count


def chart_coverage(count):
    """Separates exactly 100% coverage from almost 100%."""
    pct = count / len(EXPECTED_INDEX) * 100
    categories = [
        ("0 %", pct.eq(0)),
        (">0–<50 %", (pct > 0) & (pct < 50)),
        ("50–<80 %", (pct >= 50) & (pct < 80)),
        ("80–<95 %", (pct >= 80) & (pct < 95)),
        ("95–<100 %", (pct >= 95) & (pct < 100)),
        ("100 %", pct.eq(100)),
    ]
    labels = [label for label, _ in categories]
    heights = [int(mask.sum()) for _, mask in categories]
    assert sum(heights) == len(count)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, heights, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(heights) * 1.14 + 1)
    ax.set_ylabel("Number of buildings")
    ax.set_xlabel("Coverage of ORIGINAL hourly electricity measurements (2016)")
    ax.set_title("Data coverage before interpolation")
    fig.tight_layout()
    save("01_original_data_coverage.png")


def pick_week(data):
    """Find a genuinely varying nonzero week rather than assuming a date/building."""
    medians = data.median(axis=0)
    positive_fraction = data.gt(0).mean(axis=0)
    upper = data.quantile(0.99)
    eligible = medians.index[(medians > 1) & (positive_fraction > 0.90) &
                             (upper < 12 * medians)]
    if len(eligible) == 0:
        raise ValueError("Fant ingen egnede bygninger med regelmessig positivt forbruk.")
    candidate_ids = np.linspace(0, len(eligible) - 1, min(80, len(eligible)), dtype=int)
    candidates = eligible[candidate_ids]
    starts = ["2016-02-01", "2016-04-04", "2016-07-04", "2016-10-03"]
    best = None
    for start in starts:
        end = pd.Timestamp(start) + pd.Timedelta(days=7)
        subset = data.loc[(data.index >= start) & (data.index < end), candidates]
        for name in subset.columns:
            series = subset[name]
            typical = float(series.median())
            if typical <= 0 or series.le(0).mean() > 0.02:
                continue
            spread = float((series.quantile(.90) - series.quantile(.10)) / typical)
            peak = float(series.quantile(.99) / typical)
            if not (0.12 < spread < 4 and peak < 8):
                continue
            score = min(spread, 1.5)
            if best is None or score > best[0]:
                best = (score, name, series)
    if best is None:
        raise ValueError("Fant ikke en uke med tydelig og rimelig variasjon; ingen tom graf generert.")
    return best[1], best[2]


def chart_week(data):
    name, week = pick_week(data)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(week.index, week.values, linewidth=1.2)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Example of hourly electricity consumption: {name}")
    ax.set_xlabel("Date (2016)")
    ax.set_ylabel("Electricity per hour (kWh)")
    fig.autofmt_xdate()
    fig.tight_layout()
    save("02_representative_week.png")
    print(f"  Representative week: {name}, {week.index[0].date()} (actual data)")


def find_real_gap(before, count):
    """Retrieve raw 2016 readings for selected partly incomplete, retained buildings."""
    candidates = count.index[(count > len(EXPECTED_INDEX) * .70) &
                             (count < len(EXPECTED_INDEX)) &
                             count.index.isin(before.columns)]
    if len(candidates) == 0:
        raise ValueError("Ingen beholdte bygninger med delvis manglende ra-data.")
    # Sample across the entire list rather than assuming a particular building.
    selected = candidates[np.linspace(0, len(candidates)-1, min(220, len(candidates)), dtype=int)]
    columns = ["timestamp"] + selected.tolist()
    rows = []
    for chunk in pd.read_csv(RAW_FILE, usecols=columns, chunksize=2000):
        times = pd.to_datetime(chunk["timestamp"], errors="raise")
        mask = times.dt.year.eq(YEAR)
        if mask.any():
            rows.append(chunk.loc[mask].copy())
    raw = pd.concat(rows, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index("timestamp").reindex(EXPECTED_INDEX)
    raw = raw.apply(pd.to_numeric, errors="coerce")

    best = None
    for name in selected:
        values = raw[name].to_numpy(dtype=float)
        missing = np.isnan(values)
        if not missing.any():
            continue
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
        for left, right in zip(starts, ends):
            gap = right - left + 1
            if not (6 <= gap <= 36 and left >= 30 and right < len(values)-31):
                continue
            a, b = values[left-1], values[right+1]
            if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
                continue
            filled = before[name].iloc[left:right+1].to_numpy()
            predicted = np.linspace(a, b, gap+2)[1:-1]
            if not np.allclose(filled, predicted, atol=1e-4, rtol=1e-4):
                continue  # Only show real gaps the original pipeline filled linearly.
            context = values[left-24:right+25]
            context = context[np.isfinite(context)]
            median = float(np.median(context)) if len(context) else 0
            if median <= 0:
                continue
            slope = abs(a - b) / max(median, 1)
            local_spread = (np.quantile(context, .90) - np.quantile(context, .10)) / median
            if local_spread < .04:
                continue
            score = min(slope, 3) + min(local_spread, 2) * .3
            if best is None or score > best[0]:
                best = (score, name, raw[name], left, right)
    if best is None:
        raise ValueError("Fant ikke et etterprovbart 6–36-timers gap som er lineart interpolert.")
    return best[1:]


def chart_interpolation(before, count):
    name, original, left, right = find_real_gap(before, count)
    start = max(0, left-24)
    stop = min(len(original), right+26)
    raw_part = original.iloc[start:stop]
    after_part = before[name].iloc[start:stop]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(raw_part.index, raw_part.values, "o", markersize=2.8,
            label="Original available measurements")
    ax.plot(after_part.index, after_part.values, linewidth=1.5,
            label="After linear interpolation")
    ax.axvspan(original.index[left], original.index[right], color="orange",
               alpha=.20, label=f"Actual missing gap: {right-left+1} hours")
    ax.set_title(f"Interpolation of an ACTUAL data gap: {name}")
    ax.set_xlabel("Date (2016)")
    ax.set_ylabel("Electricity per hour (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    save("03_real_interpolation.png")
    print(f"  Actual gap: {name}, {original.index[left]} – {original.index[right]}")


def pick_correction(before, after):
    """Pick a genuine corrected spike, favouring the user's previous example."""
    names = ([PREFERRED_BUILDING] if PREFERRED_BUILDING in before.columns else [])
    if not names:
        totals = before.sum() - after.sum()
        names = totals.sort_values(ascending=False).index[:40].tolist()
    else:
        totals = before.sum() - after.sum()
        names += [n for n in totals.nlargest(30).index if n != PREFERRED_BUILDING]

    for name in names:
        diff = (before[name] - after[name]).to_numpy(dtype=float)
        if diff.max() > 0 and np.count_nonzero(~np.isclose(diff, 0, atol=1e-8)) > 0:
            location = int(np.argmax(diff))
            return name, location
    raise ValueError("Fant ingen faktiske korrigerte topper mellom datafilene.")


def correction_range(before, location, hours=48):
    start = max(0, location-hours)
    end = min(len(before), location+hours+1)
    return before.index[start:end]


def pick_iqr_z_example(before, after):
    """Find a real 4-day period that demonstrates IQR / Z-score disagreement."""
    preferred = [n for n in ["Fox_public_Belle", PREFERRED_BUILDING]
                 if n in before.columns]
    sample_ids = np.linspace(0, len(before.columns)-1,
                             min(200, len(before.columns)), dtype=int)
    candidates = list(dict.fromkeys(preferred + before.columns[sample_ids].tolist()))
    best = None
    window = 96
    for name in candidates:
        series = before[name]
        q1, q3 = series.quantile([.25, .75])
        iqr_flag = (series < q1 - 1.5 * (q3-q1)) | (series > q3 + 1.5 * (q3-q1))
        std = float(series.std())
        if std <= 0:
            continue
        z_flag = (series - series.mean()).abs() > 3 * std
        iqr_only = (iqr_flag & ~z_flag).astype(int)
        # Count disagreements within each 96-hour window.
        disagreements = iqr_only.rolling(window, min_periods=window).sum()
        for end_index in disagreements.nlargest(3).index:
            right = series.index.get_loc(end_index)
            left = right - window + 1
            part = series.iloc[left:right+1]
            median = float(part.median())
            if median <= 0 or part.quantile(.9) <= part.quantile(.1):
                continue
            different = int(disagreements.loc[end_index])
            # Choose a period that shows differences without massive spikes
            # dominating the whole graph.
            peak_ratio = float(part.max() / max(median, 1))
            score = different - max(0, peak_ratio-12)*.5
            if best is None or score > best[0]:
                best = (score, name, part.index)
    if best is not None and best[0] > 3:
        return best[1], best[2]
    # If no informative disagreement period exists, use a genuine corrected event.
    name, loc = pick_correction(before, after)
    return name, correction_range(before, loc)


def chart_iqr_z(before, after):
    # No 'invented' example: the chosen window is selected from actual readings.
    name, index = pick_iqr_z_example(before, after)
    values = before.loc[index, name]
    annual = before[name]
    q1, q3 = annual.quantile([.25, .75])
    iqr = q3 - q1
    iqr_flag = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    std = annual.std()
    z_flag = ((values - annual.mean()).abs() > 3 * std) if std > 0 else pd.Series(False, index=index)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, flags, title in zip(axes, [iqr_flag, z_flag], ["IQR (annual thresholds)", "Z-score > 3 (annual thresholds)"]):
        ax.plot(index, values.values, linewidth=1, color="0.40", label="Actual readings")
        ax.scatter(index[flags], values.loc[flags], s=24, color="tab:orange",
                   label=f"Flagged: {int(flags.sum())}", zorder=3)
        typical = max(float(annual.median()), 1)
        ax.set_yscale("symlog", linthresh=typical / 4)
        ax.set_ylabel("Hourly consumption (kWh, symlog)")
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Date (2016)")
    fig.suptitle(f"IQR versus Z-score, same actual readings: {name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    save("04_iqr_vs_zscore.png")
    print(f"  IQR/Z-score comparison: {name}, {index[0].date()} – {index[-1].date()}")


def chart_mad_correction(before, after):
    name, location = pick_correction(before, after)
    index = correction_range(before, location)
    original = before.loc[index, name]
    corrected = after.loc[index, name]
    changed = ~np.isclose(original.values, corrected.values, rtol=1e-9, atol=1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(index, original, color="tab:red", lw=1.1, label="Before MAD correction")
    axes[0].plot(index, corrected, color="tab:blue", lw=1.1, label="After MAD correction")
    axes[0].scatter(index[changed], original.values[changed], s=26,
                    color="tab:orange", label="Measurements replaced", zorder=3)
    axes[0].set_ylabel("Hourly consumption (kWh)")
    axes[0].set_title("Full scale: extreme peaks")
    axes[0].legend()

    axes[1].plot(index, original, color="tab:red", lw=1.0, alpha=.55,
                 label="Before correction")
    axes[1].plot(index, corrected, color="tab:blue", lw=1.4,
                 label="After interpolation")
    axes[1].scatter(index[changed], corrected.values[changed], s=28,
                    color="tab:green", label="Replacement values", zorder=3)
    # Zoom to the usual range. Explicitly describe this scale in the title.
    zoom_top = float(np.quantile(corrected, .96) * 1.2)
    if zoom_top > 0:
        axes[1].set_ylim(0, zoom_top)
    axes[1].set_title("Zoomed view: ordinary consumption (values above range are clipped)")
    axes[1].set_ylabel("Hourly consumption (kWh)")
    axes[1].set_xlabel("Date (2016)")
    axes[1].legend(loc="upper right")
    fig.suptitle(f"Before / after rolling median + MAD: {name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    save("05_mad_before_after.png")


def chart_validation(before, after):
    original_totals, clean_totals = before.sum(), after.sum()
    nonzero = original_totals.gt(0)
    diff_percent = ((clean_totals[nonzero] - original_totals[nonzero])
                    / original_totals[nonzero] * 100)
    if (diff_percent.abs() > 3.00001).any():
        raise ValueError("Noen bygninger overskrider 3 %-grensen. Kontroller inputfilene.")
    abs_change = diff_percent.abs()
    category_counts = [
        int((abs_change < 1e-8).sum()),
        int(((abs_change >= 1e-8) & (abs_change < .5)).sum()),
        int(((abs_change >= .5) & (abs_change < 1)).sum()),
        int(((abs_change >= 1) & (abs_change < 2)).sum()),
        int(((abs_change >= 2) & (abs_change <= 3)).sum()),
    ]
    labels = ["No change", ">0–<0.5 %", "0.5–<1 %", "1–<2 %", "2–3 %"]
    if not sum(category_counts) == int(nonzero.sum()):
        raise ValueError("Intervallene i valideringsgrafen dekker ikke alle bygningene.")
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    bars = ax.bar(labels, category_counts, edgecolor="black", linewidth=.6)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(category_counts)*1.14+1)
    ax.set_ylabel("Number of buildings")
    ax.set_xlabel("Absolute change in annual consumption")
    ax.set_title("Impact of MAD correction on annual consumption")
    fig.tight_layout()
    save("06_validation_change.png")
    print(f"Validation: mean absolute change={abs_change.mean():.3f} %, "
          f"max={abs_change.max():.3f} %, "
          f">1%={int((abs_change>1).sum())}, >3%={int((abs_change>3).sum())}")
    if not nonzero.all():
        print(f"  Note: {int((~nonzero).sum())} buildings with zero original total excluded from relative %.")


def main():
    print("Loading original 2016 electricity coverage...")
    coverage = read_original_coverage()
    chart_coverage(coverage)

    print("Loading two PREPROCESSED 2016 files...")
    before = read_processed(BEFORE_FILE)
    after = read_processed(AFTER_FILE)
    if not before.columns.equals(after.columns):
        raise ValueError("Before- og after-filen har ulike bygningkolonner.")
    if not set(before.columns).issubset(coverage.index):
        raise ValueError("Behandlet data har bygninger som mangler fra raw + gyldig sqm.")
    print(f"Preprocessed files: {len(before.columns)} buildings; 8784 hours")

    chart_week(after)
    try:
        chart_interpolation(before, coverage)
    except ValueError as exc:
        print("  Skipper interpolasjonsgraf (ikke sikker kandidat):", exc)
    chart_iqr_z(before, after)
    chart_mad_correction(before, after)
    chart_validation(before, after)
    print("Finished. Figures saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()