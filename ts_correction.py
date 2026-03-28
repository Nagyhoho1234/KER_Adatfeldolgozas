"""
ts_correction.py  --  Time-series cleaning and level-shift correction
for KER groundwater monitoring stations.

Steps per channel (CH0, CH1, CH3):
  1. Segment into sections separated by gaps > 3 h
  2. Remove outlier spikes (rolling median +/- 5*MAD)
  3. Trim garbage readings at section edges
  4. Correct level shifts between consecutive sections
  5. Interpolate small remaining gaps (<= 6 h)
"""

import sys, os, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────
GAP_THRESHOLD_H = 3          # hours of missing data -> new section
OUTLIER_K = 5                # rolling median ± k*MAD
OUTLIER_WINDOW = 25          # rolling window size (hours)
EDGE_POINTS = 3              # points to inspect at each section edge
EDGE_K = 3                   # section-interior median ± k*MAD for edge test
LEVEL_REF_PTS = 6            # points used to estimate section boundary level
INTERP_MAX_GAP_H = 6         # max gap to interpolate (hours)
CHANNELS = ["CH0", "CH1", "CH3"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_station(station_id: str) -> pd.DataFrame:
    """Load raw CSV, keep only hourly rows, set proper DatetimeIndex."""
    path = DATA_DIR / station_id / f"{station_id}_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    # Keep only the science channels
    cols = [c for c in CHANNELS if c in df.columns]
    df = df[cols].copy()
    # Drop rows where ALL science channels are NaN (battery-only rows)
    df = df.dropna(how="all")
    # Resample to exact hourly grid (picks the value closest to each hour)
    df = df.resample("1h").first()
    return df


def segment(series: pd.Series, gap_hours: int = GAP_THRESHOLD_H):
    """Split a series into sections based on gaps > gap_hours.
    Returns list of (start_idx, end_idx) integer index pairs for valid
    (non-NaN) contiguous blocks."""
    valid = series.dropna()
    if len(valid) == 0:
        return []
    dt = valid.index.to_series().diff()
    break_mask = dt > pd.Timedelta(hours=gap_hours)
    break_points = valid.index[break_mask]

    sections = []
    prev_start = valid.index[0]
    for bp in break_points:
        # end of previous section = last valid before this break
        sec = valid.loc[prev_start:bp]
        # exclude the break point itself (it starts the next section)
        sec = sec.iloc[:-1] if len(sec) > 1 else sec
        if len(sec) > 0:
            sections.append((sec.index[0], sec.index[-1]))
        prev_start = bp
    # last section
    sec = valid.loc[prev_start:]
    if len(sec) > 0:
        sections.append((sec.index[0], sec.index[-1]))
    return sections


def rolling_mad_outliers(series: pd.Series, window: int = OUTLIER_WINDOW,
                         k: float = OUTLIER_K) -> pd.Series:
    """Flag outliers using rolling median ± k * MAD. Returns cleaned series."""
    s = series.copy()
    roll_med = s.rolling(window, center=True, min_periods=3).median()
    roll_dev = (s - roll_med).abs()
    roll_mad = roll_dev.rolling(window, center=True, min_periods=3).median()
    # Avoid zero MAD (constant sections)
    roll_mad = roll_mad.replace(0, np.nan).fillna(roll_dev.median())
    if roll_mad.max() == 0:
        roll_mad[:] = 1.0  # truly constant series, nothing to flag
    outlier_mask = roll_dev > k * roll_mad
    n_outliers = outlier_mask.sum()
    s[outlier_mask] = np.nan
    return s, int(n_outliers)


def trim_edges(series: pd.Series, sections, n: int = EDGE_POINTS,
               k: float = EDGE_K) -> pd.Series:
    """Mark edge points as NaN if they deviate from the section interior."""
    s = series.copy()
    trimmed = 0
    for (start, end) in sections:
        sec = s.loc[start:end].dropna()
        if len(sec) < 2 * n + 4:
            continue  # section too short to trim
        interior = sec.iloc[n:-n]
        if len(interior) == 0:
            continue
        med = interior.median()
        mad = (interior - med).abs().median()
        if mad == 0:
            mad = (interior - med).abs().mean()
        if mad == 0:
            continue  # constant section

        threshold = k * mad
        # Check first n points
        for idx in sec.index[:n]:
            if abs(s.loc[idx] - med) > threshold:
                s.loc[idx] = np.nan
                trimmed += 1
        # Check last n points
        for idx in sec.index[-n:]:
            if abs(s.loc[idx] - med) > threshold:
                s.loc[idx] = np.nan
                trimmed += 1
    return s, trimmed


def correct_level_shifts(series: pd.Series, sections,
                         ref_pts: int = LEVEL_REF_PTS) -> (pd.Series, list):
    """Shift sections so they connect smoothly.
    LAST section = reference (offset 0). Earlier sections are shifted
    backwards to align with it, because the most recent sensor reading
    is considered the current truth."""
    s = series.copy()
    if len(sections) < 2:
        return s, [0.0] if sections else []

    n = len(sections)
    offsets = [0.0] * n  # last section stays at 0

    # Walk backwards from the last section
    for i in range(n - 2, -1, -1):
        curr_start, curr_end = sections[i]      # the earlier section
        next_start, next_end = sections[i + 1]   # the later section (already aligned)

        curr_vals = s.loc[curr_start:curr_end].dropna()
        next_vals = s.loc[next_start:next_end].dropna()

        if len(curr_vals) < 2 or len(next_vals) < 2:
            offsets[i] = offsets[i + 1]
            continue

        # End of earlier section vs start of later section
        curr_end_level = curr_vals.iloc[-min(ref_pts, len(curr_vals)):].median()
        next_start_level = next_vals.iloc[:min(ref_pts, len(next_vals))].median()

        # Shift the earlier section so its end matches the next section's start
        shift = next_start_level - curr_end_level
        offsets[i] = shift + offsets[i + 1]

    # Apply offsets (last section has offset=0, earlier sections get shifted)
    for i, (start, end) in enumerate(sections):
        if offsets[i] != 0:
            mask = (s.index >= start) & (s.index <= end) & s.notna()
            s.loc[mask] += offsets[i]

    return s, offsets


def interpolate_small_gaps(series: pd.Series,
                           max_gap_h: int = INTERP_MAX_GAP_H) -> (pd.Series, int):
    """Linearly interpolate gaps of <= max_gap_h hours."""
    s = series.copy()
    before_nans = s.isna().sum()
    s = s.interpolate(method="time", limit=max_gap_h)
    after_nans = s.isna().sum()
    filled = int(before_nans - after_nans)
    return s, filled


def process_channel(series: pd.Series, channel_name: str, station: str) -> (pd.Series, dict):
    """Full pipeline for one channel. Returns corrected series + stats dict."""
    stats = {"channel": channel_name, "station": station,
             "n_total": len(series), "n_valid_raw": int(series.notna().sum())}

    # 1. Segment
    sections = segment(series)
    stats["n_sections"] = len(sections)

    # 2. Outlier removal
    s, n_outliers = rolling_mad_outliers(series)
    stats["n_outliers_removed"] = n_outliers

    # Re-segment after outlier removal (boundaries may have changed)
    sections = segment(s)

    # 3. Trim edges
    s, n_trimmed = trim_edges(s, sections)
    stats["n_edges_trimmed"] = n_trimmed

    # Re-segment
    sections = segment(s)
    stats["n_sections_after_trim"] = len(sections)

    # 4. Level shift correction
    s, offsets = correct_level_shifts(s, sections)
    stats["level_offsets"] = [round(o, 6) for o in offsets]
    stats["max_level_shift"] = round(max(abs(o) for o in offsets), 6) if offsets else 0

    # 5. Interpolate small gaps
    s, n_filled = interpolate_small_gaps(s)
    stats["n_interpolated"] = n_filled
    stats["n_valid_final"] = int(s.notna().sum())
    stats["n_remaining_nan"] = int(s.isna().sum())

    return s, stats


def process_station(station_id: str, verbose: bool = True) -> dict:
    """Process all channels for one station."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Station: {station_id}")
        print(f"{'='*60}")

    df = load_station(station_id)
    if verbose:
        print(f"  Loaded {len(df)} hourly rows, "
              f"period {df.index.min()} to {df.index.max()}")

    result = df.copy()
    all_stats = {}

    for ch in CHANNELS:
        if ch not in df.columns:
            if verbose:
                print(f"  {ch}: not present, skipping")
            continue
        corrected, stats = process_channel(df[ch], ch, station_id)
        result[ch] = corrected
        all_stats[ch] = stats

        if verbose:
            print(f"\n  --- {ch} ---")
            print(f"    Sections:          {stats['n_sections']} "
                  f"(-> {stats['n_sections_after_trim']} after trim)")
            print(f"    Outliers removed:  {stats['n_outliers_removed']}")
            print(f"    Edges trimmed:     {stats['n_edges_trimmed']}")
            print(f"    Level offsets:     {stats['level_offsets']}")
            print(f"    Max shift:         {stats['max_level_shift']}")
            print(f"    Gaps interpolated: {stats['n_interpolated']} points")
            print(f"    Valid: {stats['n_valid_raw']} -> {stats['n_valid_final']} "
                  f"({stats['n_remaining_nan']} NaN remain)")

    # Save
    out_path = DATA_DIR / station_id / f"{station_id}_corrected.csv"
    result.to_csv(out_path)
    if verbose:
        print(f"\n  Saved -> {out_path}")

    return all_stats


def main():
    stations_arg = sys.argv[1:] if len(sys.argv) > 1 else None

    if stations_arg:
        stations = stations_arg
    else:
        # All KER* directories
        stations = sorted([
            p.name for p in DATA_DIR.iterdir()
            if p.is_dir() and p.name.startswith("KER")
            and (p / f"{p.name}_raw.csv").exists()
        ])

    print(f"Processing {len(stations)} station(s): {', '.join(stations)}")

    all_results = {}
    for st in stations:
        try:
            all_results[st] = process_station(st)
        except Exception as e:
            print(f"\n  *** ERROR on {st}: {e}")

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Station':<10} {'Chan':<5} {'Sections':>8} {'Outliers':>9} "
          f"{'Trimmed':>8} {'Interp':>7} {'MaxShift':>10} {'Final%':>7}")
    print("-" * 68)
    for st, channels in all_results.items():
        for ch, s in channels.items():
            pct = 100 * s["n_valid_final"] / s["n_total"] if s["n_total"] else 0
            print(f"{st:<10} {ch:<5} {s['n_sections']:>8} "
                  f"{s['n_outliers_removed']:>9} {s['n_edges_trimmed']:>8} "
                  f"{s['n_interpolated']:>7} {s['max_level_shift']:>10.4f} "
                  f"{pct:>6.1f}%")

    print(f"\nDone. Corrected files saved to data/KER*/KER*_corrected.csv")


if __name__ == "__main__":
    main()
