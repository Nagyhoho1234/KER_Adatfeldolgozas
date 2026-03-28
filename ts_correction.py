"""
ts_correction.py  --  Time-series cleaning and level-shift correction
for KER groundwater monitoring stations.

Steps per channel (CH0, CH1, CH3):
  1. Segment into sections separated by gaps > 3 h
  2. Remove outlier spikes (rolling median ± k*MAD), two passes
  3. Detect intra-section level shifts (change-point detection)
  4. Trim garbage readings at section/shift edges
  5. Correct level shifts between all sections (last = reference)
  6. Interpolate small remaining gaps (<= 6 h)
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────
GAP_THRESHOLD_H = 3          # hours of missing data -> new section
OUTLIER_K = 5                # rolling median ± k*MAD (pass 1)
OUTLIER_K2 = 3.5             # tighter threshold (pass 2, within sections)
OUTLIER_WINDOW = 25          # rolling window size (hours)
SPIKE_WINDOW = 5             # short spike detector window
EDGE_POINTS = 3              # points to inspect at each section edge
EDGE_K = 3                   # section-interior median ± k*MAD for edge test
LEVEL_REF_PTS = 6            # points used to estimate section boundary level
CHANGEPOINT_WINDOW = 12      # hours: compare median of this many points before/after
CHANGEPOINT_MIN_SHIFT = None # auto-calculated per section as multiple of local MAD
CHANGEPOINT_K = 4            # change-point must exceed k * local_MAD
CHANGEPOINT_MIN_SECTION = 12 # minimum section length to keep after splitting
INTERP_MAX_GAP_H = 6         # max gap to interpolate (hours)
CHANNELS = ["CH0", "CH1", "CH3"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_station(station_id: str) -> pd.DataFrame:
    """Load raw CSV, keep only hourly rows, set proper DatetimeIndex."""
    path = DATA_DIR / station_id / f"{station_id}_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    cols = [c for c in CHANNELS if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(how="all")
    df = df.resample("1h").first()
    return df


def segment(series: pd.Series, gap_hours: int = GAP_THRESHOLD_H):
    """Split a series into sections based on gaps > gap_hours."""
    valid = series.dropna()
    if len(valid) == 0:
        return []
    dt = valid.index.to_series().diff()
    break_mask = dt > pd.Timedelta(hours=gap_hours)
    break_points = valid.index[break_mask]

    sections = []
    prev_start = valid.index[0]
    for bp in break_points:
        sec = valid.loc[prev_start:bp]
        sec = sec.iloc[:-1] if len(sec) > 1 else sec
        if len(sec) > 0:
            sections.append((sec.index[0], sec.index[-1]))
        prev_start = bp
    sec = valid.loc[prev_start:]
    if len(sec) > 0:
        sections.append((sec.index[0], sec.index[-1]))
    return sections


def rolling_mad_outliers(series: pd.Series, window: int = OUTLIER_WINDOW,
                         k: float = OUTLIER_K) -> tuple:
    """Flag outliers using rolling median ± k * MAD. Returns cleaned series."""
    s = series.copy()
    roll_med = s.rolling(window, center=True, min_periods=3).median()
    roll_dev = (s - roll_med).abs()
    roll_mad = roll_dev.rolling(window, center=True, min_periods=3).median()
    roll_mad = roll_mad.replace(0, np.nan).fillna(roll_dev.median())
    if roll_mad.max() == 0:
        roll_mad[:] = 1.0
    outlier_mask = roll_dev > k * roll_mad
    n_outliers = outlier_mask.sum()
    s[outlier_mask] = np.nan
    return s, int(n_outliers)


def detect_spikes(series: pd.Series, sections, window: int = SPIKE_WINDOW,
                  k: float = OUTLIER_K2) -> tuple:
    """Second-pass spike detector: within each section, find isolated
    points or short bursts that deviate from local neighbors."""
    s = series.copy()
    n_removed = 0
    for (start, end) in sections:
        sec = s.loc[start:end].dropna()
        if len(sec) < window * 2:
            continue
        # Local median filter
        local_med = sec.rolling(window, center=True, min_periods=2).median()
        dev = (sec - local_med).abs()
        local_mad = dev.rolling(window * 3, center=True, min_periods=3).median()
        local_mad = local_mad.replace(0, np.nan).fillna(dev.median())
        if local_mad.max() == 0:
            continue
        spike_mask = dev > k * local_mad
        n = spike_mask.sum()
        if n > 0:
            s.loc[sec.index[spike_mask]] = np.nan
            n_removed += int(n)
    return s, n_removed


def detect_changepoints(series: pd.Series, sections,
                        window: int = CHANGEPOINT_WINDOW,
                        k: float = CHANGEPOINT_K,
                        min_section_len: int = CHANGEPOINT_MIN_SECTION) -> list:
    """Detect intra-section level shifts by comparing medians before/after
    each point. Returns expanded list of sections split at change points."""
    all_sections = []

    for (start, end) in sections:
        sec = series.loc[start:end].dropna()
        if len(sec) < window * 3:
            all_sections.append((start, end))
            continue

        # Compute running difference of local medians
        # For each point i, compare median(i-window:i) vs median(i:i+window)
        vals = sec.values
        n = len(vals)
        jumps = np.zeros(n)

        for i in range(window, n - window):
            before = np.nanmedian(vals[max(0, i - window):i])
            after = np.nanmedian(vals[i:i + window])
            jumps[i] = after - before

        # The local variability (MAD of first differences, excluding the jumps)
        first_diff = np.abs(np.diff(vals))
        local_mad = np.nanmedian(first_diff)
        if local_mad == 0:
            local_mad = np.nanmean(first_diff)
        if local_mad == 0:
            all_sections.append((start, end))
            continue

        # Find significant jumps
        threshold = k * local_mad
        # Also require minimum absolute jump (avoid splitting on noise)
        abs_min = local_mad * 2
        threshold = max(threshold, abs_min)

        jump_abs = np.abs(jumps)
        candidates = np.where(jump_abs > threshold)[0]

        if len(candidates) == 0:
            all_sections.append((start, end))
            continue

        # Cluster nearby change points (within 2*window) and pick the strongest
        change_indices = []
        i = 0
        while i < len(candidates):
            cluster = [candidates[i]]
            j = i + 1
            while j < len(candidates) and candidates[j] - candidates[j-1] <= window * 2:
                cluster.append(candidates[j])
                j += 1
            # Pick the index with the largest jump in this cluster
            best = cluster[np.argmax(jump_abs[cluster])]
            change_indices.append(best)
            i = j

        # Split the section at change points
        split_starts = [0] + change_indices
        split_ends = change_indices + [n]

        for s_start, s_end in zip(split_starts, split_ends):
            if s_end - s_start >= min_section_len:
                sub = sec.iloc[s_start:s_end]
                all_sections.append((sub.index[0], sub.index[-1]))

    return all_sections


def trim_edges(series: pd.Series, sections, n: int = EDGE_POINTS,
               k: float = EDGE_K) -> tuple:
    """Mark edge points as NaN if they deviate from the section interior."""
    s = series.copy()
    trimmed = 0
    for (start, end) in sections:
        sec = s.loc[start:end].dropna()
        if len(sec) < 2 * n + 4:
            continue
        interior = sec.iloc[n:-n]
        if len(interior) == 0:
            continue
        med = interior.median()
        mad = (interior - med).abs().median()
        if mad == 0:
            mad = (interior - med).abs().mean()
        if mad == 0:
            continue

        threshold = k * mad
        for idx in sec.index[:n]:
            if abs(s.loc[idx] - med) > threshold:
                s.loc[idx] = np.nan
                trimmed += 1
        for idx in sec.index[-n:]:
            if abs(s.loc[idx] - med) > threshold:
                s.loc[idx] = np.nan
                trimmed += 1
    return s, trimmed


def correct_level_shifts(series: pd.Series, sections,
                         ref_pts: int = LEVEL_REF_PTS) -> tuple:
    """Shift sections so they connect smoothly.
    LAST section = reference (offset 0). Earlier sections are shifted
    backwards to align with it."""
    s = series.copy()
    if len(sections) < 2:
        return s, [0.0] if sections else []

    n = len(sections)
    offsets = [0.0] * n

    for i in range(n - 2, -1, -1):
        curr_start, curr_end = sections[i]
        next_start, next_end = sections[i + 1]

        curr_vals = s.loc[curr_start:curr_end].dropna()
        next_vals = s.loc[next_start:next_end].dropna()

        if len(curr_vals) < 2 or len(next_vals) < 2:
            offsets[i] = offsets[i + 1]
            continue

        curr_end_level = curr_vals.iloc[-min(ref_pts, len(curr_vals)):].median()
        next_start_level = next_vals.iloc[:min(ref_pts, len(next_vals))].median()

        shift = next_start_level - curr_end_level
        offsets[i] = shift + offsets[i + 1]

    for i, (start, end) in enumerate(sections):
        if offsets[i] != 0:
            mask = (s.index >= start) & (s.index <= end) & s.notna()
            s.loc[mask] += offsets[i]

    return s, offsets


def interpolate_small_gaps(series: pd.Series,
                           max_gap_h: int = INTERP_MAX_GAP_H) -> tuple:
    """Linearly interpolate gaps of <= max_gap_h hours."""
    s = series.copy()
    before_nans = s.isna().sum()
    s = s.interpolate(method="time", limit=max_gap_h)
    after_nans = s.isna().sum()
    filled = int(before_nans - after_nans)
    return s, filled


def process_channel(series: pd.Series, channel_name: str, station: str) -> tuple:
    """Full pipeline for one channel. Returns corrected series + stats dict."""
    stats = {"channel": channel_name, "station": station,
             "n_total": len(series), "n_valid_raw": int(series.notna().sum())}

    # 1. Segment by gaps
    sections = segment(series)
    stats["n_gap_sections"] = len(sections)

    # 2. Outlier removal - pass 1 (global rolling MAD)
    s, n_outliers1 = rolling_mad_outliers(series, window=OUTLIER_WINDOW, k=OUTLIER_K)

    # Re-segment
    sections = segment(s)

    # 3. Outlier removal - pass 2 (tighter, within-section spike detection)
    s, n_outliers2 = detect_spikes(s, sections, window=SPIKE_WINDOW, k=OUTLIER_K2)
    stats["n_outliers_removed"] = n_outliers1 + n_outliers2

    # Re-segment
    sections = segment(s)

    # 4. Detect intra-section level shifts (change-point detection)
    sections = detect_changepoints(s, sections)
    stats["n_sections_with_changepoints"] = len(sections)

    # 5. Trim edges (now including edges at detected change points)
    s, n_trimmed = trim_edges(s, sections)
    stats["n_edges_trimmed"] = n_trimmed

    # Re-segment (trimming may have created new gaps)
    sections = segment(s)
    stats["n_final_sections"] = len(sections)

    # 6. Level shift correction (last section = reference)
    s, offsets = correct_level_shifts(s, sections)
    stats["max_level_shift"] = round(max(abs(o) for o in offsets), 4) if offsets else 0

    # 7. Interpolate small gaps
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
            print(f"    Gap sections:      {stats['n_gap_sections']}")
            print(f"    + changepoints:    {stats['n_sections_with_changepoints']}")
            print(f"    Final sections:    {stats['n_final_sections']}")
            print(f"    Outliers removed:  {stats['n_outliers_removed']}")
            print(f"    Edges trimmed:     {stats['n_edges_trimmed']}")
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
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Station':<10} {'Chan':<5} {'GapSec':>6} {'ChgPt':>6} {'Final':>6} "
          f"{'Outliers':>8} {'Trimmed':>8} {'Interp':>7} {'MaxShift':>9} {'Final%':>7}")
    print("-" * 78)
    for st, channels in all_results.items():
        for ch, s in channels.items():
            pct = 100 * s["n_valid_final"] / s["n_total"] if s["n_total"] else 0
            print(f"{st:<10} {ch:<5} {s['n_gap_sections']:>6} "
                  f"{s['n_sections_with_changepoints']:>6} "
                  f"{s['n_final_sections']:>6} "
                  f"{s['n_outliers_removed']:>8} {s['n_edges_trimmed']:>8} "
                  f"{s['n_interpolated']:>7} {s['max_level_shift']:>9.4f} "
                  f"{pct:>6.1f}%")

    print(f"\nDone. Corrected files saved to data/KER*/KER*_corrected.csv")


if __name__ == "__main__":
    main()
