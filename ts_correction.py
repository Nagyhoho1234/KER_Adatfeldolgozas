"""
ts_correction.py — Clean rewrite.

Per channel:
  1. Resample to hourly
  2. Remove spikes: median filter, anything > 5*MAD from rolling median → NaN
  3. Segment by gaps > 3h
  4. For each segment boundary: take median of last/first 7 DAYS (168h)
     as the segment level. Spikes can't move a median.
  5. Align segments backward (last segment = reference)
  6. Interpolate gaps ≤ 6h
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

CHANNELS = ["CH0", "CH1", "CH3"]
GAP_H = 3
MEDIAN_WINDOW = 25
SPIKE_K = 4
ALIGN_DAYS = 7        # 7 days = 168 hours for robust boundary level
INTERP_LIMIT = 6

DATA_DIR = Path(__file__).resolve().parent / "data"


def load(station_id):
    path = DATA_DIR / station_id / f"{station_id}_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    cols = [c for c in CHANNELS if c in df.columns]
    df = df[cols].dropna(how="all")
    return df.resample("1h").first()


def remove_spikes(s, window=MEDIAN_WINDOW, k=SPIKE_K, passes=3):
    """Multi-pass median filter. Each pass removes spikes and recalculates."""
    out = s.copy()
    total = 0
    for _ in range(passes):
        med = out.rolling(window, center=True, min_periods=3).median()
        dev = (out - med).abs()
        mad = dev.rolling(window, center=True, min_periods=3).median()
        mad = mad.replace(0, np.nan).fillna(dev.median())
        if mad.max() == 0:
            break
        bad = dev > k * mad
        n = bad.sum()
        if n == 0:
            break
        out[bad] = np.nan
        total += int(n)
    return out, total


def segment(s, gap_h=GAP_H):
    """Split into contiguous sections separated by gaps > gap_h."""
    valid = s.dropna()
    if len(valid) == 0:
        return []
    dt = valid.index.to_series().diff()
    breaks = valid.index[dt > pd.Timedelta(hours=gap_h)]
    sections = []
    start = valid.index[0]
    for bp in breaks:
        sec = valid.loc[start:bp].iloc[:-1]
        if len(sec) > 0:
            sections.append((sec.index[0], sec.index[-1]))
        start = bp
    sec = valid.loc[start:]
    if len(sec) > 0:
        sections.append((sec.index[0], sec.index[-1]))
    return sections


def align(s, sections, days=ALIGN_DAYS):
    """Align segments using median of last/first N days as boundary level.
    Last segment = reference (offset 0). Walk backwards."""
    out = s.copy()
    n = len(sections)
    if n < 2:
        return out, []

    hrs = days * 24
    offsets = [0.0] * n

    for i in range(n - 2, -1, -1):
        # This segment's end level: median of last N days
        cs, ce = sections[i]
        ns, ne = sections[i + 1]

        this_data = out.loc[cs:ce].dropna()
        next_data = out.loc[ns:ne].dropna()

        if len(this_data) < 6 or len(next_data) < 6:
            offsets[i] = offsets[i + 1]
            continue

        this_end = this_data.iloc[-min(hrs, len(this_data)):].median()
        next_start = next_data.iloc[:min(hrs, len(next_data))].median()

        offsets[i] = (next_start - this_end) + offsets[i + 1]

    for i, (start, end) in enumerate(sections):
        if offsets[i] != 0:
            mask = (out.index >= start) & (out.index <= end) & out.notna()
            out.loc[mask] += offsets[i]

    return out, offsets


def process(station_id, verbose=True):
    if verbose:
        print(f"\n  {station_id}")

    df = load(station_id)
    result = df.copy()

    for ch in CHANNELS:
        if ch not in df.columns:
            continue

        s = df[ch].copy()
        n_raw = int(s.notna().sum())

        # 1. Remove spikes (3 passes)
        s, n_spikes = remove_spikes(s)

        # 2. Segment
        secs = segment(s)

        # 3. Align using 7-day median at boundaries
        s, offsets = align(s, secs)

        # 4. One more spike pass after alignment (catches spikes revealed by shift)
        s, n_spikes2 = remove_spikes(s, passes=1)

        # 5. Interpolate small gaps
        before_nan = int(s.isna().sum())
        s = s.interpolate(method="time", limit=INTERP_LIMIT)
        n_interp = before_nan - int(s.isna().sum())

        result[ch] = s
        n_final = int(s.notna().sum())
        max_shift = max(abs(o) for o in offsets) if offsets else 0

        if verbose:
            print(f"    {ch}: {len(secs)} segs, {n_spikes+n_spikes2} spikes, "
                  f"max_shift={max_shift:.3f}, {n_final}/{n_raw} "
                  f"({100*n_final/len(s):.1f}%)")

    out_path = DATA_DIR / station_id / f"{station_id}_corrected.csv"
    result.to_csv(out_path)
    if verbose:
        print(f"    -> {out_path}")


def main():
    args = sys.argv[1:]
    if args:
        stations = args
    else:
        stations = sorted([
            p.name for p in DATA_DIR.iterdir()
            if p.is_dir() and p.name.startswith("KER")
            and (p / f"{p.name}_raw.csv").exists()
        ])

    print(f"Processing {len(stations)} stations")
    for st in stations:
        try:
            process(st)
        except Exception as e:
            print(f"  {st}: ERROR {e}")
            import traceback; traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()
