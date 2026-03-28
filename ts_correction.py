"""
ts_correction.py — Using ruptures (PELT) + hampel filter.

Per channel:
  1. Resample to hourly
  2. Hampel filter: remove spikes (robust median-based)
  3. ruptures PELT: find all change points (level shifts)
  4. Align segments to last segment (7-day median at boundaries)
  5. Hampel again post-alignment (catch revealed spikes)
  6. Interpolate gaps ≤ 6h
"""

import sys
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from hampel import hampel
from pathlib import Path

warnings.filterwarnings("ignore")

CHANNELS = ["CH0", "CH1", "CH3"]
GAP_H = 3
HAMPEL_WINDOW = 25       # hours
HAMPEL_N_SIGMA = 4       # MADs for spike detection
PELT_PEN = 200           # PELT penalty — higher = fewer changepoints (only big shifts)
PELT_MIN_SIZE = 24       # minimum segment size (hours)
ALIGN_DAYS = 7           # 7-day median for boundary level estimation
INTERP_LIMIT = 6         # max gap to interpolate (hours)

DATA_DIR = Path(__file__).resolve().parent / "data"


def load(station_id):
    path = DATA_DIR / station_id / f"{station_id}_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    cols = [c for c in CHANNELS if c in df.columns]
    df = df[cols].dropna(how="all")
    return df.resample("1h").first()


def hampel_clean(s, window=HAMPEL_WINDOW, n_sigma=HAMPEL_N_SIGMA):
    """Apply Hampel filter to remove spikes. Returns cleaned series + count."""
    valid = s.dropna()
    if len(valid) < window * 2:
        return s, 0
    result = hampel(valid, window_size=window, n_sigma=float(n_sigma))
    cleaned = s.copy()
    outlier_idx = result.outlier_indices  # integer positions in valid
    n_changed = len(outlier_idx)
    if n_changed > 0:
        # Convert integer positions to actual timestamps
        outlier_timestamps = valid.index[outlier_idx]
        cleaned.loc[outlier_timestamps] = np.nan
    return cleaned, n_changed


def find_changepoints(s, pen=PELT_PEN, min_size=PELT_MIN_SIZE):
    """Use ruptures PELT to find level shifts. Returns list of breakpoint indices."""
    valid = s.dropna()
    if len(valid) < min_size * 3:
        return []

    signal = valid.values.reshape(-1, 1)
    algo = rpt.Pelt(model="l1", min_size=min_size, jump=5)
    try:
        result = algo.fit_predict(signal, pen=pen)
    except Exception:
        return []

    # result contains breakpoint indices (last element is len(signal))
    # Convert to timestamps
    breakpoints = []
    for bp in result[:-1]:  # skip the last one (end of signal)
        if 0 < bp < len(valid):
            breakpoints.append(valid.index[bp])
    return breakpoints


def segment_by_gaps_and_changepoints(s, gap_h=GAP_H, pen=PELT_PEN):
    """Segment by both data gaps AND detected changepoints."""
    valid = s.dropna()
    if len(valid) == 0:
        return []

    # Find gaps
    dt = valid.index.to_series().diff()
    gap_breaks = set(valid.index[dt > pd.Timedelta(hours=gap_h)])

    # Find changepoints
    cp_breaks = set(find_changepoints(s, pen=pen))

    # Combine all break points
    all_breaks = sorted(gap_breaks | cp_breaks)

    # Build sections
    sections = []
    start = valid.index[0]
    for bp in all_breaks:
        sec = valid.loc[start:bp]
        if bp in gap_breaks:
            sec = sec.iloc[:-1] if len(sec) > 1 else sec
        if len(sec) > 0:
            sections.append((sec.index[0], sec.index[-1]))
        start = bp
    sec = valid.loc[start:]
    if len(sec) > 0:
        sections.append((sec.index[0], sec.index[-1]))

    return sections


def align(s, sections, days=ALIGN_DAYS):
    """Align segments using 7-day median at boundaries.
    Last segment = reference. Walk backwards."""
    out = s.copy()
    n = len(sections)
    if n < 2:
        return out, []

    hrs = days * 24
    offsets = [0.0] * n

    for i in range(n - 2, -1, -1):
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


def process_channel(args):
    """Process a single channel for a single station. Designed for parallel execution."""
    station_id, ch = args
    df = load(station_id)
    if ch not in df.columns:
        return station_id, ch, None, ""

    s = df[ch].copy()
    n_raw = int(s.notna().sum())

    # 1. Hampel filter — remove spikes
    s, n_spikes1 = hampel_clean(s)

    # 2. Segment by gaps + PELT changepoints
    secs = segment_by_gaps_and_changepoints(s)

    # 3. Align using 7-day median
    s, offsets = align(s, secs)

    # 4. Hampel again post-alignment
    s, n_spikes2 = hampel_clean(s)

    # 5. Interpolate small gaps
    before_nan = int(s.isna().sum())
    s = s.interpolate(method="time", limit=INTERP_LIMIT)
    n_interp = before_nan - int(s.isna().sum())

    n_final = int(s.notna().sum())
    max_shift = max(abs(o) for o in offsets) if offsets else 0
    msg = (f"    {ch}: {len(secs)} segs, {n_spikes1+n_spikes2} spikes, "
           f"max_shift={max_shift:.3f}, {n_final}/{n_raw} "
           f"({100*n_final/len(s):.1f}%)")

    return station_id, ch, s, msg


def process(station_id, verbose=True):
    """Non-parallel fallback for single station."""
    if verbose:
        print(f"\n  {station_id}")
    df = load(station_id)
    result = df.copy()
    for ch in CHANNELS:
        _, _, s, msg = process_channel((station_id, ch))
        if s is not None:
            result[ch] = s
            if verbose:
                print(msg)
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

    # Build all (station, channel) tasks — 18 stations × 3 channels = 54 tasks
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(st, ch) for st in stations for ch in CHANNELS]
    print(f"Processing {len(tasks)} tasks ({len(stations)} stations × {len(CHANNELS)} channels) on 32 cores")

    results = {}  # station -> {ch: series}
    with ProcessPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(process_channel, t): t for t in tasks}
        for f in as_completed(futures):
            try:
                station_id, ch, s, msg = f.result()
                if s is not None:
                    if station_id not in results:
                        results[station_id] = {}
                    results[station_id][ch] = s
                    print(msg)
            except Exception as e:
                st, ch = futures[f]
                print(f"  {st}/{ch}: ERROR {e}")

    # Write results
    for st in stations:
        if st not in results:
            continue
        df = load(st)
        for ch, s in results[st].items():
            df[ch] = s
        out_path = DATA_DIR / st / f"{st}_corrected.csv"
        df.to_csv(out_path)
        print(f"  {st} -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
