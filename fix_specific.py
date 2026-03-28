"""
Station-specific corrections applied AFTER ts_correction.py.
These handle cases where the automated algorithm makes things worse
due to known instrument issues.

Usage:
    python fix_specific.py          # Apply all fixes
    python fix_specific.py KER02    # Fix specific station
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def simple_two_block_align(series, split_date, ref="last"):
    """Align two blocks of data at a known split point.
    ref='last' means the later block is the reference."""
    s = series.copy()
    before = s[:split_date].dropna()
    after = s[split_date:].dropna()

    if len(before) < 6 or len(after) < 6:
        return s

    # Use median of last/first 48 hours for stable level estimate
    before_level = before.iloc[-min(48, len(before)):].median()
    after_level = after.iloc[:min(48, len(after))].median()

    offset = after_level - before_level
    if ref == "last":
        # Shift earlier block to match later
        s.loc[:split_date] += offset
    else:
        s.loc[split_date:] -= offset

    return s


def fix_ker02(verbose=True):
    """KER02: dual instrument station with 3 eras of different calibrations.
    Era 1: Oct 2024 - Sep 15 2025 (old instrument, CH3 ~ -9.5m)
    Era 2: Sep 16 - Nov 25 2025 (new instrument, CH3 ~ -6.5m)
    Era 3: Feb 21 - Mar 28 2026 (new instrument, CH3 ~ -9.6m)

    The automated correction fails because changepoint detection
    over-fragments the nearly-constant signals.
    Fix: start from raw, simple outlier removal, multi-block alignment."""

    raw_path = DATA_DIR / "KER02" / "KER02_raw.csv"
    out_path = DATA_DIR / "KER02" / "KER02_corrected.csv"

    df = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    channels = ["CH0", "CH1", "CH3"]
    df = df[[c for c in channels if c in df.columns]].copy()
    df = df.dropna(how="all")
    df = df.resample("1h").first()

    # Define era boundaries per channel (detected from raw data analysis)
    # CH3 switches at Sep 16 00:00, CH1 switches at Sep 15 14:00, CH0 at Sep 16
    era_splits_per_ch = {
        "CH0": [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-21")],
        "CH1": [pd.Timestamp("2025-09-15 14:00"), pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-21")],
        "CH3": [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-21")],
    }

    for ch in channels:
        if ch not in df.columns:
            continue
        s = df[ch].copy()

        # 1. Simple outlier removal: rolling median ± 5*MAD
        roll_med = s.rolling(25, center=True, min_periods=3).median()
        dev = (s - roll_med).abs()
        roll_mad = dev.rolling(25, center=True, min_periods=3).median()
        roll_mad = roll_mad.replace(0, np.nan).fillna(dev.median())
        if roll_mad.max() > 0:
            s[dev > 5 * roll_mad] = np.nan

        # 2. Remove CH0 end spike (Mar 19+ conductivity sensor failure)
        if ch == "CH0":
            end_vals = s['2026-03-18':].dropna()
            if len(end_vals) > 48:
                stable = end_vals.iloc[:48]
                stable_med = stable.median()
                stable_mad = (stable - stable_med).abs().median()
                if stable_mad > 0:
                    for ts in end_vals.index:
                        if pd.notna(s[ts]) and abs(s[ts] - stable_med) > 10 * stable_mad:
                            s[ts] = np.nan

        # 3. Multi-era alignment: split into eras, align to last era
        era_splits = era_splits_per_ch.get(ch, era_splits_per_ch["CH3"])
        boundaries = [s.index.min()] + era_splits + [s.index.max() + pd.Timedelta(hours=1)]
        eras = []
        for i in range(len(boundaries) - 1):
            era = s[boundaries[i]:boundaries[i+1]]
            era_valid = era.dropna()
            if len(era_valid) >= 6:
                level = era_valid.iloc[:min(48, len(era_valid))].median()
                eras.append({"start": boundaries[i], "end": boundaries[i+1],
                            "level": level, "n": len(era_valid)})

        if len(eras) >= 2:
            # Last era is reference
            ref_level = eras[-1]["level"]
            for era in eras[:-1]:
                offset = ref_level - era["level"]
                mask = (s.index >= era["start"]) & (s.index < era["end"])
                s.loc[mask & s.notna()] += offset
                if verbose:
                    print(f"    {ch} era {era['start'].strftime('%Y-%m')} to "
                          f"{era['end'].strftime('%Y-%m')}: offset {offset:+.3f}")

        # 4. Interpolate small gaps
        s = s.interpolate(method="time", limit=6)

        df[ch] = s

        if verbose:
            valid = s.dropna()
            print(f"  {ch}: {len(valid)} valid, range [{valid.min():.3f}, {valid.max():.3f}]")

    df.to_csv(out_path)
    if verbose:
        print(f"  Saved -> {out_path}")


def fix_station(station_id, verbose=True):
    if station_id == "KER02":
        fix_ker02(verbose)
    else:
        if verbose:
            print(f"  No specific fix for {station_id}")


def main():
    stations = sys.argv[1:] if len(sys.argv) > 1 else ["KER02"]

    for st in stations:
        print(f"\nFixing {st}...")
        fix_station(st, verbose=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
