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

    # Define era boundaries per channel (from raw data analysis)
    # CH3 has extra intra-instrument shifts in Jun and Aug 2025
    # CH1 switches at Sep 15 14:00, CH0/CH3 at Sep 16 00:00
    era_splits_per_ch = {
        "CH0": [pd.Timestamp("2025-09-16"), pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-19")],
        "CH1": [pd.Timestamp("2025-09-15 14:00"), pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-19")],
        "CH3": [pd.Timestamp("2025-06-17"), pd.Timestamp("2025-08-08"),
                pd.Timestamp("2025-09-08 12:00"), pd.Timestamp("2025-09-16"),
                pd.Timestamp("2025-11-26"), pd.Timestamp("2026-02-19")],
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
        #    The values jump from ~1.34 to ~3.4, clearly sensor failure
        if ch == "CH0":
            stable_period = s['2026-02-21':'2026-03-18'].dropna()
            if len(stable_period) > 24:
                stable_med = stable_period.median()
                stable_mad = (stable_period - stable_med).abs().median()
                if stable_mad == 0:
                    stable_mad = 0.01
                # Remove anything after the stable period that deviates >5*MAD
                for ts in s['2026-03-19':].index:
                    if pd.notna(s[ts]) and abs(s[ts] - stable_med) > 5 * stable_mad:
                        s[ts] = np.nan

        # 3. Multi-era alignment: chain from last era backwards
        #    Match each era's END to the next era's START
        era_splits = era_splits_per_ch.get(ch, era_splits_per_ch["CH3"])
        boundaries = [s.index.min()] + era_splits + [s.index.max() + pd.Timedelta(hours=1)]
        eras = []
        for i in range(len(boundaries) - 1):
            era = s[boundaries[i]:boundaries[i+1]]
            era_valid = era.dropna()
            if len(era_valid) >= 6:
                start_level = era_valid.iloc[:min(24, len(era_valid))].median()
                end_level = era_valid.iloc[-min(24, len(era_valid)):].median()
                eras.append({"start": boundaries[i], "end": boundaries[i+1],
                            "start_level": start_level, "end_level": end_level,
                            "n": len(era_valid)})

        if len(eras) >= 2:
            # Walk backwards: last era = reference (offset 0)
            offsets = [0.0] * len(eras)
            for i in range(len(eras) - 2, -1, -1):
                # Match this era's END to next era's START (after next is shifted)
                this_end = eras[i]["end_level"]
                next_start = eras[i + 1]["start_level"]
                offsets[i] = (next_start - this_end) + offsets[i + 1]

            for i, era in enumerate(eras):
                if offsets[i] != 0:
                    mask = (s.index >= era["start"]) & (s.index < era["end"])
                    s.loc[mask & s.notna()] += offsets[i]
                    if verbose:
                        print(f"    {ch} era {era['start'].strftime('%Y-%m')} to "
                              f"{era['end'].strftime('%Y-%m')}: offset {offsets[i]:+.4f}")

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
