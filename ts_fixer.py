"""
Time Series Fixer for DataQua monitoring data
==============================================
Detects and fixes common issues in downloaded time series:
- Duplicate timestamps
- Gaps (missing data periods)
- Irregular intervals
- Outliers / sensor errors (optional)
- Timezone/DST issues

Usage:
    python ts_fixer.py                        # Fix all stations in data/
    python ts_fixer.py --station KER10        # Fix specific station
    python ts_fixer.py --report               # Report issues without fixing
    python ts_fixer.py --interpolate          # Fill gaps with interpolation
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def load_station_data(station_dir):
    """Load raw CSV for a station."""
    csv_files = list(station_dir.glob("*_raw.csv"))
    if not csv_files:
        return None
    df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    return df


def detect_issues(df, expected_interval_minutes=60):
    """Analyze time series for problems. Returns a report dict."""
    report = {
        "total_rows": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())] if len(df) > 0 else None,
        "columns": list(df.columns),
        "issues": [],
    }

    if len(df) < 2:
        report["issues"].append({"type": "insufficient_data", "detail": f"Only {len(df)} rows"})
        return report

    # 1. Duplicate timestamps
    dupes = df.index.duplicated()
    n_dupes = dupes.sum()
    if n_dupes > 0:
        report["issues"].append({
            "type": "duplicate_timestamps",
            "count": int(n_dupes),
            "examples": [str(t) for t in df.index[dupes][:5]],
        })

    # 2. Non-monotonic (out-of-order timestamps)
    if not df.index.is_monotonic_increasing:
        n_unsorted = (df.index[1:] < df.index[:-1]).sum()
        report["issues"].append({
            "type": "non_monotonic",
            "count": int(n_unsorted),
        })

    # 3. Gaps - missing data periods
    expected_delta = timedelta(minutes=expected_interval_minutes)
    diffs = pd.Series(df.index[1:] - df.index[:-1])
    gap_threshold = expected_delta * 1.5
    gaps = diffs[diffs > gap_threshold]
    if len(gaps) > 0:
        gap_details = []
        for i, gap in gaps.items():
            gap_start = df.index[i]
            gap_end = df.index[i + 1]
            missing_steps = int(gap / expected_delta) - 1
            gap_details.append({
                "start": str(gap_start),
                "end": str(gap_end),
                "duration_hours": round(gap.total_seconds() / 3600, 1),
                "missing_steps": missing_steps,
            })
        report["issues"].append({
            "type": "gaps",
            "count": len(gaps),
            "total_missing_hours": round(sum(g["duration_hours"] for g in gap_details), 1),
            "largest_gap_hours": round(max(g["duration_hours"] for g in gap_details), 1),
            "details": gap_details[:20],  # limit output
        })

    # 4. Irregular intervals (not matching expected)
    unique_intervals = diffs.value_counts()
    dominant_interval = diffs.mode()[0] if len(diffs) > 0 else expected_delta
    irregular = diffs[(diffs != dominant_interval) & (diffs <= gap_threshold)]
    if len(irregular) > 0:
        report["issues"].append({
            "type": "irregular_intervals",
            "count": int(len(irregular)),
            "dominant_interval_minutes": round(dominant_interval.total_seconds() / 60, 1),
            "examples_minutes": [round(d.total_seconds() / 60, 1) for d in irregular.head(10)],
        })

    # 5. Per-column stats and potential outliers
    col_stats = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats = {
            "count": int(len(s)),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().mean() * 100, 1),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
        }
        # Flag potential outliers (beyond 4 sigma)
        if s.std() > 0:
            z = (s - s.mean()) / s.std()
            n_outliers = int((z.abs() > 4).sum())
            if n_outliers > 0:
                stats["outliers_4sigma"] = n_outliers
        col_stats[col] = stats
    report["column_stats"] = col_stats

    # 6. Constant value runs (sensor stuck)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        # Find runs of identical values
        changes = s.diff().ne(0)
        runs = changes.cumsum()
        run_lengths = runs.value_counts()
        long_runs = run_lengths[run_lengths > 24]  # >24 hours of constant value
        if len(long_runs) > 0:
            report["issues"].append({
                "type": "constant_runs",
                "column": col,
                "longest_run": int(long_runs.max()),
                "count": int(len(long_runs)),
            })

    return report


def fix_timeseries(df, expected_interval_minutes=60, interpolate=False):
    """Fix common time series issues. Returns cleaned DataFrame and log."""
    log = []

    # 1. Sort by timestamp
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        log.append("Sorted timestamps")

    # 2. Remove duplicate timestamps (keep last value)
    n_dupes = df.index.duplicated(keep="last").sum()
    if n_dupes > 0:
        df = df[~df.index.duplicated(keep="last")]
        log.append(f"Removed {n_dupes} duplicate timestamps")

    # 3. Reindex to regular interval (fills NaN for missing timestamps)
    expected_delta = timedelta(minutes=expected_interval_minutes)
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{expected_interval_minutes}min",
    )
    n_before = len(df)
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    n_added = len(df) - n_before
    if n_added > 0:
        log.append(f"Regularized to {expected_interval_minutes}-min intervals (+{n_added} NaN rows for gaps)")

    # 4. Optional: interpolate short gaps
    if interpolate:
        max_gap = 6  # max 6 consecutive NaN to interpolate (6 hours at hourly)
        for col in df.columns:
            n_before_interp = df[col].isna().sum()
            df[col] = df[col].interpolate(method="linear", limit=max_gap)
            n_filled = n_before_interp - df[col].isna().sum()
            if n_filled > 0:
                log.append(f"Interpolated {n_filled} values in {col} (max gap: {max_gap} steps)")

    return df, log


def process_station(station_dir, report_only=False, interpolate=False):
    """Process one station directory."""
    code = station_dir.name
    df = load_station_data(station_dir)

    if df is None:
        print(f"  [{code}] No data found")
        return

    print(f"  [{code}] {len(df)} rows, {len(df.columns)} columns, "
          f"{df.index.min()} to {df.index.max()}")

    # Detect dominant interval
    if len(df) > 2:
        diffs = pd.Series(df.index[1:] - df.index[:-1])
        dominant = diffs.mode()[0]
        interval_min = int(dominant.total_seconds() / 60)
    else:
        interval_min = 60

    # Report
    report = detect_issues(df, expected_interval_minutes=interval_min)
    issues = report["issues"]

    if issues:
        print(f"    Issues found: {len(issues)}")
        for issue in issues:
            itype = issue["type"]
            if itype == "duplicate_timestamps":
                print(f"      - {issue['count']} duplicate timestamps")
            elif itype == "non_monotonic":
                print(f"      - {issue['count']} out-of-order timestamps")
            elif itype == "gaps":
                print(f"      - {issue['count']} gaps (total {issue['total_missing_hours']}h, "
                      f"largest {issue['largest_gap_hours']}h)")
            elif itype == "irregular_intervals":
                print(f"      - {issue['count']} irregular intervals "
                      f"(dominant: {issue['dominant_interval_minutes']}min)")
            elif itype == "constant_runs":
                print(f"      - Constant value runs in {issue['column']} "
                      f"(longest: {issue['longest_run']})")
            elif itype == "insufficient_data":
                print(f"      - {issue['detail']}")
    else:
        print(f"    No issues detected")

    # Save report
    report_path = station_dir / f"{code}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if report_only:
        return

    # Fix
    if issues:
        df_fixed, fix_log = fix_timeseries(df, expected_interval_minutes=interval_min,
                                            interpolate=interpolate)
        if fix_log:
            print(f"    Fixes applied:")
            for entry in fix_log:
                print(f"      - {entry}")

            # Save fixed version
            fixed_path = station_dir / f"{code}_fixed.csv"
            df_fixed.to_csv(fixed_path, encoding="utf-8")
            print(f"    Saved: {fixed_path}")


def main():
    parser = argparse.ArgumentParser(description="Time Series Fixer for DataQua data")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--station", nargs="*", help="Station code(s) to process")
    parser.add_argument("--report", action="store_true", help="Report only, don't fix")
    parser.add_argument("--interpolate", action="store_true",
                        help="Interpolate short gaps (up to 6 hours)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Find station directories
    station_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    if args.station:
        station_dirs = [d for d in station_dirs if d.name in args.station]

    if not station_dirs:
        print("No station directories found!")
        sys.exit(1)

    mode = "Report" if args.report else "Fix"
    print(f"Time Series {mode} - {len(station_dirs)} station(s)\n")

    for sd in station_dirs:
        process_station(sd, report_only=args.report, interpolate=args.interpolate)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
