"""
DataQua EtherSense Monitoring Data Downloader
=============================================
Downloads time series data from all configured stations.

Usage:
    python dataqua_downloader.py                          # Download last 30 days
    python dataqua_downloader.py --start 2024-06-01       # From date to today
    python dataqua_downloader.py --start 2024-06-01 --end 2024-12-31  # Date range
    python dataqua_downloader.py --update                 # Only download new data since last download
    python dataqua_downloader.py --station KER10           # Single station only
    python dataqua_downloader.py --station KER10 KER15     # Multiple stations
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()


def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_session(config):
    """Login and return authenticated session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    base = config["base_url"]
    creds = config["credentials"]
    session.post(f"{base}/login.php", data={
        "userlogin": creds["userlogin"],
        "password": creds["password"],
        "loginuser": "Belépés",
    }, verify=False)
    # Login redirects via JS; verify by checking index page
    r = session.get(f"{base}/index.php", verify=False)
    if "Kijelentkez" not in r.text:
        raise RuntimeError("Login failed!")
    return session


def parse_txt_export(raw_text):
    """Parse the DataQua .txt export format into a dict of DataFrames per channel."""
    channels = {}
    # Split by channel blocks
    blocks = re.split(r"'CSATORNA:\s*", raw_text)

    for block in blocks[1:]:  # skip header before first channel
        # Parse channel header
        lines = block.strip().split("\n")
        ch_id = lines[0].strip().rstrip(".")

        # Parse metadata
        meta = {}
        data_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line.startswith("SRSZ.:"):
                data_start = i + 1
                break
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip().strip("\t")
                if "dimenzió" in key or "fizikai" in key:
                    meta["unit"] = val
                elif "start" in key:
                    meta["start"] = val
                elif "stop" in key:
                    meta["stop"] = val
                elif "adatok száma" in key:
                    meta["count"] = int(val)
                elif "ciklus" in key:
                    meta["interval"] = val

        # Parse data rows
        timestamps = []
        values = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("'") or line.startswith("="):
                break
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    ts = pd.to_datetime(parts[1].strip())
                    val = float(parts[2].strip())
                    timestamps.append(ts)
                    values.append(val)
                except (ValueError, IndexError):
                    continue

        if timestamps:
            df = pd.DataFrame({
                "timestamp": timestamps,
                f"CH{ch_id}": values
            })
            df.set_index("timestamp", inplace=True)
            channels[ch_id] = {"data": df, "meta": meta}

    return channels


def download_station(session, config, meplid, station_info, start_date, end_date):
    """Download all data for one station, return dict of channel DataFrames."""
    base = config["base_url"]
    fmt = config.get("export_format", "txt")

    results = {}
    for inst in station_info["instruments"]:
        instid = inst["instid"]
        if not inst.get("channels"):
            continue

        # First visit the report page (establishes session context)
        session.get(f"{base}/report.php?meplid={meplid}&instid={instid}", verify=False)

        # Request export
        export_data = {
            "format": fmt,
            "start": start_date,
            "end": end_date,
            "instid": instid,
            "meplid": meplid,
            "save": "Mentés",
        }
        r = session.post(f"{base}/export.php", data=export_data, verify=False)

        # Extract download URL from JS response
        m = re.search(r"document\.location\.href\s*=\s*'([^']+)'", r.text)
        if not m:
            print(f"    WARNING: No download URL for {station_info['code']} instid={instid}")
            continue

        download_url = m.group(1)
        r2 = session.get(f"{base}/{download_url}", verify=False)

        if r2.status_code != 200:
            print(f"    WARNING: Download failed for {station_info['code']} ({r2.status_code})")
            continue

        # Decode content
        try:
            raw = r2.content.decode("utf-8")
        except UnicodeDecodeError:
            raw = r2.content.decode("iso-8859-2", errors="replace")

        # Parse channels
        channels = parse_txt_export(raw)
        results[instid] = channels

    return results


def save_station_data(station_code, results, output_dir, station_info):
    """Save parsed data to CSV files, one per station with all channels merged."""
    station_dir = Path(output_dir) / station_code
    station_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    channel_meta = {}

    # Group DataFrames by column name — multiple instruments may share columns
    col_dfs = {}  # column_name -> list of Series
    for instid, channels in results.items():
        for ch_id, ch_data in channels.items():
            df = ch_data["data"]
            for col in df.columns:
                if col not in col_dfs:
                    col_dfs[col] = []
                col_dfs[col].append(df[col])
            channel_meta[f"CH{ch_id}"] = ch_data["meta"]

    if not col_dfs:
        return None

    # For each column, combine all series (older first, newer overwrites overlaps)
    merged_cols = {}
    for col, series_list in col_dfs.items():
        combined = series_list[0]
        for s in series_list[1:]:
            combined = combined.combine_first(s)
        merged_cols[col] = combined

    merged = pd.DataFrame(merged_cols)

    # Sort by timestamp
    merged.sort_index(inplace=True)

    # Save merged CSV
    out_path = station_dir / f"{station_code}_raw.csv"
    merged.to_csv(out_path, encoding="utf-8")

    # Save metadata
    meta_path = station_dir / f"{station_code}_meta.json"
    meta = {
        "station_code": station_code,
        "station_name": station_info["name"],
        "lat": station_info.get("lat"),
        "lng": station_info.get("lng"),
        "eovx": station_info.get("eovx"),
        "eovy": station_info.get("eovy"),
        "channels": channel_meta,
        "download_time": datetime.now().isoformat(),
        "rows": len(merged),
        "date_range": [str(merged.index.min()), str(merged.index.max())] if len(merged) > 0 else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_path


def get_last_download_date(station_code, output_dir):
    """Check existing data to find the last timestamp for incremental updates."""
    csv_path = Path(output_dir) / station_code / f"{station_code}_raw.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True, nrows=0)
            # Read just the last few rows efficiently
            df_tail = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if len(df_tail) > 0:
                return df_tail.index.max().strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def merge_update(station_code, new_results, output_dir, station_info):
    """Merge new data into existing CSV (for --update mode)."""
    csv_path = Path(output_dir) / station_code / f"{station_code}_raw.csv"

    # Build new DataFrame — handle duplicate columns from multiple instruments
    col_dfs = {}
    for instid, channels in new_results.items():
        for ch_id, ch_data in channels.items():
            df = ch_data["data"]
            for col in df.columns:
                if col not in col_dfs:
                    col_dfs[col] = []
                col_dfs[col].append(df[col])

    if not col_dfs:
        return None

    merged_cols = {}
    for col, series_list in col_dfs.items():
        combined = series_list[0]
        for s in series_list[1:]:
            combined = combined.combine_first(s)
        merged_cols[col] = combined
    new_merged = pd.DataFrame(merged_cols)
    new_merged.sort_index(inplace=True)

    if csv_path.exists():
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Combine: new data overwrites existing for overlapping timestamps
        combined = pd.concat([existing, new_merged])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_merged

    station_dir = Path(output_dir) / station_code
    station_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, encoding="utf-8")

    # Update metadata
    meta_path = station_dir / f"{station_code}_meta.json"
    meta = {
        "station_code": station_code,
        "station_name": station_info["name"],
        "lat": station_info.get("lat"),
        "lng": station_info.get("lng"),
        "eovx": station_info.get("eovx"),
        "eovy": station_info.get("eovy"),
        "download_time": datetime.now().isoformat(),
        "rows": len(combined),
        "date_range": [str(combined.index.min()), str(combined.index.max())] if len(combined) > 0 else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="DataQua Monitoring Data Downloader")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--update", action="store_true",
                        help="Incremental update: only download new data since last download")
    parser.add_argument("--station", nargs="*",
                        help="Station code(s) to download (e.g. KER10 KER15). Default: all")
    parser.add_argument("--output", help="Output directory (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output or config.get("output_dir", "data")

    # Determine date range
    if args.update:
        end_date = datetime.now().strftime("%Y-%m-%d")
    elif args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start:
        default_start = args.start
    elif not args.update:
        default_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        default_start = None  # will be per-station for update mode

    # Filter stations
    stations_to_process = {}
    for meplid, info in config["stations"].items():
        if args.station is None or info["code"] in args.station:
            stations_to_process[meplid] = info

    if not stations_to_process:
        print("No matching stations found!")
        sys.exit(1)

    print(f"DataQua Downloader - {len(stations_to_process)} station(s)")
    print(f"Output: {output_dir}/")

    # Login
    print("Logging in...")
    session = create_session(config)
    print("Login OK\n")

    # Download each station
    success = 0
    errors = 0
    for meplid, info in stations_to_process.items():
        code = info["code"]

        # Determine start date for this station
        if args.update:
            last_date = get_last_download_date(code, output_dir)
            if last_date:
                start_date = last_date
                print(f"[{code}] Updating from {start_date} to {end_date}...")
            else:
                start_date = default_start or "2024-04-01"
                print(f"[{code}] No existing data, full download from {start_date} to {end_date}...")
        else:
            start_date = default_start
            print(f"[{code}] Downloading {start_date} to {end_date}...")

        try:
            results = download_station(session, config, meplid, info, start_date, end_date)

            if args.update:
                out_path = merge_update(code, results, output_dir, info)
            else:
                out_path = save_station_data(code, results, output_dir, info)

            if out_path:
                # Count rows
                df = pd.read_csv(out_path, index_col=0, nrows=1)
                total = sum(
                    ch_data["data"].shape[0]
                    for inst_channels in results.values()
                    for ch_data in inst_channels.values()
                )
                print(f"  -> {out_path} ({total} records)")
                success += 1
            else:
                print(f"  -> No data returned")
                errors += 1

        except Exception as e:
            print(f"  -> ERROR: {e}")
            errors += 1

        # Small delay to be polite to the server
        time.sleep(0.5)

    print(f"\nDone! {success} stations downloaded, {errors} errors.")


if __name__ == "__main__":
    main()
