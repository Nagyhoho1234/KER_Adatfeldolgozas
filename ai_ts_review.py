"""
AI-assisted time series review and correction.
Uses Claude to analyze each station's corrected data, identify remaining
artifacts (spikes, shifts, bad segments), and apply targeted fixes.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from anthropic import Anthropic

DATA_DIR = Path("data")
CHANNELS = ["CH0", "CH1", "CH3"]
CHANNEL_NAMES = {"CH0": "Conductivity (mS/cm)", "CH1": "Temperature (°C)", "CH3": "Water level (m)"}

client = Anthropic()


def summarize_ts(series: pd.Series, name: str) -> str:
    """Create a compact text summary of a time series for the AI."""
    s = series.dropna()
    if len(s) == 0:
        return f"{name}: no data"

    # Basic stats
    lines = [f"{name}: {len(s)} pts, {s.index.min()} to {s.index.max()}"]
    lines.append(f"  range: [{s.min():.4f}, {s.max():.4f}], mean={s.mean():.4f}, std={s.std():.4f}")

    # Resample to daily for compact representation
    daily = s.resample("1D").median().dropna()
    # Show weekly values for long series
    weekly = s.resample("7D").median().dropna()
    lines.append(f"  Weekly medians ({len(weekly)} weeks):")
    vals = [f"{v:.4f}" for v in weekly.values]
    # Show in rows of 10
    for i in range(0, len(vals), 10):
        chunk = vals[i:i+10]
        dates = weekly.index[i:i+min(10, len(vals)-i)]
        lines.append(f"    {dates[0].strftime('%Y-%m-%d')}: {', '.join(chunk)}")

    # Show biggest jumps (daily)
    if len(daily) > 1:
        d = daily.diff().dropna()
        big = d.abs().nlargest(10)
        lines.append(f"  10 biggest daily jumps:")
        for ts, v in big.items():
            before = daily.loc[:ts].iloc[-2] if len(daily.loc[:ts]) >= 2 else np.nan
            after = daily.loc[ts]
            lines.append(f"    {ts.strftime('%Y-%m-%d')}: {v:+.4f} ({before:.4f} -> {after:.4f})")

    return "\n".join(lines)


def ai_analyze_station(station_id: str) -> dict:
    """Use Claude to analyze a station's corrected time series."""
    csv_path = DATA_DIR / station_id / f"{station_id}_corrected.csv"
    if not csv_path.exists():
        return {"error": f"No corrected data for {station_id}"}

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    # Build summary for each channel
    summaries = []
    for ch in CHANNELS:
        if ch in df.columns:
            summaries.append(summarize_ts(df[ch], f"{ch} ({CHANNEL_NAMES.get(ch, ch)})"))

    prompt = f"""You are analyzing corrected groundwater monitoring time series for station {station_id}.
The data has already been through automated correction (outlier removal, level-shift alignment,
gap interpolation). Your job is to find REMAINING artifacts that the automated correction missed.

For each channel, I'll show you weekly median values and the biggest daily jumps.

IMPORTANT: These are groundwater observations. Water level changes slowly and smoothly (daily
changes typically <0.01m). Temperature is nearly constant in deep wells or follows slow seasonal
patterns. Conductivity is stable with very slow trends.

KER16 and KER17 are Tócó surface water stations — they have naturally higher variability.

Here are the summaries:

{chr(10).join(summaries)}

Analyze each channel and identify:
1. Remaining SPIKES: sudden jumps that reverse (not natural behavior)
2. Remaining SHIFTS: step changes where the level suddenly changes permanently
3. BAD SEGMENTS: periods where values are clearly wrong (e.g., constant at a wrong level)

For each issue, specify:
- Channel (CH0, CH1, or CH3)
- Start datetime (YYYY-MM-DD HH:MM)
- End datetime
- Type: "spike", "shift", or "bad_segment"
- Action: "remove" (set to NaN) or "shift" (adjust level)
- If shift: the approximate offset to apply

Respond with ONLY a JSON array of issues. If no issues, return [].
Example:
[
  {{"channel": "CH3", "start": "2025-08-08 10:00", "end": "2025-08-08 14:00", "type": "spike", "action": "remove"}},
  {{"channel": "CH1", "start": "2025-09-15 12:00", "end": "2025-09-15 18:00", "type": "bad_segment", "action": "remove"}}
]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    text = response.content[0].text.strip()
    # Find JSON array in response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            issues = json.loads(text[start:end])
            return {"station": station_id, "issues": issues}
        except json.JSONDecodeError:
            return {"station": station_id, "error": "Failed to parse AI response", "raw": text}
    return {"station": station_id, "issues": [], "raw": text}


def apply_fixes(station_id: str, issues: list) -> int:
    """Apply AI-identified fixes to the corrected CSV."""
    csv_path = DATA_DIR / station_id / f"{station_id}_corrected.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    n_fixed = 0
    for issue in issues:
        ch = issue["channel"]
        if ch not in df.columns:
            continue
        start = pd.Timestamp(issue["start"])
        end = pd.Timestamp(issue["end"])
        action = issue["action"]

        mask = (df.index >= start) & (df.index <= end)
        n_points = mask.sum()

        if action == "remove":
            df.loc[mask, ch] = np.nan
            n_fixed += n_points
            print(f"  {ch} {start} to {end}: removed {n_points} points ({issue['type']})")
        elif action == "shift":
            offset = issue.get("offset", 0)
            if offset != 0:
                df.loc[mask, ch] += offset
                n_fixed += n_points
                print(f"  {ch} {start} to {end}: shifted {n_points} points by {offset}")

    if n_fixed > 0:
        # Interpolate small gaps created by removals
        for ch in CHANNELS:
            if ch in df.columns:
                df[ch] = df[ch].interpolate(method="time", limit=6)
        df.to_csv(csv_path)

    return n_fixed


def main():
    stations_arg = sys.argv[1:] if len(sys.argv) > 1 else None

    if stations_arg:
        stations = stations_arg
    else:
        stations = sorted([
            p.name for p in DATA_DIR.iterdir()
            if p.is_dir() and p.name.startswith("KER")
            and (p / f"{p.name}_corrected.csv").exists()
        ])

    print(f"AI Review: {len(stations)} station(s)\n")

    all_issues = {}
    for st in stations:
        print(f"Analyzing {st}...")
        result = ai_analyze_station(st)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        issues = result["issues"]
        if not issues:
            print(f"  No issues found")
            continue

        print(f"  Found {len(issues)} issue(s):")
        for iss in issues:
            print(f"    {iss['channel']} {iss['type']}: {iss['start']} to {iss['end']} -> {iss['action']}")

        all_issues[st] = issues

    # Ask to apply
    if all_issues:
        print(f"\n{'='*60}")
        print(f"Total: {sum(len(v) for v in all_issues.values())} issues across {len(all_issues)} stations")
        print(f"Applying fixes...")
        for st, issues in all_issues.items():
            print(f"\n  {st}:")
            n = apply_fixes(st, issues)
            print(f"  -> {n} points fixed")

    print("\nDone.")


if __name__ == "__main__":
    main()
