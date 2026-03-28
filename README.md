# KER Groundwater Monitoring Dashboard

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19291001.svg)](https://doi.org/10.5281/zenodo.19291001)

Automated data acquisition, quality control, and visualization system for 18 groundwater monitoring wells in Debrecen, Hungary (KER network).

## Overview

This system downloads hourly monitoring data from the DataQua EtherSense platform, applies automated time series correction (outlier removal and level-shift alignment), and serves a web dashboard for interactive visualization.

### Monitoring Network

18 stations in and around Debrecen, measuring:
- **Water level** (m) — piezometric head
- **Temperature** (°C) — groundwater temperature
- **Electrical conductivity** (mS/cm)

Data period: October 2024 – present, hourly resolution.

## Architecture

```
dataqua_downloader.py   → Downloads raw data from monitoring.dataqua.hu
ts_correction.py        → Quality control: ruptures PELT + Hampel filter
fix_specific.py         → Station-specific corrections (dual-instrument sites)
dashboard/api.py        → FastAPI backend serving JSON APIs
dashboard/frontend/     → React + MapLibre GL + Plotly.js web dashboard
```

## Quick Start

### Requirements

```
Python 3.10+
Node.js 18+
```

### Install dependencies

```bash
pip install pandas requests beautifulsoup4 lxml ruptures hampel fastapi uvicorn
cd dashboard/frontend && npm install && npm run build
```

### Download data

```bash
# Full download (all stations, from deployment)
python dataqua_downloader.py --start 2024-04-01

# Incremental update (only new data)
python dataqua_downloader.py --update

# Single station
python dataqua_downloader.py --station KER10 --start 2025-01-01
```

### Run quality control

```bash
# All stations (parallel on all CPU cores)
python ts_correction.py

# Station-specific fixes
python fix_specific.py KER02
```

### Start dashboard

```bash
cd dashboard && python api.py
# Open http://localhost:8000
```

The dashboard auto-updates when visited (max once per 24h): downloads new data, runs QC, refreshes charts.

## Time Series Quality Control

### Pipeline

1. **Hampel filter** — removes spikes > 0.1m deviation from local median (window=25h, 4 MADs)
2. **ruptures PELT** — detects level shifts (change points) using penalized exact linear time algorithm
3. **Segment alignment** — aligns segments using 7-day median at boundaries; last segment = reference
4. **Minimum shift threshold** — only corrects shifts > 5× daily variability (preserves natural trends)
5. **Post-alignment spike check** — second Hampel pass catches spikes revealed after alignment
6. **Gap interpolation** — linear interpolation for gaps ≤ 6 hours

### Station-specific

- **KER02**: Dual-instrument station with 3 calibration eras. Uses manual era boundaries for alignment due to known instrument swap dates.

## Data Structure

```
data/
  KER01/
    KER01_raw.csv           # Original downloaded data
    KER01_corrected.csv     # QC-corrected data
    KER01_meta.json         # Station metadata
  KER02/
    ...
config.json                 # Station catalog with coordinates and instrument IDs
```

## Dashboard

- **Left panel**: MapLibre GL map with 18 well markers + station list
- **Right panel**: Interactive Plotly charts (corrected water level, raw spike-cleaned water level, temperature, conductivity)
- **Auto-update**: Triggered on page visit, max once per 24h

## License

MIT License — see [LICENSE](LICENSE)

## Author

Fehér Zsolt Zoltán

## Citation

If you use this software, please cite:

> Fehér, Z. Z. (2026). KER Groundwater Monitoring Dashboard. https://doi.org/10.5281/zenodo.19291001
