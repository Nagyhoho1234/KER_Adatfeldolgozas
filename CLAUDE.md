# CLAUDE.md

## Project Overview

KER Groundwater Monitoring Dashboard for 18 wells in Debrecen, Hungary. Downloads data from DataQua EtherSense, applies QC (ruptures PELT + Hampel filter), serves a React+MapLibre+Plotly dashboard via FastAPI.

## Key Files

- `dataqua_downloader.py` — Scrapes monitoring.dataqua.hu (login required, browser UA needed)
- `ts_correction.py` — QC pipeline: Hampel spike removal + PELT changepoint detection + segment alignment. Runs 54 tasks (18×3) in parallel.
- `fix_specific.py` — KER02 dual-instrument alignment (3 calibration eras)
- `dashboard/api.py` — FastAPI backend with auto-update on visit (max 24h)
- `dashboard/frontend/` — React + Vite SPA
- `config.json` — Station catalog (meplid, instid, coordinates, channels)

## Important Decisions

- **Spike threshold**: Only remove deviations >= 0.1m. Smaller variations are real groundwater dynamics.
- **PELT penalty**: 200. Lower = too many changepoints (flattens real trends). Higher = misses real shifts.
- **Alignment threshold**: Only correct shifts > 5× daily MAD (minimum 0.05). Preserves natural variability.
- **Last segment = reference** for alignment. The most recent sensor reading is considered truth.
- **KER02** has two instruments with different calibrations. fix_specific.py handles this with manual era boundaries. When instruments change, update the era_splits_per_ch dict.
- **KER16, KER17** are Tócó surface water stations — higher natural variability, noisier signals.

## Running

```bash
python ts_correction.py          # All stations, parallel
python fix_specific.py KER02     # After generic correction
python dashboard/api.py          # Serves on :8000
```

## Credentials

DataQua EtherSense credentials are loaded from a `.env` file in the project root (gitignored):

```
DATAQUA_USER=your_username
DATAQUA_PASS=your_password
```

The downloader (`dataqua_downloader.py`) reads these via `os.environ`, falling back to `config.json` (which has empty placeholders). Never commit real credentials to the repo.
