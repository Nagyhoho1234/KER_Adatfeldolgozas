"""
FastAPI backend for KER Groundwater Monitoring Dashboard.
Serves station metadata and time series data for 18 wells in Debrecen.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

# Load config once at startup
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

app = FastAPI(title="KER Groundwater Dashboard API")

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_station_list():
    """Build a flat list of stations from config."""
    stations = []
    for station_id, info in CONFIG["stations"].items():
        stations.append({
            "id": station_id,
            "code": info["code"],
            "name": info["name"],
            "lat": info["lat"],
            "lng": info["lng"],
        })
    # Sort by code for consistent ordering
    stations.sort(key=lambda s: s["code"])
    return stations


STATION_LIST = _build_station_list()
# Build code->info lookup
CODE_MAP = {info["code"]: {**info, "station_id": sid} for sid, info in CONFIG["stations"].items()}


def _find_csv(station_code: str) -> Optional[Path]:
    """Find the best available CSV: corrected > fixed > raw."""
    folder = DATA_DIR / station_code
    if not folder.exists():
        return None
    for suffix in ["_corrected.csv", "_fixed.csv", "_raw.csv"]:
        p = folder / f"{station_code}{suffix}"
        if p.exists():
            return p
    return None


@app.get("/api/stations")
def get_stations():
    """Return list of all stations with coordinates."""
    return STATION_LIST


@app.get("/api/timeseries/{station_code}")
def get_timeseries(
    station_code: str,
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Return time series data for a station (CH0, CH1, CH3)."""
    if station_code not in CODE_MAP:
        raise HTTPException(status_code=404, detail=f"Station {station_code} not found")

    csv_path = _find_csv(station_code)
    if csv_path is None:
        raise HTTPException(status_code=404, detail=f"No data file for {station_code}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

    # Date filtering
    if start:
        try:
            df = df[df.index >= pd.Timestamp(start)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start date")
    if end:
        try:
            df = df[df.index <= pd.Timestamp(end)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end date")

    # Only science channels
    channels = ["CH0", "CH1", "CH3"]
    result = {
        "station_code": station_code,
        "station_name": CODE_MAP[station_code]["name"],
        "source_file": csv_path.name,
        "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
    }

    for ch in channels:
        if ch in df.columns:
            # Convert to list, replacing NaN with None for JSON compatibility
            series = df[ch]
            result[ch] = [None if pd.isna(v) else float(v) for v in series]
        else:
            result[ch] = []

    return result


# Serve frontend static files (production mode)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        """Catch-all: serve index.html for SPA routing."""
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
