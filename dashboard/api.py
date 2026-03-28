"""
FastAPI backend for KER Groundwater Monitoring Dashboard.
Serves station metadata and time series data for 18 wells in Debrecen.
Supports background data updates triggered by frontend visits.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
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
STATE_FILE = BASE_DIR / "data" / ".update_state.json"
PYTHON = sys.executable

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

# ---- Update state management ----

_update_lock = threading.Lock()
_update_state = {
    "status": "idle",        # idle | downloading | correcting | done | error
    "last_update": None,     # ISO timestamp of last successful update
    "last_check": None,      # ISO timestamp of last check
    "progress": "",          # human-readable progress message
    "error": None,
    "stations_done": 0,
    "stations_total": 18,
}


def _load_persisted_state():
    """Load last_update timestamp from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            _update_state["last_update"] = saved.get("last_update")
        except Exception:
            pass


def _save_persisted_state():
    """Persist last_update to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_update": _update_state["last_update"]}, f)


_load_persisted_state()


def _run_update():
    """Background thread: download new data then run correction."""
    try:
        with _update_lock:
            _update_state["status"] = "downloading"
            _update_state["progress"] = "Downloading new observations..."
            _update_state["stations_done"] = 0
            _update_state["error"] = None

        # Step 1: Download (--update mode: only new data since last download)
        proc = subprocess.run(
            [PYTHON, str(BASE_DIR / "dataqua_downloader.py"), "--update"],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=600,
        )

        if proc.returncode != 0:
            with _update_lock:
                _update_state["status"] = "error"
                _update_state["error"] = f"Download failed: {proc.stderr[-500:]}"
            return

        with _update_lock:
            _update_state["status"] = "correcting"
            _update_state["progress"] = "Correcting time series..."

        # Step 2: Run TS correction
        proc = subprocess.run(
            [PYTHON, str(BASE_DIR / "ts_correction.py")],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=600,
        )

        if proc.returncode != 0:
            with _update_lock:
                _update_state["status"] = "error"
                _update_state["error"] = f"Correction failed: {proc.stderr[-500:]}"
            return

        with _update_lock:
            now = datetime.now().isoformat()
            _update_state["status"] = "done"
            _update_state["progress"] = "Update complete"
            _update_state["last_update"] = now
            _update_state["stations_done"] = _update_state["stations_total"]
            _save_persisted_state()

    except subprocess.TimeoutExpired:
        with _update_lock:
            _update_state["status"] = "error"
            _update_state["error"] = "Update timed out (>10 min)"
    except Exception as e:
        with _update_lock:
            _update_state["status"] = "error"
            _update_state["error"] = str(e)


# ---- Station helpers ----

def _build_station_list():
    stations = []
    for station_id, info in CONFIG["stations"].items():
        stations.append({
            "id": station_id,
            "code": info["code"],
            "name": info["name"],
            "lat": info["lat"],
            "lng": info["lng"],
        })
    stations.sort(key=lambda s: s["code"])
    return stations


STATION_LIST = _build_station_list()
CODE_MAP = {info["code"]: {**info, "station_id": sid} for sid, info in CONFIG["stations"].items()}


def _find_csv(station_code: str) -> Optional[Path]:
    folder = DATA_DIR / station_code
    if not folder.exists():
        return None
    for suffix in ["_corrected.csv", "_fixed.csv", "_raw.csv"]:
        p = folder / f"{station_code}{suffix}"
        if p.exists():
            return p
    return None


# ---- API endpoints ----

@app.get("/api/stations")
def get_stations():
    return STATION_LIST


@app.get("/api/timeseries/{station_code}")
def get_timeseries(
    station_code: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    if station_code not in CODE_MAP:
        raise HTTPException(status_code=404, detail=f"Station {station_code} not found")

    csv_path = _find_csv(station_code)
    if csv_path is None:
        raise HTTPException(status_code=404, detail=f"No data file for {station_code}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

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

    channels = ["CH0", "CH1", "CH3"]
    result = {
        "station_code": station_code,
        "station_name": CODE_MAP[station_code]["name"],
        "source_file": csv_path.name,
        "timestamps": df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
    }

    for ch in channels:
        if ch in df.columns:
            series = df[ch]
            result[ch] = [None if pd.isna(v) else float(v) for v in series]
        else:
            result[ch] = []

    return result


@app.get("/api/update/status")
def update_status():
    """Return current update state. Frontend polls this."""
    with _update_lock:
        return {
            "status": _update_state["status"],
            "last_update": _update_state["last_update"],
            "progress": _update_state["progress"],
            "error": _update_state["error"],
            "stations_done": _update_state["stations_done"],
            "stations_total": _update_state["stations_total"],
        }


@app.post("/api/update/trigger")
def trigger_update():
    """
    Trigger a background data update. Only runs if:
    - Not already updating
    - Last update was >24h ago (or never)
    """
    with _update_lock:
        if _update_state["status"] in ("downloading", "correcting"):
            return {"triggered": False, "reason": "Update already in progress"}

        last = _update_state["last_update"]
        if last:
            last_dt = datetime.fromisoformat(last)
            if datetime.now() - last_dt < timedelta(hours=24):
                return {
                    "triggered": False,
                    "reason": "Last update was less than 24h ago",
                    "last_update": last,
                }

        # Reset state and launch background thread
        _update_state["status"] = "downloading"
        _update_state["progress"] = "Starting update..."
        _update_state["error"] = None
        _update_state["stations_done"] = 0

    t = threading.Thread(target=_run_update, daemon=True)
    t.start()
    return {"triggered": True, "reason": "Update started"}


# ---- Serve frontend ----
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
