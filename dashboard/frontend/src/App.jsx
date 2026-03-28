import React, { useEffect, useRef, useState, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import Plot from 'react-plotly.js';

const API_BASE = '/api';

export default function App() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);
  const [stations, setStations] = useState([]);
  const [selected, setSelected] = useState(null);
  const [tsData, setTsData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Update state
  const [updateStatus, setUpdateStatus] = useState(null); // null | {status, progress, ...}
  const [showToast, setShowToast] = useState(false);
  const pollRef = useRef(null);

  // Fetch stations on mount
  useEffect(() => {
    fetch(`${API_BASE}/stations`)
      .then((r) => r.json())
      .then(setStations)
      .catch((e) => console.error('Failed to load stations:', e));
  }, []);

  // Fetch timeseries for selected station
  const fetchTimeseries = useCallback((stationCode) => {
    if (!stationCode) return;
    setLoading(true);
    fetch(`${API_BASE}/timeseries/${stationCode}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setTsData(data);
        setLoading(false);
      })
      .catch((e) => {
        console.error('Failed to load timeseries:', e);
        setTsData(null);
        setLoading(false);
      });
  }, []);

  // Trigger update check on page load
  useEffect(() => {
    fetch(`${API_BASE}/update/trigger`, { method: 'POST' })
      .then((r) => r.json())
      .then((data) => {
        if (data.triggered) {
          setShowToast(true);
          setUpdateStatus({ status: 'downloading', progress: 'Starting update...' });
          startPolling();
        } else if (data.last_update) {
          // Show brief "up to date" message
          const ago = timeSince(data.last_update);
          setUpdateStatus({ status: 'fresh', progress: `Data is current (updated ${ago})` });
          setShowToast(true);
          setTimeout(() => setShowToast(false), 4000);
        }
      })
      .catch(() => {}); // silently fail if update endpoint not available
  }, []);

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(() => {
      fetch(`${API_BASE}/update/status`)
        .then((r) => r.json())
        .then((st) => {
          setUpdateStatus(st);
          if (st.status === 'done') {
            stopPolling();
            // Refresh current chart if one is selected
            if (selected) {
              fetchTimeseries(selected.code);
            }
            // Auto-hide toast after 5 seconds
            setTimeout(() => setShowToast(false), 5000);
          } else if (st.status === 'error') {
            stopPolling();
            setTimeout(() => setShowToast(false), 8000);
          }
        })
        .catch(() => {});
    }, 3000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Cleanup polling on unmount
  useEffect(() => () => stopPolling(), []);

  // Init map
  useEffect(() => {
    if (mapRef.current || !mapContainer.current) return;
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [21.63, 47.53],
      zoom: 11,
    });
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    mapRef.current = map;
    return () => map.remove();
  }, []);

  // Add markers when stations load
  useEffect(() => {
    const map = mapRef.current;
    if (!map || stations.length === 0) return;

    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    stations.forEach((s) => {
      const el = document.createElement('div');
      el.className = 'well-marker';
      el.dataset.code = s.code;
      el.title = `${s.code} - ${s.name}`;

      const label = document.createElement('div');
      label.className = 'well-label';
      label.textContent = s.code;
      el.appendChild(label);

      el.addEventListener('click', (e) => {
        e.stopPropagation();
        setSelected(s);
      });

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([s.lng, s.lat])
        .addTo(map);
      markersRef.current.push(marker);
    });
  }, [stations]);

  // Highlight selected marker
  useEffect(() => {
    document.querySelectorAll('.well-marker').forEach((el) => {
      el.classList.toggle('selected', el.dataset.code === selected?.code);
    });
  }, [selected]);

  // Fetch time series when station selected
  useEffect(() => {
    if (!selected) {
      setTsData(null);
      return;
    }
    fetchTimeseries(selected.code);
  }, [selected, fetchTimeseries]);

  const plotLayout = (title, yLabel, height) => ({
    title: { text: title, font: { size: 14 } },
    xaxis: { title: '', type: 'date' },
    yaxis: { title: yLabel },
    height,
    margin: { l: 60, r: 20, t: 40, b: 40 },
    hovermode: 'x unified',
  });

  const plotConfig = { responsive: true, displayModeBar: true, displaylogo: false };

  // Toast icon and color based on status
  const toastStyle = () => {
    if (!updateStatus) return {};
    switch (updateStatus.status) {
      case 'downloading': return { borderColor: '#f59e0b', icon: '\u2B07' };
      case 'correcting':  return { borderColor: '#3b82f6', icon: '\u2699' };
      case 'done':        return { borderColor: '#22c55e', icon: '\u2713' };
      case 'fresh':       return { borderColor: '#22c55e', icon: '\u2713' };
      case 'error':       return { borderColor: '#ef4444', icon: '\u2717' };
      default:            return { borderColor: '#888', icon: '\u22EF' };
    }
  };

  const statusLabel = () => {
    if (!updateStatus) return '';
    switch (updateStatus.status) {
      case 'downloading': return 'Downloading new data...';
      case 'correcting':  return 'Processing time series...';
      case 'done':        return 'Update complete — charts refreshed';
      case 'fresh':       return updateStatus.progress;
      case 'error':       return `Update failed: ${updateStatus.error || 'unknown error'}`;
      default:            return updateStatus.progress || '';
    }
  };

  const ts = toastStyle();

  return (
    <div className="app">
      {/* Left panel: map */}
      <div className="map-panel">
        <div className="map-header">
          <h2>KER Talajv\u00EDz Monitoring</h2>
          <p>Debrecen \u2014 18 megfigyel\u0151 k\u00FAt</p>
        </div>
        <div ref={mapContainer} className="map-container" />
        <div className="station-list">
          {stations.map((s) => (
            <div
              key={s.id}
              className={`station-item ${selected?.code === s.code ? 'active' : ''}`}
              onClick={() => setSelected(s)}
            >
              <span className="station-code">{s.code}</span>
              <span className="station-name">{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel: charts */}
      <div className="chart-panel">
        {!selected && (
          <div className="placeholder">
            <h2>Select a station</h2>
            <p>Click a marker on the map or choose from the list to view time series data.</p>
          </div>
        )}
        {selected && loading && (
          <div className="placeholder">
            <h2>Loading data for {selected.code}...</h2>
          </div>
        )}
        {selected && !loading && tsData && (
          <>
            <div className="chart-header">
              <h2>{selected.code} &mdash; {selected.name}</h2>
              <span className="source-badge">Source: {tsData.source_file}</span>
            </div>
            <div className="charts-scroll">
              <Plot
                data={[{
                  x: tsData.timestamps, y: tsData.CH3,
                  type: 'scattergl', mode: 'lines', name: 'Water level',
                  line: { color: '#1f77b4', width: 1.5 },
                }]}
                layout={plotLayout('V\u00EDzszint (Water Level)', 'm', 320)}
                config={plotConfig} useResizeHandler style={{ width: '100%' }}
              />
              <Plot
                data={[{
                  x: tsData.timestamps, y: tsData.CH1,
                  type: 'scattergl', mode: 'lines', name: 'Temperature',
                  line: { color: '#d62728', width: 1.5 },
                }]}
                layout={plotLayout('H\u0151m\u00E9rs\u00E9klet (Temperature)', '\u00B0C', 220)}
                config={plotConfig} useResizeHandler style={{ width: '100%' }}
              />
              <Plot
                data={[{
                  x: tsData.timestamps, y: tsData.CH0,
                  type: 'scattergl', mode: 'lines', name: 'Conductivity',
                  line: { color: '#2ca02c', width: 1.5 },
                }]}
                layout={plotLayout('Vezet\u0151k\u00E9pess\u00E9g (Conductivity)', 'mS/cm', 220)}
                config={plotConfig} useResizeHandler style={{ width: '100%' }}
              />
            </div>
          </>
        )}
        {selected && !loading && !tsData && (
          <div className="placeholder">
            <h2>No data available for {selected.code}</h2>
          </div>
        )}
      </div>

      {/* Update toast - lower right */}
      {showToast && updateStatus && (
        <div className="update-toast" style={{ borderLeftColor: ts.borderColor }}>
          <span className="update-icon">{ts.icon}</span>
          <div className="update-text">
            <div className="update-label">{statusLabel()}</div>
            {(updateStatus.status === 'downloading' || updateStatus.status === 'correcting') && (
              <div className="update-bar">
                <div className="update-bar-fill update-bar-animate" />
              </div>
            )}
          </div>
          <button className="update-close" onClick={() => setShowToast(false)}>&times;</button>
        </div>
      )}
    </div>
  );
}

function timeSince(isoStr) {
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
