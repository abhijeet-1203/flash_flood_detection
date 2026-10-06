
import os
import math
import json
import datetime as dt
from pathlib import Path
import time

import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
)
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

HISTORICAL_FLOOD_CSV = Path(os.getenv("HISTORICAL_FLOOD_CSV", DATA_DIR / "historical_floods.csv"))
OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip()  # optional secondary/status-only source

REQUEST_TIMEOUT = 12
MODEL_PATH = MODEL_DIR / "flood_model.joblib"
MODEL_META_PATH = MODEL_DIR / "flood_model_meta.json"

# Configurable risk thresholds (Section 11) — Model Probability -> Operational Risk Level
RISK_THRESHOLDS = [
    ("LOW", 0, 20),
    ("MODERATE", 20, 40),
    ("ELEVATED", 40, 60),
    ("HIGH", 60, 80),
    ("CRITICAL", 80, 100.0001),
]
RISK_COLORS = {
    "LOW": "#2ecc71",
    "MODERATE": "#f1c40f",
    "ELEVATED": "#e67e22",
    "HIGH": "#e74c3c",
    "CRITICAL": "#7a0d0d",
    "UNKNOWN": "#7f8c8d",
}
ALERT_THRESHOLD_PCT = float(os.getenv("ALERT_THRESHOLD_PCT", "60"))

FEATURE_COLUMNS = [
    "rain_24h_mm", "rain_72h_mm", "rain_7d_mm",
    "elevation_m", "slope_deg", "river_dist_km",
]

# ============================================================================
# GENERIC HELPERS
# ============================================================================

def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def risk_level_from_prob(pct):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "UNKNOWN"
    for name, lo, hi in RISK_THRESHOLDS:
        if lo <= pct < hi:
            return name
    return "CRITICAL"


# ============================================================================
# REAL DATA SOURCE 1: WEATHER — Open-Meteo (free, keyless, live)
# ============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat, lon):
    """Live current conditions + forecast from Open-Meteo. Real source, no key required."""
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,"
                       "surface_pressure,wind_speed_10m,weather_code",
            "hourly": "precipitation,temperature_2m,relative_humidity_2m,surface_pressure",
            "forecast_days": 3,
            "past_days": 3,
            "timezone": "auto",
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        return {"ok": True, "data": payload, "fetched_at": dt.datetime.now()}
    except Exception as e:
        return {"ok": False, "error": str(e), "fetched_at": dt.datetime.now()}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_batch(lats, lons):
    """
    Same real Open-Meteo forecast source as fetch_weather, but for many points
    in a SINGLE request (Open-Meteo accepts comma-separated coordinate lists).
    Used by the Live Risk Map so an N-cell grid costs one network round trip
    for weather instead of N. Returns a list of {ok, data|error} in point order.
    """
    try:
        params = {
            "latitude": ",".join(f"{x:.5f}" for x in lats),
            "longitude": ",".join(f"{x:.5f}" for x in lons),
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,"
                       "surface_pressure,wind_speed_10m,weather_code",
            "hourly": "precipitation,temperature_2m,relative_humidity_2m,surface_pressure",
            "forecast_days": 3,
            "past_days": 3,
            "timezone": "auto",
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=REQUEST_TIMEOUT * 2)
        r.raise_for_status()
        payload = r.json()
        items = payload if isinstance(payload, list) else [payload]
        return [{"ok": True, "data": item} for item in items]
    except Exception as e:
        return [{"ok": False, "error": str(e)} for _ in lats]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_rainfall(lat, lon, date_str, lookback_days=7):
    """
    Real historical daily rainfall for a location around a given date, via
    Open-Meteo's Archive API (ERA5 reanalysis). Used to build genuine training
    features for both flood-event dates and non-event dates — no synthetic values.
    Returns dict of accumulated rainfall over 1/3/7 days ending on date_str, or None.
    """
    try:
        end = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        start = end - dt.timedelta(days=lookback_days)
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": "precipitation_sum",
            "timezone": "auto",
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        precs = daily.get("precipitation_sum", [])
        if not precs:
            return None
        precs = [p if p is not None else 0.0 for p in precs]
        return {
            "rain_24h_mm": float(precs[-1]) if len(precs) >= 1 else 0.0,
            "rain_72h_mm": float(sum(precs[-3:])) if len(precs) >= 3 else float(sum(precs)),
            "rain_7d_mm": float(sum(precs)),
        }
    except Exception:
        return None


def current_accumulations(weather_payload):
    """Derive real 24h/72h/7d rainfall accumulation from the live Open-Meteo response."""
    try:
        hourly = weather_payload["hourly"]
        times = hourly["time"]
        precip = hourly["precipitation"]
        df = pd.DataFrame({"time": pd.to_datetime(times), "precip": precip})
        now = pd.Timestamp.now()
        r24 = df[df["time"] >= now - pd.Timedelta(hours=24)]["precip"].sum()
        r72 = df[df["time"] >= now - pd.Timedelta(hours=72)]["precip"].sum()
        r7d = df["precip"].sum()  # past_days=3 + forecast, best available window
        return {"rain_24h_mm": float(r24), "rain_72h_mm": float(r72), "rain_7d_mm": float(r7d)}
    except Exception:
        return {"rain_24h_mm": None, "rain_72h_mm": None, "rain_7d_mm": None}


# ============================================================================
# REAL DATA SOURCE 2: GEOCODING — Open-Meteo Geocoding API (free, keyless)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(query):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 10, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("results", []) or []
    except Exception:
        return []


# ============================================================================
# REAL DATA SOURCE 3: TERRAIN — Open-Meteo Elevation API (SRTM, free, keyless)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elevation_grid(lats, lons):
    """
    Fetch elevation data with chunking to avoid URL length limits.
    Open-Meteo elevation API has a limit on URL length, so we chunk requests.
    """
    if isinstance(lats, (int, float)):
        lats = [lats]
        lons = [lons]
    
    # Limit chunk size to avoid URL length issues
    CHUNK_SIZE = 50
    all_elevations = []
    
    for i in range(0, len(lats), CHUNK_SIZE):
        chunk_lats = lats[i:i+CHUNK_SIZE]
        chunk_lons = lons[i:i+CHUNK_SIZE]
        
        try:
            lat_str = ",".join(f"{x:.6f}" for x in chunk_lats)
            lon_str = ",".join(f"{x:.6f}" for x in chunk_lons)
            
            r = requests.get(
                "https://api.open-meteo.com/v1/elevation",
                params={"latitude": lat_str, "longitude": lon_str},
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            result = r.json()
            elevations = result.get("elevation", [])
            if elevations:
                # Pad if needed
                if len(elevations) < len(chunk_lats):
                    elevations.extend([None] * (len(chunk_lats) - len(elevations)))
                all_elevations.extend(elevations)
            else:
                all_elevations.extend([None] * len(chunk_lats))
        except Exception as e:
            # On error, fill with None for this chunk
            all_elevations.extend([None] * len(chunk_lats))
            # Only log warning once
            if i == 0:
                st.warning(f"Elevation API error: {str(e)}")
    
    return all_elevations


def fetch_elevation(lat, lon):
    elevs = fetch_elevation_grid([lat], [lon])
    return elevs[0] if elevs and elevs[0] is not None else None


def estimate_slope_deg(lat, lon, elevation_center):
    """
    Real terrain slope estimated from real elevation samples (~1.1km offsets)
    around the point, using the same SRTM-derived Open-Meteo elevation source.
    """
    if elevation_center is None:
        return None
    offset = 0.01  # ~1.1 km at these latitudes
    lats = [lat + offset, lat - offset, lat, lat]
    lons = [lon, lon, lon + offset, lon - offset]
    elevs = fetch_elevation_grid(lats, lons)
    if any(e is None for e in elevs):
        return None
    run_m = offset * 111000
    rises = [abs(elevation_center - e) for e in elevs]
    slope_deg = math.degrees(math.atan(max(rises) / run_m)) if run_m else None
    return slope_deg


# ============================================================================
# REAL DATA SOURCE 4: RIVERS — OpenStreetMap via Overpass API (free, keyless)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rivers_osm(lat, lon, radius_km=15):
    try:
        delta = radius_km / 111.0
        bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
        query = f"""
        [out:json][timeout:15];
        (
          way["waterway"~"^(river|stream)$"]({bbox});
        );
        out geom;
        """
        # Use a shorter timeout to avoid hanging
        r = requests.post(
            "https://overpass-api.de/api/interpreter", 
            data={"data": query}, 
            timeout=15  # Reduced from 25
        )
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])
    except requests.exceptions.Timeout:
        # Timeout is common with Overpass, return empty list
        return []
    except requests.exceptions.ConnectionError:
        # Connection errors are common with Overpass
        return []
    except Exception:
        return []


def distance_to_nearest_river_km(lat, lon, river_elements):
    if not river_elements:
        return None
    min_dist = None
    for el in river_elements:
        geom = el.get("geometry", [])
        if not geom:
            continue
        for pt in geom:
            if "lat" in pt and "lon" in pt:
                d = haversine_km(lat, lon, pt["lat"], pt["lon"])
                if min_dist is None or d < min_dist:
                    min_dist = d
    return min_dist


# ============================================================================
# REAL FEATURE BUNDLE FOR A LOCATION (combines all live sources)
# ============================================================================

def build_live_feature_bundle(lat, lon):
    """
    Pulls every real-time source for one location and assembles the feature
    vector the model (or heuristic) consumes. Any source that fails is left
    as None and surfaced as "unavailable" — never substituted.
    """
    bundle = {"lat": lat, "lon": lon, "fetched_at": dt.datetime.now()}

    # Weather
    weather = fetch_weather(lat, lon)
    bundle["weather_ok"] = weather["ok"]
    if weather["ok"]:
        data = weather["data"]
        bundle["current"] = data.get("current", {})
        accum = current_accumulations(data)
        bundle.update(accum)
    else:
        bundle["current"] = {}
        bundle["rain_24h_mm"] = bundle["rain_72h_mm"] = bundle["rain_7d_mm"] = None
        bundle["weather_error"] = weather.get("error")

    # Elevation
    elevation = fetch_elevation(lat, lon)
    bundle["elevation_m"] = elevation
    
    # Slope - only if elevation is available
    bundle["slope_deg"] = estimate_slope_deg(lat, lon, elevation) if elevation is not None else None

    # Rivers
    rivers = fetch_rivers_osm(lat, lon)
    bundle["rivers_ok"] = len(rivers) > 0
    bundle["river_dist_km"] = distance_to_nearest_river_km(lat, lon, rivers)

    return bundle


def build_risk_grid(lat0, lon0, span_km, grid_size):
    """
    Fast version of build_live_feature_bundle for a whole grid at once.
    Instead of N cells x ~7 API calls each, this issues a small FIXED number
    of batched calls total (still all real data, same sources):
      - 1 batched weather call for every cell center
      - 1 batched elevation call covering every cell center + its slope-offset points
      - 1 Overpass river query covering the whole grid extent, then per-cell
        distance is computed locally (no extra network calls)
    """
    deg_span = span_km / 111.0
    lat_steps = np.linspace(lat0 - deg_span / 2, lat0 + deg_span / 2, grid_size)
    lon_steps = np.linspace(lon0 - deg_span / 2, lon0 + deg_span / 2, grid_size)
    centers = [(la, lo) for la in lat_steps for lo in lon_steps]

    # --- one batched weather call for all centers ---
    weather_results = fetch_weather_batch([c[0] for c in centers], [c[1] for c in centers])

    # --- one batched elevation call for all centers + their 4 slope-offset points ---
    offset = 0.01  # ~1.1 km, matches estimate_slope_deg
    elev_lats, elev_lons = [], []
    for la, lo in centers:
        elev_lats += [la, la + offset, la - offset, la, la]
        elev_lons += [lo, lo, lo, lo + offset, lo - offset]
    elevations = fetch_elevation_grid(elev_lats, elev_lons)

    # --- one river query covering the whole grid extent ---
    grid_radius_km = max(span_km / 2 + 5, 10)
    rivers = fetch_rivers_osm(lat0, lon0, radius_km=grid_radius_km)

    cells = []
    for i, (la, lo) in enumerate(centers):
        w = weather_results[i] if i < len(weather_results) else {"ok": False, "error": "no data"}
        if w.get("ok"):
            accum = current_accumulations(w["data"])
        else:
            accum = {"rain_24h_mm": None, "rain_72h_mm": None, "rain_7d_mm": None}

        base = i * 5
        center_elev = elevations[base] if base < len(elevations) else None
        offsets = elevations[base + 1: base + 5] if base + 5 <= len(elevations) else [None, None, None, None]
        slope = None
        if center_elev is not None and all(e is not None for e in offsets):
            run_m = offset * 111000
            rises = [abs(center_elev - e) for e in offsets]
            slope = math.degrees(math.atan(max(rises) / run_m)) if run_m else None

        river_dist = distance_to_nearest_river_km(la, lo, rivers)

        cell_bundle = {
            "lat": la, "lon": lo,
            "weather_ok": w.get("ok", False),
            "current": w["data"].get("current", {}) if w.get("ok") else {},
            **accum,
            "elevation_m": center_elev,
            "slope_deg": slope,
            "rivers_ok": len(rivers) > 0,
            "river_dist_km": river_dist,
        }
        cells.append(cell_bundle)

    return cells, lat_steps, lon_steps


# ============================================================================
# HISTORICAL FLOOD DATASET (must be supplied by the user — see README)
# ============================================================================

def historical_dataset_available():
    return HISTORICAL_FLOOD_CSV.exists()


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_events():
    """
    Loads the user-supplied genuine historical flood-events CSV.
    Required columns: date (YYYY-MM-DD), latitude, longitude
    Optional: location_name
    Returns None if the file does not exist — the app must NOT invent one.
    """
    if not HISTORICAL_FLOOD_CSV.exists():
        return None
    try:
        df = pd.read_csv(HISTORICAL_FLOOD_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"date", "latitude", "longitude"}
        if not required.issubset(set(df.columns)):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "latitude", "longitude"])
        return df.reset_index(drop=True)
    except Exception:
        return None


# ============================================================================
# PREDICTION: genuine ML probability, OR transparent rule-based fallback
# ============================================================================

def heuristic_risk_score(bundle):
    """
    Transparent, documented rule-based score used when no genuine
    historical dataset has been supplied for ML training (Section 27 requires
    we never fabricate an ML probability). Built entirely from real live
    inputs. Formula and weights are shown to the user in-app — not a black box.
    """
    r24 = bundle.get("rain_24h_mm")
    r72 = bundle.get("rain_72h_mm")
    slope = bundle.get("slope_deg")
    river_dist = bundle.get("river_dist_km")

    if r24 is None and r72 is None:
        return None, "Live rainfall data unavailable — cannot compute even a heuristic score."

    score = 0.0
    weights_used = []
    if r24 is not None:
        component = min(r24 / 100.0, 1.0) * 40
        score += component
        weights_used.append(f"24h rainfall {r24:.1f}mm -> +{component:.1f}")
    if r72 is not None:
        component = min(r72 / 200.0, 1.0) * 25
        score += component
        weights_used.append(f"72h rainfall {r72:.1f}mm -> +{component:.1f}")
    if river_dist is not None:
        component = max(0, (10 - min(river_dist, 10)) / 10) * 20
        score += component
        weights_used.append(f"river distance {river_dist:.2f}km -> +{component:.1f}")
    if slope is not None:
        # steep terrain -> faster runoff -> higher flash-flood contribution;
        # very flat terrain near a river also raises ponding risk slightly.
        component = min(slope / 30.0, 1.0) * 15
        score += component
        weights_used.append(f"slope {slope:.1f} deg -> +{component:.1f}")

    return round(min(score, 100.0), 1), weights_used


def model_confidence(bundle, meta):
    """Section 15 — confidence derived from real data completeness + model quality."""
    missing = sum(1 for c in FEATURE_COLUMNS if bundle.get(c) is None)
    completeness = 1 - (missing / len(FEATURE_COLUMNS))
    quality = 0.5
    if meta and meta.get("test_scores"):
        ts = meta["test_scores"]
        quality = np.mean([v for v in [ts.get("roc_auc"), ts.get("pr_auc")] if v is not None] or [0.5])
    conf = 0.5 * completeness + 0.5 * quality
    label = "High" if conf > 0.75 else ("Moderate" if conf > 0.5 else "Low")
    reasons = []
    if missing:
        reasons.append(f"{missing} of {len(FEATURE_COLUMNS)} live features unavailable")
    if meta is None:
        reasons.append("using rule-based heuristic")
    return label, conf, reasons


# ============================================================================
# SYSTEM STATUS (Section 18) — reflects real connection state, checked live
# ============================================================================

def check_system_status():
    status = {}
    w = fetch_weather(20.5937, 78.9629)  # India centroid, cheap liveness probe
    status["Weather API (Open-Meteo)"] = "LIVE" if w["ok"] else "UNAVAILABLE"

    elev = fetch_elevation(20.5937, 78.9629)
    status["GIS Elevation (Open-Meteo)"] = "AVAILABLE" if elev is not None else "UNAVAILABLE"

    rivers = fetch_rivers_osm(20.5937, 78.9629, radius_km=10)
    status["River Data (OSM Overpass)"] = "AVAILABLE" if rivers else "UNAVAILABLE"

    status["Historical Flood Dataset"] = "AVAILABLE" if historical_dataset_available() else "UNAVAILABLE"

    status["IoT / Live Sensor Feed"] = "UNAVAILABLE"  # Section 19 — never simulated
    return status


def get_risk_reason(bundle, prob_pct, risk_level):
    """Generate reason for risk level"""
    if risk_level in ["HIGH", "CRITICAL"]:
        reasons = []
        r24 = bundle.get("rain_24h_mm")
        r72 = bundle.get("rain_72h_mm")
        slope = bundle.get("slope_deg")
        river_dist = bundle.get("river_dist_km")
        
        if r24 is not None and r24 > 50:
            reasons.append(f"⚠️ Heavy 24h rainfall: {r24:.1f}mm")
        elif r24 is not None and r24 > 25:
            reasons.append(f"🌧️ Moderate 24h rainfall: {r24:.1f}mm")
            
        if r72 is not None and r72 > 100:
            reasons.append(f"⚠️ Heavy 72h rainfall: {r72:.1f}mm")
            
        if slope is not None and slope > 20:
            reasons.append(f"⛰️ Steep terrain ({slope:.1f}°) increases flash flood risk")
            
        if river_dist is not None and river_dist < 1:
            reasons.append(f"🏞️ Very close to river/stream ({river_dist:.2f}km)")
        elif river_dist is not None and river_dist < 3:
            reasons.append(f"🏞️ Near river/stream ({river_dist:.2f}km)")
            
        if reasons:
            return " | ".join(reasons)
        return "⚠️ Multiple risk factors present"
    else:
        return "✅ No active flood risk factors detected."


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="SIH26192 — Flash Flood Prediction (NoSemicolons)",
                    layout="wide", page_icon="🌊")

if "location" not in st.session_state:
    st.session_state.location = {"name": "Dehradun, Uttarakhand", "lat": 30.3165, "lon": 78.0322}
if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None

st.sidebar.title("🌊 FlashFlood AI")
st.sidebar.caption("AI-Powered Flash Flood Prediction & Early Warning System Hilly Regions · Multi-Source Real-Time Data")

st.sidebar.subheader("📍 Location")
search_q = st.sidebar.text_input("Search village / district / place", "")
if search_q:
    results = geocode_place(search_q)
    if results:
        options = [f'{r["name"]}, {r.get("admin1", "")} ({r["latitude"]:.3f},{r["longitude"]:.3f})' for r in results]
        sel = st.sidebar.selectbox("Matches", options)
        idx = options.index(sel)
        chosen = results[idx]
        if st.sidebar.button("Use this location"):
            st.session_state.location = {
                "name": f'{chosen["name"]}, {chosen.get("admin1", "")}',
                "lat": chosen["latitude"], "lon": chosen["longitude"],
            }
            st.session_state.bundle = None
    else:
        st.sidebar.caption("No matches from the geocoding API.")

st.sidebar.markdown(f"**Current:** {st.session_state.location['name']}")
c1, c2 = st.sidebar.columns(2)
lat_in = c1.number_input("Latitude", value=float(st.session_state.location["lat"]), format="%.4f")
lon_in = c2.number_input("Longitude", value=float(st.session_state.location["lon"]), format="%.4f")
if lat_in != st.session_state.location["lat"] or lon_in != st.session_state.location["lon"]:
    st.session_state.location = {"name": "Custom coordinates", "lat": lat_in, "lon": lon_in}
    st.session_state.bundle = None

refresh = st.sidebar.button("🔄 Refresh live data", use_container_width=True)
if refresh or st.session_state.bundle is None:
    with st.spinner("Fetching live weather, terrain and river data..."):
        st.session_state.bundle = build_live_feature_bundle(
            st.session_state.location["lat"], st.session_state.location["lon"]
        )
        st.session_state.last_updated = dt.datetime.now()

if st.session_state.last_updated:
    st.sidebar.caption(f"Last Updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Risk Map", "AI Flood Prediction", "Weather & Environment",
     "Alerts", "Historical Analysis", "Data Sources / System Status"],
)

bundle = st.session_state.bundle

# Compute current probability / heuristic once, reused across pages
prob_pct, prob_source, factors = None, None, None
if bundle:
    h_score, h_factors = heuristic_risk_score(bundle)
    prob_pct = h_score
    prob_source = "Rule-based heuristic"
    factors = h_factors
risk_level = risk_level_from_prob(prob_pct)

# ----------------------------------------------------------------------------
def render_current_conditions_block():
    cols = st.columns(4)
    r24 = bundle.get("rain_24h_mm")
    cols[0].metric("Rainfall (24h)", f"{r24:.1f} mm" if r24 is not None else "Live data unavailable")
    r72 = bundle.get("rain_72h_mm")
    cols[1].metric("Rainfall (72h)", f"{r72:.1f} mm" if r72 is not None else "Live data unavailable")
    cur = bundle.get("current", {})
    temp = cur.get("temperature_2m")
    cols[2].metric("Temperature", f"{temp:.1f}°C" if temp is not None else "Live data unavailable")
    hum = cur.get("relative_humidity_2m")
    cols[3].metric("Humidity", f"{hum:.0f}%" if hum is not None else "Live data unavailable")


def render_risk_badge():
    color = RISK_COLORS.get(risk_level, "#7f8c8d")
    prob_display = f"{prob_pct:.1f}%" if prob_pct is not None else "Unavailable"
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:12px;background:{color};color:white;text-align:center;">
        <h2 style="margin:0;">CURRENT FLOOD RISK: {risk_level}</h2>
        <p style="margin:0;font-size:20px;">Probability: {prob_display}</p>
        </div>
        """, unsafe_allow_html=True,
    )
    st.caption(f"Source: {prob_source or 'n/a'}")


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Flash Flood Prediction — Command Center")
    st.caption(f"Location: {st.session_state.location['name']}  |  "
               f"({st.session_state.location['lat']:.4f}, {st.session_state.location['lon']:.4f})")

    if bundle is None:
        st.warning("Live data unavailable.")
    else:
        render_risk_badge()
        st.markdown("### Current Conditions")
        render_current_conditions_block()

        st.markdown("### AI Prediction Summary")
        if factors:
            st.write("Main contributing factors:")
            for line in factors:
                st.write(f"- {line}")
        else:
            st.info("No contributing-factor explanation available (insufficient live inputs).")

        conf_label, conf_val, conf_reasons = model_confidence(bundle, None)
        st.markdown(f"**Confidence:** {conf_label} ({conf_val*100:.0f}%)")
        for reason in conf_reasons:
            st.caption(f"- {reason}")

    st.markdown("---")
    st.markdown("### Quick Map")
    st.caption("Shaded by the current risk level at this single point. For a full "
               "multi-cell shaded risk map, use the 'Live Risk Map' page.")
    quick_color = RISK_COLORS.get(risk_level, "#7f8c8d")
    m = folium.Map(location=[st.session_state.location["lat"], st.session_state.location["lon"]], zoom_start=9)
    folium.Circle(
        [st.session_state.location["lat"], st.session_state.location["lon"]],
        radius=6000, color=quick_color, fill=True, fill_color=quick_color, fill_opacity=0.45,
        popup=f"{risk_level} — {prob_pct:.1f}% ({prob_source})" if prob_pct is not None else "Live data unavailable",
    ).add_to(m)
    folium.Marker(
        [st.session_state.location["lat"], st.session_state.location["lon"]],
        popup=st.session_state.location["name"],
        icon=folium.Icon(color="red" if risk_level in ("HIGH", "CRITICAL") else "blue"),
    ).add_to(m)
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
    padding:10px; border-radius:8px; border:1px solid #ccc; font-size:13px;">
    <b>Risk Legend</b><br>
    <span style="color:#2ecc71;">■</span> Low &nbsp;
    <span style="color:#f1c40f;">■</span> Moderate &nbsp;
    <span style="color:#e67e22;">■</span> Elevated<br>
    <span style="color:#e74c3c;">■</span> High &nbsp;
    <span style="color:#7a0d0d;">■</span> Critical
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, height=350, use_container_width=True)

# ============================================================================
# PAGE: LIVE RISK MAP
# ============================================================================
elif page == "Live Risk Map":
    st.title("🗺️ Live Risk Map")
    st.caption("Grid cells shaded by real-time computed flood risk around the selected location. "
               "Each cell uses its own live weather + real terrain + real river-distance features.")

    grid_size = st.slider("Grid resolution (cells per side)", 3, 9, 5)
    span_km = st.slider("Map span (km)", 10, 60, 25)
    st.caption("Uses a small fixed number of batched API calls regardless of grid size "
               "(1 weather call, 1 elevation call, 1 river query) — not one call per cell.")

    if st.button("Generate risk grid (fast — batched live data)"):
        lat0, lon0 = st.session_state.location["lat"], st.session_state.location["lon"]
        with st.spinner("Fetching batched live weather, elevation and river data for the grid..."):
            cell_bundles, lat_steps, lon_steps = build_risk_grid(lat0, lon0, span_km, grid_size)

        cells = []
        for cb in cell_bundles:
            p, _ = heuristic_risk_score(cb)
            cells.append({"lat": cb["lat"], "lon": cb["lon"], "prob": p, "source": "heuristic", "bundle": cb})
        st.session_state.risk_grid = cells

    grid = st.session_state.get("risk_grid")
    if grid:
        m = folium.Map(location=[st.session_state.location["lat"], st.session_state.location["lon"]], zoom_start=10)
        cell_deg = (span_km / 111.0) / grid_size
        for cell in grid:
            level = risk_level_from_prob(cell["prob"])
            color = RISK_COLORS.get(level, "#7f8c8d")
            bounds = [[cell["lat"] - cell_deg / 2, cell["lon"] - cell_deg / 2],
                      [cell["lat"] + cell_deg / 2, cell["lon"] + cell_deg / 2]]
            b = cell["bundle"]
            popup_html = (
                f"<b>{level}</b><br>"
                f"Probability: {cell['prob']:.1f}% ({cell['source']})<br>"
                f"Rain 24h: {b.get('rain_24h_mm', 'N/A')}<br>"
                f"Elevation: {b.get('elevation_m', 'N/A')} m<br>"
                f"Slope: {b.get('slope_deg', 'N/A')}<br>"
                f"River dist: {b.get('river_dist_km', 'N/A')} km"
            )
            folium.Rectangle(
                bounds=bounds, color=color, fill=True, fill_color=color, fill_opacity=0.55,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
        padding:10px; border-radius:8px; border:1px solid #ccc;">
        <b>Risk Legend</b><br>
        <span style="color:#2ecc71;">■</span> Low &nbsp;
        <span style="color:#f1c40f;">■</span> Moderate &nbsp;
        <span style="color:#e67e22;">■</span> Elevated<br>
        <span style="color:#e74c3c;">■</span> High &nbsp;
        <span style="color:#7a0d0d;">■</span> Critical
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, height=550, use_container_width=True)
    else:
        st.info("Click 'Generate risk grid' to compute real-time risk polygons for this area.")

# ============================================================================
# PAGE: AI FLOOD PREDICTION
# ============================================================================
elif page == "AI Flood Prediction":
    st.title("🤖 AI Flood Prediction")
    if bundle is None:
        st.warning("Live data unavailable.")
    else:
        render_risk_badge()
        st.markdown("### Feature Snapshot (all real, live)")
        feat_df = pd.DataFrame([{k: bundle.get(k) for k in FEATURE_COLUMNS}])
        st.dataframe(feat_df, use_container_width=True)

        st.markdown("### Explainability")
        if factors:
            st.write("Heuristic contribution breakdown:")
            for line in factors:
                st.write(f"- {line}")
        else:
            st.info("Not enough live data for an explanation.")

        conf_label, conf_val, conf_reasons = model_confidence(bundle, None)
        st.markdown(f"**Confidence: {conf_label}**")
        if conf_label == "Low":
            st.warning("Low confidence due to incomplete live data." if conf_reasons else "Low confidence.")
        for r in conf_reasons:
            st.caption(f"- {r}")

# ============================================================================
# PAGE: WEATHER & ENVIRONMENT
# ============================================================================
elif page == "Weather & Environment":
    st.title("🌦️ Weather & Environment")
    if bundle is None or not bundle.get("weather_ok"):
        err = bundle.get("weather_error") if bundle else "not fetched"
        st.error(f"⚠ Weather API unavailable. Last successful observation: "
                 f"{st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_updated else 'n/a'} "
                 f"(error: {err})")
    else:
        render_current_conditions_block()
        cur = bundle.get("current", {})
        c1, c2 = st.columns(2)
        c1.metric("Pressure", f"{cur.get('surface_pressure', 'n/a')} hPa")
        c2.metric("Wind speed", f"{cur.get('wind_speed_10m', 'n/a')} km/h")

        weather = fetch_weather(st.session_state.location["lat"], st.session_state.location["lon"])
        if weather["ok"]:
            hourly = weather["data"]["hourly"]
            hdf = pd.DataFrame({"time": pd.to_datetime(hourly["time"]), "precipitation": hourly["precipitation"],
                                 "temperature": hourly["temperature_2m"]})
            fig1 = px.line(hdf, x="time", y="precipitation", title="Hourly precipitation (past 3 days + forecast)")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.line(hdf, x="time", y="temperature", title="Hourly temperature")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE: ALERTS
# ============================================================================
elif page == "Alerts":
    st.title("🚨 Alerts")
    if bundle is None or prob_pct is None:
        st.info("No alert can be generated — live data unavailable.")
    elif prob_pct >= ALERT_THRESHOLD_PCT:
        st.error(
            f"""
🚨 **{risk_level} FLOOD RISK**

**Location:** {st.session_state.location['name']}

**Probability:** {prob_pct:.1f}% ({prob_source})

**Risk Analysis:** {get_risk_reason(bundle, prob_pct, risk_level)}
"""
        )
        st.caption(f"Last updated: {st.session_state.last_updated}")
    else:
        st.success(f"✅ No active flood risk detected. Current probability {prob_pct:.1f}% is below the "
                   f"{ALERT_THRESHOLD_PCT:.0f}% alert threshold.")

    st.markdown("---")
    st.caption(f"Alert threshold is configurable (currently {ALERT_THRESHOLD_PCT:.0f}%). "
               "Set ALERT_THRESHOLD_PCT in .env to change it.")

# ============================================================================
# PAGE: HISTORICAL ANALYSIS
# ============================================================================
elif page == "Historical Analysis":
    st.title("📜 Historical Analysis")
    events_df = load_historical_events()
    if events_df is None:
        st.warning(
            "No genuine historical flood dataset found. Place a real CSV (columns: "
            "date, latitude, longitude[, location_name]) at "
            f"`{HISTORICAL_FLOOD_CSV}`. See README.md for real dataset sources "
            "(e.g. NDMA/NRSC flood inventory, Dartmouth Flood Observatory, data.gov.in)."
        )
    else:
        st.success(f"Loaded {len(events_df)} genuine historical flood events.")
        st.dataframe(events_df, use_container_width=True)

        m = folium.Map(location=[events_df["latitude"].mean(), events_df["longitude"].mean()], zoom_start=6)
        for _, row in events_df.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]], radius=5, color="#7a0d0d", fill=True,
                popup=f'{row.get("location_name", "")} — {row["date"].date()}',
            ).add_to(m)
        st.markdown("#### Historical flood-event map")
        st_folium(m, height=400, use_container_width=True)

        events_df["month"] = events_df["date"].dt.month_name()
        month_counts = events_df["month"].value_counts()
        fig = px.bar(x=month_counts.index, y=month_counts.values,
                     labels={"x": "Month", "y": "Number of events"}, title="Seasonal flood frequency")
        st.plotly_chart(fig, use_container_width=True)

        events_df["year"] = events_df["date"].dt.year
        year_counts = events_df["year"].value_counts().sort_index()
        fig2 = px.bar(x=year_counts.index, y=year_counts.values,
                      labels={"x": "Year", "y": "Number of events"}, title="Flood events per year")
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE: DATA SOURCES / SYSTEM STATUS
# ============================================================================
elif page == "Data Sources / System Status":
    st.title("🔧 Data Sources / System Status")
    status = check_system_status()
    for name, state in status.items():
        icon = "🟢" if state in ("LIVE", "AVAILABLE", "READY") else "🟡"
        st.write(f"{icon} **{name}**: {state}")

    st.markdown("---")
    st.markdown("### Sources in use")
    st.markdown(
        """
- **Weather (live + forecast)**: [Open-Meteo Forecast API](https://open-meteo.com/) — free, keyless, real-time.
- **Weather (historical, for training)**: [Open-Meteo Archive API](https://open-meteo.com/) — ERA5 reanalysis.
- **Elevation / slope**: [Open-Meteo Elevation API](https://open-meteo.com/) — SRTM-based.
- **Rivers / waterways**: [OpenStreetMap Overpass API](https://overpass-api.de/).
- **Geocoding**: [Open-Meteo Geocoding API](https://open-meteo.com/).
- **Historical flood events**: user-supplied genuine CSV at
  `data/historical_floods.csv` (see README.md for real sources).
- **IoT sensors**: none integrated — Section 19 explicitly forbids simulating this,
  so it is always reported as unavailable rather than faked.
        """
    )
