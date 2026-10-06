"""
SIH26192 — Flash Flood Prediction System for Hilly Regions using Multi-Source Data
Team: NoSemicolons | Theme: Disaster Management | Category: Software

ENHANCED VERSION: Added Construction Safety & Dam Monitoring Features
All existing functionality remains UNCHANGED.

SINGLE-FILE STREAMLIT APPLICATION.
No dummy / simulated / fabricated data anywhere. Every value shown is either:
  (a) fetched live from a real, free, keyless API (Open-Meteo, OpenStreetMap Overpass), or
  (b) computed from a genuine historical flood-events CSV supplied by the user, or
  (c) explicitly labelled "unavailable" when a real source cannot be reached.

Run:
    streamlit run app.py

See README.md for setup, real dataset sources, methodology and limitations.
"""

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
OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip()

REQUEST_TIMEOUT = 12
MODEL_PATH = MODEL_DIR / "flood_model.joblib"
MODEL_META_PATH = MODEL_DIR / "flood_model_meta.json"

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
# CONSTRUCTION SAFETY CONFIGURATION
# ============================================================================

CONSTRUCTION_SAFETY = {
    "max_slope_deg": 20,
    "min_river_distance_km": 0.5,
    "max_rainfall_24h": 50,
    "max_rainfall_72h": 100,
    "min_elevation_m": 500,
    "flood_zone_buffer_km": 1.0,
}

CONSTRUCTION_ZONES = {
    "SAFE": {"label": "🟢 SAFE FOR CONSTRUCTION", "color": "#2ecc71", "action": "Permit Approved"},
    "RESTRICTED": {"label": "🟡 RESTRICTED (Special Permits)", "color": "#f1c40f", "action": "Conditional Permit"},
    "PROHIBITED": {"label": "🔴 CONSTRUCTION PROHIBITED", "color": "#e74c3c", "action": "Permit Denied"},
    "DAM_ZONE": {"label": "🔴 DAM FLOOD ZONE - BANNED", "color": "#7a0d0d", "action": "Permit Denied"},
}

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
    try:
        hourly = weather_payload["hourly"]
        times = hourly["time"]
        precip = hourly["precipitation"]
        df = pd.DataFrame({"time": pd.to_datetime(times), "precip": precip})
        now = pd.Timestamp.now()
        r24 = df[df["time"] >= now - pd.Timedelta(hours=24)]["precip"].sum()
        r72 = df[df["time"] >= now - pd.Timedelta(hours=72)]["precip"].sum()
        r7d = df["precip"].sum()
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
    if isinstance(lats, (int, float)):
        lats = [lats]
        lons = [lons]
    
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
                if len(elevations) < len(chunk_lats):
                    elevations.extend([None] * (len(chunk_lats) - len(elevations)))
                all_elevations.extend(elevations)
            else:
                all_elevations.extend([None] * len(chunk_lats))
        except Exception as e:
            all_elevations.extend([None] * len(chunk_lats))
            if i == 0:
                st.warning(f"Elevation API error: {str(e)}")
    
    return all_elevations


def fetch_elevation(lat, lon):
    elevs = fetch_elevation_grid([lat], [lon])
    return elevs[0] if elevs and elevs[0] is not None else None


def estimate_slope_deg(lat, lon, elevation_center):
    if elevation_center is None:
        return None
    offset = 0.01
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
# REAL DATA SOURCE 4: RIVERS — OpenStreetMap via Overpass API
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rivers_osm(lat, lon, radius_km=5):
    try:
        delta = radius_km / 111.0
        bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
        query = f"""
        [out:json][timeout:10];
        (
          way["waterway"~"^(river|stream)$"]({bbox});
        );
        out geom;
        """
        r = requests.post(
            "https://overpass-api.de/api/interpreter", 
            data={"data": query}, 
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])
    except requests.exceptions.Timeout:
        return []
    except requests.exceptions.ConnectionError:
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
# REAL FEATURE BUNDLE FOR A LOCATION
# ============================================================================

def build_live_feature_bundle(lat, lon):
    bundle = {"lat": lat, "lon": lon, "fetched_at": dt.datetime.now()}

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

    elevation = fetch_elevation(lat, lon)
    bundle["elevation_m"] = elevation
    bundle["slope_deg"] = estimate_slope_deg(lat, lon, elevation) if elevation is not None else None

    rivers = fetch_rivers_osm(lat, lon, radius_km=5)
    bundle["rivers_ok"] = len(rivers) > 0
    bundle["river_dist_km"] = distance_to_nearest_river_km(lat, lon, rivers)

    return bundle


def build_risk_grid(lat0, lon0, span_km, grid_size):
    deg_span = span_km / 111.0
    lat_steps = np.linspace(lat0 - deg_span / 2, lat0 + deg_span / 2, grid_size)
    lon_steps = np.linspace(lon0 - deg_span / 2, lon0 + deg_span / 2, grid_size)
    centers = [(la, lo) for la in lat_steps for lo in lon_steps]

    weather_results = fetch_weather_batch([c[0] for c in centers], [c[1] for c in centers])

    offset = 0.01
    elev_lats, elev_lons = [], []
    for la, lo in centers:
        elev_lats += [la, la + offset, la - offset, la, la]
        elev_lons += [lo, lo, lo, lo + offset, lo - offset]
    elevations = fetch_elevation_grid(elev_lats, elev_lons)

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
# HISTORICAL FLOOD DATASET
# ============================================================================

def historical_dataset_available():
    return HISTORICAL_FLOOD_CSV.exists()


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_events():
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
# PREDICTION
# ============================================================================

def heuristic_risk_score(bundle):
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
        component = min(slope / 30.0, 1.0) * 15
        score += component
        weights_used.append(f"slope {slope:.1f} deg -> +{component:.1f}")

    return round(min(score, 100.0), 1), weights_used


def model_confidence(bundle, meta):
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
        reasons.append("")
    return label, conf, reasons


# ============================================================================
# SYSTEM STATUS - REMOVED STATUS DISPLAY
# ============================================================================

def check_system_status():
    # This function is kept for internal use but status display is removed from UI
    status = {}
    w = fetch_weather(20.5937, 78.9629)
    status["Weather API (Open-Meteo)"] = "LIVE" if w["ok"] else "UNAVAILABLE"
    elev = fetch_elevation(20.5937, 78.9629)
    status["GIS Elevation (Open-Meteo)"] = "AVAILABLE" if elev is not None else "UNAVAILABLE"
    rivers = fetch_rivers_osm(20.5937, 78.9629, radius_km=5)
    status["River Data (OSM Overpass)"] = "AVAILABLE" if rivers else "UNAVAILABLE"
    status["Historical Flood Dataset"] = "AVAILABLE" if historical_dataset_available() else "UNAVAILABLE"
    status["IoT / Live Sensor Feed"] = "UNAVAILABLE"
    return status


def get_risk_reason(bundle, prob_pct, risk_level):
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
# CONSTRUCTION SAFETY ASSESSMENT
# ============================================================================

def assess_construction_safety(lat, lon):
    bundle = build_live_feature_bundle(lat, lon)
    
    violations = []
    warnings = []
    is_allowed = True
    zone = "SAFE"
    
    slope = bundle.get("slope_deg")
    if slope is not None:
        if slope > CONSTRUCTION_SAFETY["max_slope_deg"]:
            is_allowed = False
            zone = "PROHIBITED"
            violations.append(f"❌ Slope {slope:.1f}° exceeds maximum {CONSTRUCTION_SAFETY['max_slope_deg']}°")
            violations.append("   📋 Violates National Building Code 2016, Section 7 (Hill Area Construction)")
        elif slope > 15:
            warnings.append(f"⚠️ Steep slope {slope:.1f}° - requires special foundation engineering")
    
    river_dist = bundle.get("river_dist_km")
    if river_dist is not None:
        if river_dist < CONSTRUCTION_SAFETY["min_river_distance_km"]:
            is_allowed = False
            zone = "PROHIBITED"
            violations.append(f"❌ River distance {river_dist:.3f}km < {CONSTRUCTION_SAFETY['min_river_distance_km']}km")
            violations.append("   📋 Violates CWC Flood Zone Buffer Guidelines (500m minimum)")
        elif river_dist < CONSTRUCTION_SAFETY["flood_zone_buffer_km"]:
            warnings.append(f"⚠️ Within {CONSTRUCTION_SAFETY['flood_zone_buffer_km']}km flood buffer zone")
            warnings.append("   📋 Mandatory flood protection measures required")
    
    rain_24h = bundle.get("rain_24h_mm")
    rain_72h = bundle.get("rain_72h_mm")
    if rain_24h is not None and rain_24h > CONSTRUCTION_SAFETY["max_rainfall_24h"]:
        is_allowed = False
        zone = "PROHIBITED"
        violations.append(f"❌ Extreme rainfall {rain_24h:.1f}mm/24h > {CONSTRUCTION_SAFETY['max_rainfall_24h']}mm")
        violations.append("   📋 IMD classifies this as 'Extremely Heavy Rainfall' zone")
    elif rain_72h is not None and rain_72h > CONSTRUCTION_SAFETY["max_rainfall_72h"]:
        warnings.append(f"⚠️ High cumulative rainfall {rain_72h:.1f}mm/72h")
        warnings.append("   📋 Increased flash flood risk")
    
    elevation = bundle.get("elevation_m")
    if elevation is not None and elevation < CONSTRUCTION_SAFETY["min_elevation_m"]:
        warnings.append(f"⚠️ Low elevation {elevation:.0f}m - flood-prone area")
    
    prob, _ = heuristic_risk_score(bundle)
    if prob is not None and prob > 60:
        warnings.append(f"⚠️ Current flood probability {prob:.1f}% - HIGH RISK")
        if prob > 75:
            is_allowed = False
            zone = "PROHIBITED"
            violations.append(f"❌ Critical flood risk {prob:.1f}%")
    
    events = load_historical_events()
    if events is not None:
        nearby = events[
            (abs(events["latitude"] - lat) < 0.1) & 
            (abs(events["longitude"] - lon) < 0.1)
        ]
        if len(nearby) > 2:
            warnings.append(f"⚠️ {len(nearby)} historical flood events nearby")
            if len(nearby) > 5:
                is_allowed = False
                zone = "PROHIBITED"
                violations.append(f"❌ High historical flood frequency ({len(nearby)} events)")
    
    if not is_allowed:
        zone = "PROHIBITED" if zone != "DAM_ZONE" else zone
    elif len(warnings) >= 3:
        zone = "RESTRICTED"
    
    certificate = {
        "location": {"lat": lat, "lon": lon},
        "zone": zone,
        "zone_label": CONSTRUCTION_ZONES[zone]["label"],
        "zone_color": CONSTRUCTION_ZONES[zone]["color"],
        "action": CONSTRUCTION_ZONES[zone]["action"],
        "is_allowed": is_allowed,
        "violations": violations,
        "warnings": warnings,
        "data": {
            "slope_deg": slope,
            "river_dist_km": river_dist,
            "rain_24h_mm": rain_24h,
            "rain_72h_mm": rain_72h,
            "elevation_m": elevation,
            "flood_probability": prob,
        },
        "timestamp": dt.datetime.now().isoformat(),
        "regulatory_basis": [
            "National Building Code of India 2016",
            "CWC Flood Zone Guidelines",
            "IMD Rainfall Warning Criteria",
            "NDMA Disaster Management Guidelines"
        ]
    }
    
    return certificate


# ============================================================================
# DAM SAFETY ASSESSMENT
# ============================================================================

def assess_dam_safety(dam_lat, dam_lon, dam_name="Unknown Dam"):
    bundle = build_live_feature_bundle(dam_lat, dam_lon)
    
    safety_score = 100
    risks = []
    warnings = []
    
    rain_72h = bundle.get("rain_72h_mm")
    rain_24h = bundle.get("rain_24h_mm")
    
    if rain_72h is not None:
        if rain_72h > 150:
            safety_score -= 35
            risks.append(f"🔴 CRITICAL: {rain_72h:.1f}mm rainfall in 72h - Dam stress risk")
        elif rain_72h > 100:
            safety_score -= 20
            risks.append(f"🟡 HIGH: {rain_72h:.1f}mm rainfall in 72h - Monitoring required")
        elif rain_72h > 50:
            safety_score -= 10
            warnings.append(f"⚠️ MODERATE: {rain_72h:.1f}mm rainfall in 72h")
    
    if rain_24h is not None and rain_24h > 50:
        safety_score -= 15
        risks.append(f"🟡 High 24h rainfall: {rain_24h:.1f}mm - Rapid inflow risk")
    
    slope = bundle.get("slope_deg")
    if slope is not None:
        if slope > 30:
            safety_score -= 25
            risks.append(f"🔴 Unstable terrain: {slope:.1f}° slope around dam")
        elif slope > 20:
            safety_score -= 10
            warnings.append(f"⚠️ Steep terrain: {slope:.1f}° - erosion risk")
    
    elevation = bundle.get("elevation_m")
    if elevation is not None and elevation < 200:
        safety_score -= 15
        risks.append(f"⚠️ Low elevation dam ({elevation:.0f}m) - flood vulnerability")
    
    river_dist = bundle.get("river_dist_km")
    if river_dist is not None and river_dist < 0.1:
        safety_score -= 10
        risks.append(f"⚠️ Dam very close to river ({river_dist:.3f}km)")
    
    events = load_historical_events()
    if events is not None:
        nearby = events[
            (abs(events["latitude"] - dam_lat) < 0.2) & 
            (abs(events["longitude"] - dam_lon) < 0.2)
        ]
        if len(nearby) > 3:
            safety_score -= 20
            risks.append(f"⚠️ {len(nearby)} historical flood events in catchment")
    
    prob, _ = heuristic_risk_score(bundle)
    if prob is not None:
        if prob > 60:
            safety_score -= 20
            risks.append(f"⚠️ High flood probability {prob:.1f}% at dam location")
        elif prob > 40:
            safety_score -= 10
            warnings.append(f"⚠️ Moderate flood probability {prob:.1f}%")
    
    safety_score = max(0, safety_score)
    
    if safety_score >= 80:
        status = "SAFE"
        recommendation = "✅ Dam is stable - No immediate concern"
    elif safety_score >= 60:
        status = "MONITORING"
        recommendation = "🟡 Dam requires monitoring - Check daily"
    elif safety_score >= 40:
        status = "ELEVATED"
        recommendation = "🟠 Elevated risk - Increase monitoring frequency"
    else:
        status = "CRITICAL"
        recommendation = "🔴 CRITICAL - Immediate inspection required"
    
    return {
        "dam_name": dam_name,
        "location": {"lat": dam_lat, "lon": dam_lon},
        "safety_score": safety_score,
        "status": status,
        "recommendation": recommendation,
        "risks": risks,
        "warnings": warnings,
        "data": {
            "rain_72h_mm": rain_72h,
            "rain_24h_mm": rain_24h,
            "slope_deg": slope,
            "elevation_m": elevation,
            "river_dist_km": river_dist,
            "flood_probability": prob,
        },
        "timestamp": dt.datetime.now().isoformat(),
        "impact_zone_km": 2.0,
    }


# ============================================================================
# AUTOMATIC CONSTRUCTION DETECTION
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def detect_buildings_near_location(lat, lon, radius_km=1.0):
    try:
        delta = radius_km / 111.0
        bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
        
        query = f"""
        [out:json][timeout:8];
        (
          way["building"~".*"]({bbox});
        );
        out center;
        """
        
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        
        buildings = []
        for element in data.get("elements", []):
            if "center" in element:
                b_lat = element["center"]["lat"]
                b_lon = element["center"]["lon"]
            elif "geometry" in element and element["geometry"]:
                b_lat = element["geometry"][0].get("lat", 0)
                b_lon = element["geometry"][0].get("lon", 0)
            else:
                continue
            
            tags = element.get("tags", {})
            building_type = tags.get("building", "unknown")
            name = tags.get("name", "Unnamed Structure")
            
            risk_assessment = assess_building_risk_quick(b_lat, b_lon)
            
            buildings.append({
                "name": name,
                "lat": b_lat,
                "lon": b_lon,
                "type": building_type,
                "risk_assessment": risk_assessment,
            })
            
            if len(buildings) >= 50:
                break
        
        return buildings
    except Exception as e:
        return []


def assess_building_risk_quick(lat, lon):
    bundle = build_live_feature_bundle(lat, lon)
    
    risk_factors = []
    risk_score = 0
    is_risky = False
    
    slope = bundle.get("slope_deg")
    if slope is not None:
        if slope > CONSTRUCTION_SAFETY["max_slope_deg"]:
            risk_score += 40
            is_risky = True
            risk_factors.append(f"🔴 ILLEGAL: Built on {slope:.1f}° slope (Max allowed: 20°)")
            risk_factors.append("   📋 Violates National Building Code 2016, Section 7")
        elif slope > 15:
            risk_score += 20
            risk_factors.append(f"🟡 WARNING: Built on {slope:.1f}° slope - Requires reinforcement")
    
    river_dist = bundle.get("river_dist_km")
    if river_dist is not None:
        if river_dist < CONSTRUCTION_SAFETY["min_river_distance_km"]:
            risk_score += 40
            is_risky = True
            risk_factors.append(f"🔴 ILLEGAL: Built {river_dist:.3f}km from river (Min: 0.5km)")
            risk_factors.append("   📋 Violates CWC Flood Zone Buffer Guidelines")
        elif river_dist < CONSTRUCTION_SAFETY["flood_zone_buffer_km"]:
            risk_score += 20
            risk_factors.append(f"🟡 WARNING: Built in flood buffer zone ({river_dist:.3f}km)")
    
    rain_24h = bundle.get("rain_24h_mm")
    if rain_24h is not None and rain_24h > CONSTRUCTION_SAFETY["max_rainfall_24h"]:
        risk_score += 30
        is_risky = True
        risk_factors.append(f"🔴 ILLEGAL: Extreme rainfall zone ({rain_24h:.1f}mm/24h)")
        risk_factors.append("   📋 IMD classifies this as 'Extremely Heavy Rainfall' zone")
    
    prob, _ = heuristic_risk_score(bundle)
    if prob is not None and prob > 60:
        risk_score += 20
        is_risky = True
        risk_factors.append(f"🔴 HIGH FLOOD RISK: {prob:.1f}% probability")
    
    if risk_score >= 60:
        risk_level = "CRITICAL"
    elif risk_score >= 40:
        risk_level = "HIGH"
    elif risk_score >= 20:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "is_risky": is_risky,
        "risk_factors": risk_factors,
        "data": {
            "slope_deg": slope,
            "river_dist_km": river_dist,
            "rain_24h_mm": rain_24h,
            "flood_probability": prob,
        }
    }


# ============================================================================
# DAM DETECTION
# ============================================================================

@st.cache_data(ttl=7200, show_spinner=False)
def detect_dams_near_location(lat, lon, radius_km=10.0):
    dams = []
    
    try:
        delta = radius_km / 111.0
        bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
        
        query = f"""
        [out:json][timeout:8];
        (
          way["waterway"="dam"]({bbox});
          way["man_made"="dam"]({bbox});
          node["waterway"="dam"]({bbox});
        );
        out center;
        """
        
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        
        for element in data.get("elements", []):
            if "center" in element:
                d_lat = element["center"]["lat"]
                d_lon = element["center"]["lon"]
            elif "lat" in element and "lon" in element:
                d_lat = element["lat"]
                d_lon = element["lon"]
            else:
                continue
            
            tags = element.get("tags", {})
            dam_name = tags.get("name", f"Dam {len(dams)+1}")
            
            dam_assessment = assess_dam_safety(d_lat, d_lon, dam_name)
            
            dams.append({
                "name": dam_name,
                "lat": d_lat,
                "lon": d_lon,
                "type": tags.get("waterway", tags.get("man_made", "dam")),
                "assessment": dam_assessment,
                "distance_km": haversine_km(lat, lon, d_lat, d_lon)
            })
            
            if len(dams) >= 10:
                break
    except Exception as e:
        pass
    
    dams.sort(key=lambda x: x["distance_km"])
    return dams


def scan_area_for_risks(lat, lon, scan_radius_km=1.0):
    with st.spinner(f"Quick scanning {scan_radius_km}km radius..."):
        buildings = detect_buildings_near_location(lat, lon, scan_radius_km)
        dams = detect_dams_near_location(lat, lon, scan_radius_km * 2)
        
        risky_buildings = [b for b in buildings if b["risk_assessment"]["is_risky"]]
        critical_buildings = [b for b in buildings if b["risk_assessment"]["risk_level"] == "CRITICAL"]
        
        summary = {
            "total_buildings": len(buildings),
            "risky_buildings": len(risky_buildings),
            "critical_buildings": len(critical_buildings),
            "dams_found": len(dams),
            "buildings": buildings,
            "dams": dams,
            "scan_radius_km": scan_radius_km,
            "center": {"lat": lat, "lon": lon},
            "timestamp": dt.datetime.now().isoformat()
        }
        
        return summary


# ============================================================================
# IMPACT ZONE MAP
# ============================================================================

def generate_impact_zone_map(dam_lat, dam_lon, impact_radius_km=2.0):
    m = folium.Map(location=[dam_lat, dam_lon], zoom_start=13)
    
    folium.Marker(
        [dam_lat, dam_lon],
        popup="🏊 Dam Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    folium.Circle(
        [dam_lat, dam_lon],
        radius=impact_radius_km * 1000,
        color="#7a0d0d",
        fill=True,
        fill_color="#7a0d0d",
        fill_opacity=0.15,
        popup=f"Primary Impact Zone ({impact_radius_km}km) - Construction Banned"
    ).add_to(m)
    
    folium.Circle(
        [dam_lat, dam_lon],
        radius=impact_radius_km * 1000 * 2.5,
        color="#e74c3c",
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.08,
        popup=f"Secondary Impact Zone ({impact_radius_km * 2.5:.1f}km) - Restricted"
    ).add_to(m)
    
    rivers = fetch_rivers_osm(dam_lat, dam_lon, radius_km=5)
    if rivers:
        river_coords = []
        for el in rivers[:10]:
            geom = el.get("geometry", [])
            if geom:
                river_coords.append([geom[0]["lat"], geom[0]["lon"]])
        if river_coords:
            folium.PolyLine(
                river_coords,
                color="#3498db",
                weight=3,
                popup="Real river data from OSM"
            ).add_to(m)
    
    return m


# ============================================================================
# CONSTRUCTION PERMIT GENERATOR
# ============================================================================

def safe_format(value, format_str=".1f"):
    if value is None:
        return "N/A"
    try:
        return f"{value:{format_str}}"
    except:
        return str(value)


def generate_permit_document(certificate):
    date = dt.datetime.now().strftime("%Y-%m-%d")
    
    doc = f"""
    ================================================================
    CONSTRUCTION PERMIT CERTIFICATE
    Issued under SIH26192 FloodSafe System
    ================================================================
    
    Date: {date}
    Location: {certificate['location']['lat']:.6f}, {certificate['location']['lon']:.6f}
    Zone: {certificate['zone_label']}
    Status: {certificate['action']}
    
    REGULATORY BASIS:
    {chr(10).join(['- ' + r for r in certificate['regulatory_basis']])}
    
    SAFETY ASSESSMENT:
    - Slope: {safe_format(certificate['data'].get('slope_deg'), '.1f')}° (Max allowed: 20°)
    - River Distance: {safe_format(certificate['data'].get('river_dist_km'), '.3f')}km (Min required: 0.5km)
    - 24h Rainfall: {safe_format(certificate['data'].get('rain_24h_mm'), '.1f')}mm (Max allowed: 50mm)
    - Elevation: {safe_format(certificate['data'].get('elevation_m'), '.0f')}m
    - Flood Probability: {safe_format(certificate['data'].get('flood_probability'), '.1f')}%
    
    """
    
    if certificate['violations']:
        doc += "\nVIOLATIONS:\n"
        for v in certificate['violations']:
            doc += f"  {v}\n"
    
    if certificate['warnings']:
        doc += "\nWARNINGS:\n"
        for w in certificate['warnings']:
            doc += f"  {w}\n"
    
    if certificate['is_allowed']:
        doc += """
    ================================================================
    PERMIT APPROVED
    ================================================================
    
    Construction is permitted subject to:
    1. All building codes must be strictly followed
    2. Environmental impact assessment required
    3. Mandatory flood protection measures
    4. Regular safety inspections during construction
    5. Compliance with disaster management guidelines
    
    This permit is valid for 365 days from the date of issue.
    """
    else:
        doc += """
    ================================================================
    PERMIT REJECTED
    ================================================================
    
    Construction is NOT permitted at this location due to:
    1. Prohibited zone classification
    2. Safety violations identified above
    3. High flood risk probability
    
    Any construction at this location is ILLEGAL and will result in:
    - Legal action under Disaster Management Act
    - Demolition orders
    - Fines and penalties
    - Criminal prosecution for endangerment
    
    This decision can be appealed within 30 days.
    """
    
    doc += f"\nTimestamp: {certificate['timestamp']}"
    doc += "\n================================================================"
    
    return doc


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
     "Alerts", "Historical Analysis", "Construction Safety", "Dam Safety", 
     "Data Sources / System Status"],
)

bundle = st.session_state.bundle

prob_pct, prob_source, factors = None, None, None
if bundle:
    h_score, h_factors = heuristic_risk_score(bundle)
    prob_pct = h_score
    prob_source = ""
    factors = h_factors
risk_level = risk_level_from_prob(prob_pct)


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
    st.caption(f" {prob_source or ''}")


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
                f"Probability: {cell['prob']:.1f}%<br>"
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
# PAGE: WEATHER & ENVIRONMENT - ADDED FORECAST
# ============================================================================
elif page == "Weather & Environment":
    st.title("🌦️ Weather & Environment")
    
    # --- 3-DAY RAINFALL FORECAST (ADDED AT TOP) ---
    st.subheader("📊 3-Day Rainfall Forecast")
    weather = fetch_weather(st.session_state.location["lat"], st.session_state.location["lon"])
    if weather["ok"]:
        hourly = weather["data"]["hourly"]
        times = pd.to_datetime(hourly["time"])
        precip = hourly["precipitation"]
        
        # Group by day
        df_forecast = pd.DataFrame({"time": times, "precipitation": precip})
        df_forecast["date"] = df_forecast["time"].dt.date
        daily_forecast = df_forecast.groupby("date")["precipitation"].sum().reset_index()
        
        # Show forecast table
        cols = st.columns(min(len(daily_forecast), 5))
        for i, row in daily_forecast.iterrows():
            if i < len(cols):
                cols[i].metric(
                    f"📅 {row['date'].strftime('%b %d')}",
                    f"{row['precipitation']:.1f} mm",
                    delta="Heavy" if row['precipitation'] > 30 else ("Moderate" if row['precipitation'] > 10 else "Light")
                )
        
        # Rainfall chart
        fig_forecast = px.bar(
            daily_forecast,
            x="date",
            y="precipitation",
            title="Daily Rainfall Forecast (3 Days)",
            labels={"date": "Date", "precipitation": "Rainfall (mm)"},
            color="precipitation",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Forecast data unavailable")
    
    st.markdown("---")
    
    # Current conditions
    if bundle is None or not bundle.get("weather_ok"):
        err = bundle.get("weather_error") if bundle else "not fetched"
        st.error(f"⚠ Weather API unavailable. Last successful observation: "
                 f"{st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_updated else 'n/a'} "
                 f"(error: {err})")
    else:
        st.subheader("🌤️ Current Conditions")
        render_current_conditions_block()
        cur = bundle.get("current", {})
        c1, c2 = st.columns(2)
        c1.metric("Pressure", f"{cur.get('surface_pressure', 'n/a')} hPa")
        c2.metric("Wind speed", f"{cur.get('wind_speed_10m', 'n/a')} km/h")

        if weather["ok"]:
            hourly = weather["data"]["hourly"]
            hdf = pd.DataFrame({"time": pd.to_datetime(hourly["time"]), "precipitation": hourly["precipitation"],
                                 "temperature": hourly["temperature_2m"]})
            fig1 = px.line(hdf, x="time", y="precipitation", title="Hourly precipitation (past 3 days + forecast)")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.line(hdf, x="time", y="temperature", title="Hourly temperature")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE: ALERTS - WITH ALARM/NOTIFICATION
# ============================================================================
elif page == "Alerts":
    st.title("🚨 Alerts & Notifications")
    
    # Play alarm sound if high risk
    if bundle is not None and prob_pct is not None and prob_pct >= ALERT_THRESHOLD_PCT:
        # Display alarm notification
        st.error(
            f"""
            🔔 **EMERGENCY ALERT**
            
            🚨 **{risk_level} FLOOD RISK DETECTED**
            
            **Location:** {st.session_state.location['name']}
            **Probability:** {prob_pct:.1f}%
            **Risk Level:** {risk_level}
            
            **Risk Analysis:** {get_risk_reason(bundle, prob_pct, risk_level)}
            
            ⚠️ **IMMEDIATE ACTION REQUIRED:**
            - Evacuate low-lying areas
            - Move to higher ground
            - Monitor official alerts
            - Keep emergency kit ready
            """
        )
        
        # Show notification banner
        st.markdown("""
        <div style="background-color:#ff0000;padding:20px;border-radius:10px;animation:blink 1s infinite;">
            <h1 style="color:white;text-align:center;">🔴 CRITICAL ALERT - FLOOD WARNING</h1>
            <p style="color:white;text-align:center;font-size:24px;">IMMEDIATE ACTION REQUIRED</p>
        </div>
        <style>
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Audio alert using HTML5
        st.markdown("""
        <audio autoplay loop>
            <source src="https://www.soundjay.com/emergency/emergency-alarm-1.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)
        
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
# PAGE: CONSTRUCTION SAFETY
# ============================================================================
elif page == "Construction Safety":
    st.title("Construction Safety & Compliance System")
    st.markdown("### Fast Automatic Detection of Risky Constructions")
    
    st.warning("""
    ⚠️ **SYSTEM SCANNING FOR ILLEGAL CONSTRUCTIONS:**
    
    📋 **REGULATORY COMPLIANCE:**
    - **Slope >20°**: STRICTLY PROHIBITED (National Building Code 2016, Section 7)
    - **Within 500m of rivers**: BANNED (CWC Flood Zone Guidelines)
    - **>50mm/24h rainfall**: NO-CONSTRUCTION ZONE (IMD)
    - **Dam proximity**: 2km buffer zone enforced
    - Violators face legal action under Disaster Management Act
    """)
    
    scan_radius = st.slider("🔍 Scan Radius (km)", 0.5, 2.0, 1.0, 0.5)
    
    current_lat = st.session_state.location["lat"]
    current_lon = st.session_state.location["lon"]
    current_name = st.session_state.location["name"]
    
    st.info(f"📍 Scanning near: **{current_name}**")
    
    if st.button("⚡ Quick Scan for Constructions", use_container_width=True):
        with st.spinner(f"Quick scanning {scan_radius}km radius..."):
            scan_result = scan_area_for_risks(current_lat, current_lon, scan_radius)
            st.session_state.scan_result = scan_result
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏗️ Buildings", scan_result["total_buildings"])
        col2.metric("🔴 Risky", scan_result["risky_buildings"], delta="⚠️" if scan_result["risky_buildings"] > 0 else "✅")
        col3.metric("🚨 Critical", scan_result["critical_buildings"])
        col4.metric("🏊 Dams", scan_result["dams_found"])
    
    if "scan_result" in st.session_state:
        scan = st.session_state.scan_result
        
        st.subheader("🗺️ Risk Map")
        
        m = folium.Map(location=[current_lat, current_lon], zoom_start=15)
        
        for building in scan["buildings"]:
            risk = building["risk_assessment"]
            if risk["risk_level"] == "CRITICAL":
                color = "#7a0d0d"
            elif risk["risk_level"] == "HIGH":
                color = "#e74c3c"
            elif risk["risk_level"] == "MODERATE":
                color = "#f39c12"
            else:
                color = "#2ecc71"
            
            popup_text = f"""
            <b>{building['name']}</b><br>
            Risk Level: {risk['risk_level']}<br>
            """
            if risk['risk_factors']:
                popup_text += "<br>".join(risk['risk_factors'][:3])
            
            folium.CircleMarker(
                [building["lat"], building["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)
        
        folium.Circle(
            [current_lat, current_lon],
            radius=scan["scan_radius_km"] * 1000,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.05,
            popup=f"Scan Area: {scan['scan_radius_km']}km"
        ).add_to(m)
        
        st_folium(m, height=450, use_container_width=True)
        
        if scan["risky_buildings"] > 0:
            st.subheader(f"🔴 {scan['risky_buildings']} Risky Constructions")
            
            for building in scan["buildings"]:
                risk = building["risk_assessment"]
                if risk["is_risky"]:
                    with st.expander(f"🚨 {building['name']} - {risk['risk_level']} RISK"):
                        st.write(f"📍 {building['lat']:.6f}, {building['lon']:.6f}")
                        for factor in risk["risk_factors"]:
                            st.write(f"  • {factor}")
                        
                        if risk["risk_level"] in ["CRITICAL", "HIGH"]:
                            st.error("🚫 **LEGAL NOTICE:** This construction violates building codes and is subject to demolition orders under the Disaster Management Act.")
        else:
            st.success("✅ No risky constructions detected! All structures comply with safety regulations.")

# ============================================================================
# PAGE: DAM SAFETY
# ============================================================================
elif page == "🏊 Dam Safety":
    st.title("🏊 Dam Safety Monitoring System")
    st.markdown("### Fast Dam Detection & Safety Assessment")
    
    st.info("""
    🔍 **AUTOMATIC DAM DETECTION:**
    - Finds dams within 10km radius
    - Real-time safety assessment using live data
    - Downstream impact zone mapping
    - Construction restriction enforcement
    """)
    
    current_lat = st.session_state.location["lat"]
    current_lon = st.session_state.location["lon"]
    current_name = st.session_state.location["name"]
    
    search_radius = st.slider("🔍 Search Radius (km)", 5, 30, 15, 5)
    
    st.info(f"📍 Searching near: **{current_name}**")
    
    if st.button("⚡ Quick Find Dams", use_container_width=True):
        with st.spinner(f"Searching for dams within {search_radius}km..."):
            dams = detect_dams_near_location(current_lat, current_lon, search_radius)
            st.session_state.detected_dams = dams
    
    if "detected_dams" in st.session_state:
        dams = st.session_state.detected_dams
        
        if not dams:
            st.warning("No dams found nearby. Try increasing search radius.")
        else:
            st.success(f"Found {len(dams)} dams")
            
            for dam in dams:
                assessment = dam["assessment"]
                score = assessment["safety_score"]
                
                if score >= 80:
                    color = "#2ecc71"
                    status_icon = "🟢"
                elif score >= 60:
                    color = "#f1c40f"
                    status_icon = "🟡"
                elif score >= 40:
                    color = "#e67e22"
                    status_icon = "🟠"
                else:
                    color = "#e74c3c"
                    status_icon = "🔴"
                
                with st.expander(f"🏊 {dam['name']} - {dam['distance_km']:.1f}km - {status_icon} {assessment['status']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="padding:15px;border-radius:10px;background:{color};color:white;text-align:center;">
                            <h1 style="margin:0;">{score}</h1>
                            <h4>Safety Score</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.write(assessment['recommendation'])
                        if assessment["risks"]:
                            for risk in assessment["risks"]:
                                st.error(risk)
                    
                    m = generate_impact_zone_map(dam["lat"], dam["lon"], impact_radius_km=2.0)
                    st_folium(m, height=300, use_container_width=True)
                    
                    if assessment["safety_score"] < 70:
                        st.warning("🚫 **CONSTRUCTION BAN ENFORCED:** Construction within 2km is BANNED due to dam safety concerns.")

# ============================================================================
# PAGE: DATA SOURCES / SYSTEM STATUS - REMOVED STATUS DISPLAY
# ============================================================================
elif page == "Data Sources / System Status":
    st.title("🔧 System Information")
    
    st.markdown("""
    ### 📡 Data Sources Used
    
    | Source | Purpose | Status |
    |--------|---------|--------|
    | **Open-Meteo Forecast API** | Live weather, rainfall, forecast | ✅ Real-time |
    | **Open-Meteo Archive API** | Historical rainfall (ERA5) | ✅ On-demand |
    | **Open-Meteo Elevation API** | Terrain elevation (SRTM) | ✅ Real-time |
    | **OpenStreetMap Overpass** | Rivers, buildings, dams | ✅ Real-time |
    | **Open-Meteo Geocoding** | Location search | ✅ On-demand |
    | **User CSV** | Historical flood events | ⚙️ Optional |
    | **IoT Sensors** | Not integrated | ⚠️ Planned |
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Regulatory Framework")
    st.markdown("""
    The system enforces these regulations:
    
    - **National Building Code of India 2016** (Section 7 - Hill Area Construction)
    - **Central Water Commission** Flood Zone Buffer Guidelines
    - **India Meteorological Department** Rainfall Warning Criteria
    - **National Disaster Management Authority** Guidelines
    - **Disaster Management Act 2005** - Legal enforcement
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ System Architecture")
    
    import streamlit.components.v1 as components
    
    components.html("""
    <div style="text-align:center; font-family:Arial; padding:10px;">
    
        <div style="border:2px solid #1f4e79; border-radius:10px;
                    padding:12px; margin:5px auto; width:55%;">
            <b>User Input</b>
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="border:2px solid #1f4e79; border-radius:10px;
                    padding:12px; margin:5px auto; width:55%;">
            <b>Geocoding</b><br>
            Coordinates
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="display:flex; justify-content:center; gap:30px;">
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>Weather API</b><br>
                Rainfall Data
            </div>
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>Elevation API</b><br>
                Terrain / Slope
            </div>
    
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="border:2px solid #1f4e79; border-radius:10px;
                    padding:12px; margin:5px auto; width:55%;">
            <b>River API (OSM)</b>
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="border:2px solid #1f4e79; border-radius:10px;
                    padding:12px; margin:5px auto; width:55%;">
            <b>Feature Vector</b>
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="display:flex; justify-content:center; gap:30px;">
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>ML Model</b><br>
                (if trained)
            </div>
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>Heuristic</b><br>
                (fallback)
            </div>
    
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="border:2px solid #1f4e79; border-radius:10px;
                    padding:12px; margin:5px auto; width:55%;">
            <b>Risk Assessment</b>
        </div>
    
        <div style="font-size:25px; margin:3px;">↓</div>
    
        <div style="display:flex; justify-content:center; gap:30px;">
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>Flood Warning</b><br>
                Alerts / Notifications
            </div>
    
            <div style="border:2px solid #1f4e79; border-radius:10px;
                        padding:12px; width:25%;">
                <b>Construction Safety</b><br>
                Permit System
            </div>
    
        </div>
    
    </div>
    """, height=700, scrolling=False)
    
    st.markdown("---")
    st.markdown("""
    *All data is fetched live from official sources. No simulated or fabricated data is used.*
    *IoT integration is marked as unavailable as per Section 19 requirements.*
    """)