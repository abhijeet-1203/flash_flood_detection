# SIH26192 — Flash Flood Prediction System for Hilly Regions
**Team:** NoSemicolons · **Theme:** Disaster Management · **Category:** Software

A single-file Streamlit disaster-management command center. Every number the
app displays comes from a real, live API or a genuine dataset you supply —
nothing is hardcoded, randomly generated, or simulated. Where a real source
is missing, the app says so explicitly instead of inventing a value.

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # edit if needed (defaults work out of the box)
streamlit run app.py
```

No API key is required to run the app immediately — the primary weather,
elevation and geocoding sources (Open-Meteo) are free and keyless.

## 2. Real data sources used

| Data | Source | Auth | Notes |
|---|---|---|---|
| Live weather, hourly forecast | [Open-Meteo Forecast API](https://open-meteo.com/) | none | rainfall, temp, humidity, pressure, wind |
| Historical weather (for training) | [Open-Meteo Archive API](https://open-meteo.com/) | none | ERA5 reanalysis, real observed rainfall |
| Elevation | [Open-Meteo Elevation API](https://open-meteo.com/) | none | SRTM-derived |
| Slope | derived from 4 real elevation samples around each point | none | not a separate API — computed from the real elevation source above |
| Rivers / waterways | [OpenStreetMap Overpass API](https://overpass-api.de/) | none | live query, real OSM geometry |
| Village/district search | [Open-Meteo Geocoding API](https://open-meteo.com/) | none | |
| Historical flood events | **you supply this** — see below | n/a | required for real ML training |
| IoT sensors | not integrated | n/a | Section 19 of the spec forbids simulating this, so it always reports "unavailable" rather than being faked |

## 3. Historical flood dataset — required for ML

The app will **not** fabricate flood-event history or a fake model accuracy.
To enable real ML training, place a genuine historical flood-events CSV at:

```
data/historical_floods.csv
```

Required columns: `date` (YYYY-MM-DD), `latitude`, `longitude`.
Optional: `location_name`. A schema template (no data) is at
`data/historical_floods_TEMPLATE.csv`.

Real public sources you can use for hilly-region India flood events:
- NDMA / NRSC (ISRO) India Flood Inventory
- Dartmouth Flood Observatory global archive
- data.gov.in disaster-management datasets
- State disaster management authority incident reports

**Until this file exists**, the Historical Analysis and Model Performance
pages clearly state the dataset is unavailable, and the AI Prediction pages
fall back to a **transparent, documented rule-based heuristic** built only
from live real inputs (rainfall, river distance, slope) — its exact formula
and weights are shown in the UI. This is intentionally not called an "ML
probability," per the project's no-fabrication requirement.

## 4. How training works (no data leakage)

1. Each real flood event's date/location is used to pull genuine historical
   rainfall (1/3/7-day accumulation) from the Open-Meteo Archive API.
2. Negative (non-flood) samples are built from the **same real locations**,
   on real dates at least 45 days from any recorded event, using the same
   real archive weather source — never synthetic values.
3. Rows are sorted chronologically and split **temporally**: oldest ~70%
   train, next ~15% validate model choice, most recent ~15% (untouched
   until the very end) test the final model.
4. RandomForest and XGBoost are both trained; the model is selected on
   **validation** Recall + PR-AUC + F1 (not raw accuracy, since missing a
   real flood event is the costliest error), then scored once on the
   held-out test set.
5. SHAP explains each prediction; if SHAP isn't available or the model
   isn't trained, the app falls back to the rule-based heuristic above.

## 5. Limitations (stated plainly, not hidden)

- **Historical flood data is not bundled.** This is a licensing/availability
  constraint, not a shortcut — the alternative would be fabricated data,
  which the spec explicitly forbids.
- **Administrative boundary polygons** (district/village shapefiles) are not
  bundled either; the app uses point-based geocoding + a computed grid
  around the selected point for the Live Risk Map rather than official
  boundary polygons. Wiring in a real GeoBoundaries/Survey of India shapefile
  is a natural next step — drop a GeoJSON into `data/` and extend
  `render` for the map page.
- **Overpass API** is a shared public instance with rate limits; under load
  it may return no rivers for a query, which the app reports as "River Data
  unavailable" rather than guessing a distance.
- **No live IoT/sensor network** is connected. The spec explicitly prohibits
  simulating one, so this always shows "No live sensor data available."
- The rule-based heuristic is a stopgap, not a substitute for the real model
  — it exists only so the app never displays a fabricated ML probability
  before you've supplied real historical training data.
- This was built and syntax-checked in a sandboxed environment without
  outbound access to the live APIs above; you should do a first live run
  yourself to confirm connectivity from your network.

## 6. Deployment

Any standard Streamlit host works (Streamlit Community Cloud, a VM, or a
container) since there's no database dependency — trained model artifacts
are cached to `models/` on disk. Set the same `.env` variables in your
hosting platform's secrets manager rather than committing `.env`.
