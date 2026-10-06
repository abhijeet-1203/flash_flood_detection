# 🌊 Flash Flood Prediction System for Hilly Regions

A real-time Streamlit-based disaster-management and environmental risk assessment platform for monitoring flash-flood conditions in hilly and mountainous regions.

The application combines live weather, terrain, elevation, river, building, dam, and historical flood-event data to provide location-based flood-risk assessment, risk mapping, alerts, construction-safety analysis, and dam-safety monitoring.

The system is designed around **real external data sources**. When a required source is unavailable, the application reports the missing data instead of silently fabricating values.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Data Sources](#-data-sources)
* [Risk Assessment](#-risk-assessment)
* [Live Risk Map](#-live-risk-map)
* [Weather and Environment](#-weather-and-environment)
* [Alerts and Notifications](#-alerts-and-notifications)
* [Historical Flood Analysis](#-historical-flood-analysis)
* [Construction Safety](#-construction-safety)
* [Dam Safety Monitoring](#-dam-safety-monitoring)
* [Historical Dataset](#-historical-dataset)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Running the Application](#-running-the-application)
* [Required Python Packages](#-required-python-packages)
* [How the System Works](#-how-the-system-works)
* [Risk Levels](#-risk-levels)
* [Construction Rules](#-construction-rules)
* [Limitations](#-limitations)
* [Data Reliability](#-data-reliability)
* [Troubleshooting](#-troubleshooting)
* [Future Improvements](#-future-improvements)

---

# 🌊 Overview

The Flash Flood Prediction System is a **single-file Streamlit application** that acts as a location-aware flood monitoring and safety command center.

A user can:

1. Search for a village, district, or place.
2. Select a location returned by the geocoding service.
3. Enter custom latitude and longitude coordinates.
4. Fetch live environmental information.
5. View current rainfall and weather conditions.
6. Calculate a flood-risk score.
7. Generate a multi-cell live risk map.
8. View rainfall forecasts.
9. Receive threshold-based flood alerts.
10. Analyze genuine historical flood events.
11. Scan nearby buildings for potential construction risks.
12. Detect nearby dams.
13. Assess dam safety.
14. Generate construction-safety assessments and permit information.
15. Inspect the application's data sources and system architecture.

The application uses Streamlit session state to maintain the selected location, fetched feature bundle, update timestamp, risk grid, building scan results, and detected dams.

---

# ✨ Key Features

## 1. 📍 Location Search

The application supports location selection through:

* Village search
* District search
* Place-name search
* Manual latitude/longitude entry

Location names are resolved using the **Open-Meteo Geocoding API**.

Users can also directly provide coordinates for locations that may not be easily searchable.

---

## 2. 🌦️ Live Weather Monitoring

The application retrieves live weather information using the Open-Meteo Forecast API.

The system obtains:

* Temperature
* Relative humidity
* Precipitation
* Rain
* Surface pressure
* Wind speed
* Weather code
* Hourly precipitation
* Hourly temperature
* Hourly humidity
* Hourly pressure

The application requests both previous and upcoming weather information to support rainfall accumulation and forecasting.

---

## 3. 🌧️ Rainfall Accumulation

The system calculates:

* 24-hour rainfall
* 72-hour rainfall
* 7-day rainfall

These values are used as important inputs to the flood-risk calculation.

The live weather request uses approximately three days of past data and three days of forecast data.

---

## 4. ⛰️ Elevation and Terrain Analysis

Elevation is retrieved through the Open-Meteo Elevation API.

Slope is not taken from a fabricated dataset.

Instead, the application obtains multiple real elevation measurements around the selected coordinate and calculates an approximate slope from the elevation differences.

The main terrain features are:

* Elevation in metres
* Estimated slope in degrees

---

## 5. 🏞️ River and Waterway Analysis

The application queries OpenStreetMap through the Overpass API to find:

* Rivers
* Streams
* Waterways

The distance from the selected location to the nearest returned river/stream geometry is calculated using the Haversine distance formula.

This produces:

```text
river_dist_km
```

River proximity contributes to the flood-risk score and is also used in construction-safety analysis.

---

# 🧠 Risk Assessment

The application uses six primary environmental features:

| Feature         | Description                                          |
| --------------- | ---------------------------------------------------- |
| `rain_24h_mm`   | Rainfall accumulated over approximately 24 hours     |
| `rain_72h_mm`   | Rainfall accumulated over approximately 72 hours     |
| `rain_7d_mm`    | Rainfall accumulated over the available 7-day period |
| `elevation_m`   | Elevation above sea level                            |
| `slope_deg`     | Estimated terrain slope                              |
| `river_dist_km` | Distance to nearest mapped river/stream              |

These features form the application's environmental feature bundle.

---

# 📊 Rule-Based Risk Score

When the live system is operating from current environmental information, the implemented risk calculation uses a transparent weighted heuristic.

### 24-hour rainfall

Maximum contribution:

```text
40 points
```

Formula:

```text
min(rain_24h / 100, 1) × 40
```

---

### 72-hour rainfall

Maximum contribution:

```text
25 points
```

Formula:

```text
min(rain_72h / 200, 1) × 25
```

---

### River proximity

Maximum contribution:

```text
20 points
```

The closer the location is to a river/stream, the larger the contribution.

Conceptually:

```text
(10 - river_distance) / 10 × 20
```

with the distance capped at 10 km.

---

### Terrain slope

Maximum contribution:

```text
15 points
```

Formula:

```text
min(slope / 30, 1) × 15
```

---

### Maximum score

```text
40 + 25 + 20 + 15 = 100
```

The final value is capped at 100.

The application also displays the individual contributions so that users can understand why the score was produced.

> **Important:** This score is a risk score based on the application's documented heuristic. It should not be interpreted as a statistically calibrated probability unless a validated trained model is actually being used.

---

# 🚦 Risk Levels

The system classifies the resulting score into five levels:

|    Score | Risk Level  |
| -------: | ----------- |
|  0–19.99 | 🟢 LOW      |
| 20–39.99 | 🟡 MODERATE |
| 40–59.99 | 🟠 ELEVATED |
| 60–79.99 | 🔴 HIGH     |
|   80–100 | 🔴 CRITICAL |

If the required rainfall inputs are unavailable, the system can return an unavailable/unknown state instead of inventing a value.

---

# 🗺️ Live Risk Map

The **Live Risk Map** generates a grid around the selected location.

Users can configure:

* Grid resolution: **3–9 cells per side**
* Map span: **10–60 km**

Each grid cell receives its own:

* Coordinates
* Rainfall
* Elevation
* Estimated slope
* River distance
* Risk score
* Risk level

The resulting cells are rendered as coloured rectangles on an interactive Folium map.

The grid is designed to use batched API requests instead of making an independent weather request for every individual cell.

Each cell's risk is calculated from its own retrieved environmental features.

---

# 🌦️ Weather & Environment

The Weather & Environment page provides a detailed view of the environmental conditions at the selected location.

## 3-Day Rainfall Forecast

The application:

1. Retrieves hourly precipitation.
2. Converts timestamps into dates.
3. Groups precipitation by day.
4. Calculates daily rainfall totals.
5. Displays daily rainfall metrics.
6. Generates a rainfall chart.

The page also provides:

* Current temperature
* Humidity
* Atmospheric pressure
* Wind speed
* Hourly precipitation graph
* Hourly temperature graph

If the weather API fails, the interface explicitly reports that forecast/current weather data is unavailable.

---

# 🚨 Alerts & Notifications

The Alerts page compares the current risk score against a configurable threshold.

The default threshold is:

```text
60%
```

The threshold can be changed through:

```env
ALERT_THRESHOLD_PCT=60
```

If the current score reaches or exceeds the threshold, the application displays an emergency alert containing:

* Location
* Risk level
* Current score
* Risk explanation
* Recommended immediate actions

The alert interface also provides a visual warning banner and an HTML audio alarm.

Recommended actions shown by the application include:

* Moving away from low-lying areas
* Moving toward higher ground
* Monitoring official alerts
* Keeping an emergency kit ready

The threshold is configurable through the environment configuration.

---

# 📜 Historical Flood Analysis

The Historical Analysis page works with a user-provided CSV containing genuine flood events.

Required fields:

```text
date
latitude
longitude
```

Optional field:

```text
location_name
```

Example:

```csv
date,latitude,longitude,location_name
2025-07-18,30.3165,78.0322,Example Location
```

The application:

* Loads the CSV
* Validates required columns
* Converts dates into datetime values
* Removes invalid records
* Displays the event table
* Plots events on an interactive map
* Shows seasonal flood frequency
* Shows flood events per year

If the dataset does not exist or does not contain the required columns, the application reports that the historical dataset is unavailable rather than fabricating events.

---

# 📁 Historical Dataset Sources

A genuine historical flood-event dataset can be supplied from appropriate public sources, such as:

* NDMA / NRSC flood inventories
* Dartmouth Flood Observatory
* data.gov.in disaster-management datasets
* State disaster-management records
* Other appropriately licensed public flood-event datasets

The application itself does not bundle fabricated historical events.

---

# 🏗️ Construction Safety

The Construction Safety page provides a location-based assessment of construction conditions.

The system evaluates:

* Terrain slope
* Distance from rivers
* Rainfall
* Elevation
* Current flood-risk score
* Nearby historical flood events

The system classifies locations into:

### 🟢 SAFE

Construction is considered allowed according to the application's configured checks.

### 🟡 RESTRICTED

Warnings exist and additional safety measures or special permits may be required.

### 🔴 PROHIBITED

One or more configured safety conditions have been violated.

The configured construction parameters include:

| Parameter                       | Configured value |
| ------------------------------- | ---------------: |
| Maximum slope                   |              20° |
| Minimum river distance          |           0.5 km |
| Maximum 24h rainfall            |            50 mm |
| Maximum 72h rainfall            |           100 mm |
| Minimum elevation warning level |            500 m |
| Flood-buffer distance           |             1 km |

These are application configuration values used for the assessment logic.

---

# 🔍 Automatic Construction Detection

The application can scan the selected area for mapped buildings using OpenStreetMap Overpass.

The scan:

1. Queries buildings within the selected radius.
2. Obtains their coordinates.
3. Reads available OSM building information.
4. Evaluates environmental conditions around each building.
5. Assigns a risk score.
6. Classifies the building as LOW, MODERATE, HIGH, or CRITICAL.

The application can display:

* Total buildings detected
* Risky buildings
* Critical buildings
* Building locations
* Risk factors
* Interactive map markers

The scan is based on buildings present in OpenStreetMap; it does not claim to detect every physical building that exists.

---

# 🏊 Dam Safety Monitoring

The Dam Safety page searches OpenStreetMap for nearby dams.

For each detected dam, the application evaluates:

* 24-hour rainfall
* 72-hour rainfall
* Terrain slope
* Elevation
* River distance
* Nearby historical flood events
* Current flood-risk score

A starting safety score of 100 is reduced when configured risk conditions are encountered.

The final classification is:

|    Score | Status        |
| -------: | ------------- |
|   80–100 | 🟢 SAFE       |
|    60–79 | 🟡 MONITORING |
|    40–59 | 🟠 ELEVATED   |
| Below 40 | 🔴 CRITICAL   |

The system also produces recommendations such as increased monitoring or immediate inspection depending on the calculated score.

The implementation additionally generates an impact-zone map around a detected dam.

---

# 🗺️ Dam Impact-Zone Mapping

The impact-zone map contains:

* Dam location marker
* Primary impact zone
* Secondary impact zone
* Available river geometry

The configured primary impact radius is:

```text
2 km
```

The secondary zone extends to:

```text
2 × 2.5 = 5 km
```

The map is rendered using Folium.

---

# 🏛️ Regulatory Framework Used by the Application

The application references the following frameworks in its safety and permit interface:

* National Building Code of India 2016
* Central Water Commission flood-zone guidance
* India Meteorological Department rainfall criteria
* National Disaster Management Authority guidelines
* Disaster Management Act 2005

These references are presented as the application's regulatory basis; users should verify the current official regulations and local authority requirements before making real-world legal, construction, evacuation, or safety decisions.

---

# 📄 Construction Permit Generation

The application contains a permit-document generator.

The generated document can include:

* Date
* Location
* Construction zone
* Permit status
* Regulatory basis
* Slope
* River distance
* Rainfall
* Elevation
* Flood-risk score
* Violations
* Warnings
* Approval/rejection information
* Timestamp

The permit output is generated as application text and is intended as an assessment/documentation feature rather than a replacement for an official government permit process.

---

# 🧩 System Architecture

The overall processing pipeline is:

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    │ Place / Coordinates  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Geocoding        │
                    │  Location → Lat/Lon  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Weather API  │ │ Elevation API│ │ OSM Overpass │
      │ Rain/Temp/etc│ │ Elevation    │ │ Rivers/etc.  │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │   Feature Bundle     │
                    │ Rainfall / Elevation │
                    │ Slope / River Dist.  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Risk Assessment    │
                    │ Rule-based scoring   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼──────────────────┐
              │                │                  │
              ▼                ▼                  ▼
       ┌────────────┐   ┌─────────────┐   ┌──────────────┐
       │ Flood Risk │   │ Alerts      │   │ Safety Tools │
       │ & Maps     │   │             │   │ Construction │
       └────────────┘   └─────────────┘   │ Dam Safety   │
                                          └──────────────┘
```

---

# 📡 Data Sources

| Data                | Source                         | Purpose                       |
| ------------------- | ------------------------------ | ----------------------------- |
| Current weather     | Open-Meteo Forecast API        | Live environmental conditions |
| Hourly weather      | Open-Meteo Forecast API        | Rainfall/temperature analysis |
| Historical rainfall | Open-Meteo Archive API         | Historical rainfall retrieval |
| Elevation           | Open-Meteo Elevation API       | Terrain elevation             |
| Slope               | Derived from elevation samples | Terrain analysis              |
| Location search     | Open-Meteo Geocoding API       | Place → coordinates           |
| Rivers/streams      | OpenStreetMap Overpass API     | River proximity               |
| Buildings           | OpenStreetMap Overpass API     | Construction detection        |
| Dams                | OpenStreetMap Overpass API     | Dam detection                 |
| Flood events        | User-provided CSV              | Historical analysis           |

The code directly implements Open-Meteo Forecast, Archive, Elevation and Geocoding requests and OpenStreetMap Overpass queries.

---

# 🛡️ No Fabricated Live Values

A major design principle of the application is that unavailable external data should remain unavailable.

For example:

```text
Live data unavailable
```

may be displayed when an API request fails.

Similarly, if no historical CSV is available, the Historical Analysis page reports that the dataset is unavailable.

This prevents the interface from presenting invented environmental observations as real measurements.


```

### `app2.py`

Main Streamlit application containing:

* API integration
* Data processing
* Risk calculations
* Mapping
* Alerts
* Construction assessment
* Dam assessment
* User interface

### `data/`

Stores user-provided historical flood-event data.

### `models/`

Reserved for trained model artifacts and model metadata where applicable.

### `.env`

Stores configurable environment variables.

---

# ⚙️ Configuration

The application reads environment variables through `python-dotenv`.

Example:

```env
HISTORICAL_FLOOD_CSV=data/historical_floods.csv
ALERT_THRESHOLD_PCT=60
OWM_API_KEY=
```

## Historical dataset path

Default:

```env
HISTORICAL_FLOOD_CSV=data/historical_floods.csv
```

You can provide a different location:

```env
HISTORICAL_FLOOD_CSV=data/my_flood_events.csv
```

## Alert threshold

Default:

```env
ALERT_THRESHOLD_PCT=60
```

Example:

```env
ALERT_THRESHOLD_PCT=70
```

This changes the threshold at which the Alerts page displays an emergency alert.

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd flash-flood-prediction
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create environment configuration

Copy the example file:

```bash
cp .env.example .env
```

On Windows, you can simply create a `.env` file manually.

Example:

```env
HISTORICAL_FLOOD_CSV=data/historical_floods.csv
ALERT_THRESHOLD_PCT=60
```

---

## 5. Add historical data if available

Create:

```text
data/historical_floods.csv
```

with:

```csv
date,latitude,longitude,location_name
2025-07-18,30.3165,78.0322,Example Location
```

Only use genuine historical event records.

---

# ▶️ Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

# 📦 Required Python Packages

The application imports the following major packages:

```text
requests
numpy
pandas
streamlit
folium
streamlit-folium
plotly
python-dotenv
scikit-learn
joblib
xgboost
shap
```

A corresponding `requirements.txt` can contain:

```txt
requests
numpy
pandas
streamlit
folium
streamlit-folium
plotly
python-dotenv
scikit-learn
joblib
xgboost
shap
```

`xgboost` and `shap` are imported conditionally in the application, allowing the application to detect whether they are available.

---

# 🖥️ Application Pages

The sidebar provides the following navigation options:

```text
1. Overview
2. Live Risk Map
3. AI Flood Prediction
4. Weather & Environment
5. Alerts
6. Historical Analysis
7. Construction Safety
8. Dam Safety
9. Data Sources / System Status
```

---

## 1. Overview

Provides a command-center style summary containing:

* Current location
* Current flood-risk level
* Risk score
* 24-hour rainfall
* 72-hour rainfall
* Temperature
* Humidity
* Contributing risk factors
* Confidence information
* Quick risk map

---

## 2. Live Risk Map

Provides:

* Configurable grid
* Real-time environmental inputs
* Risk classification per grid cell
* Interactive Folium map
* Rainfall information
* Elevation
* Slope
* River distance

---

## 3. AI Flood Prediction

Displays the current feature snapshot:

```text
rain_24h_mm
rain_72h_mm
rain_7d_mm
elevation_m
slope_deg
river_dist_km
```

It also displays the contribution breakdown used by the current risk-scoring logic.

---

## 4. Weather & Environment

Provides:

* Three-day rainfall forecast
* Daily rainfall chart
* Current weather
* Pressure
* Wind speed
* Historical + forecast hourly precipitation
* Hourly temperature

---

## 5. Alerts

Provides threshold-based flood warnings.

The alert threshold is configurable through:

```env
ALERT_THRESHOLD_PCT
```

---

## 6. Historical Analysis

Provides:

* Historical event table
* Historical event map
* Seasonal distribution
* Annual event distribution

The page only operates when a valid historical dataset is supplied.

---

## 7. Construction Safety

Provides:

* Construction rule assessment
* Building detection
* Risk classification
* Risk map
* Safety warnings
* Construction restrictions
* Permit assessment/document generation

---

## 8. Dam Safety

Provides:

* Nearby dam detection
* Dam safety score
* Risk assessment
* Monitoring recommendations
* Impact-zone maps
* Construction restrictions around higher-risk dams

---

## 9. Data Sources / System Status

Provides information about:

* APIs used
* Historical data source
* IoT availability
* Regulatory framework
* System architecture

---

# 🔄 Application Workflow

When the application starts, it initializes a default location and creates the required data/model directories.

The user can then select a location through the sidebar.

The selected location is converted into a coordinate pair:

```text
Latitude
Longitude
```

The application then builds a live feature bundle.

```text
Location
   ↓
Weather
   ↓
Elevation
   ↓
Slope
   ↓
River data
   ↓
Feature bundle
   ↓
Risk score
   ↓
Risk level
   ↓
Maps / Alerts / Safety analysis
```

---

# 💾 Caching

The application uses Streamlit caching for API-heavy operations.

Examples include:

* Weather
* Batch weather
* Historical rainfall
* Elevation
* River queries
* Building queries
* Dam queries

This reduces unnecessary repeated API requests and improves application responsiveness.

---

# 🌐 API Failure Handling

External APIs can fail because of:

* Network problems
* API downtime
* Rate limits
* Timeout
* Invalid coordinates
* Empty OSM results

The application handles these conditions by returning unavailable data rather than generating replacement values.

For example:

```text
Live data unavailable
```

or:

```text
No dams found nearby
```

may be shown when the external service cannot provide usable information.

---

# ⚠️ Important Limitations

## 1. API dependency

The application depends on publicly accessible external APIs.

Internet connectivity is therefore required for live operation.

---

## 2. OpenStreetMap coverage

River, building, and dam results depend on the quality and completeness of OpenStreetMap data.

An unmapped feature may not appear in the application.

---

## 3. Overpass rate limits

OpenStreetMap Overpass servers are shared public services.

Large numbers of requests may result in:

* Timeouts
* Empty results
* Temporary failures
* Rate limiting

---

## 4. Historical flood data

The application does not automatically possess a complete historical flood-event database.

A valid CSV must be supplied for historical-event analysis.

---

## 5. Terrain slope

Slope is derived from elevation samples rather than obtained as a dedicated slope dataset.

Consequently, it is an approximation of local terrain steepness.

---

## 6. Risk score interpretation

The current live interface calculates the displayed risk using a transparent rule-based scoring system.

Therefore:

> The displayed score should be interpreted as a risk index generated from the configured environmental rules, not automatically as a calibrated statistical probability of flooding.

---

## 7. Regulatory decisions

Construction and dam-safety outputs are decision-support features.

They should not replace:

* Government permits
* Engineering surveys
* Structural inspections
* Official flood warnings
* Local authority decisions
* Emergency-management instructions

---

# 🔐 Privacy and Credentials

Do not commit `.env` files containing private credentials.

Add:

```text
.env
```

to `.gitignore`.

Example:

```gitignore
.env
__pycache__/
*.pyc
models/
```

If a future version uses authenticated APIs, keep credentials inside environment variables or the hosting platform's secret manager.

---

# 🚀 Deployment

The application can be deployed on a standard Python/Streamlit-compatible environment.

Typical deployment flow:

```text
Repository
    ↓
Install requirements
    ↓
Configure environment variables
    ↓
Provide historical dataset if required
    ↓
Start Streamlit
    ↓
Application available through browser
```

For hosted deployment, configure environment variables through the platform's secret/environment-variable system rather than committing `.env`.

---

# 🧪 Testing Checklist

Before deployment, verify:

### Location

* [ ] Place search works
* [ ] Coordinates can be entered manually
* [ ] Invalid location handling works

### Weather

* [ ] Current weather loads
* [ ] 24h rainfall is displayed
* [ ] 72h rainfall is displayed
* [ ] Forecast loads
* [ ] API failure is handled

### Terrain

* [ ] Elevation loads
* [ ] Slope is calculated
* [ ] Missing elevation is handled

### Rivers

* [ ] OSM query works
* [ ] River distance is calculated
* [ ] Empty river results are handled

### Risk

* [ ] Risk score is generated
* [ ] Risk level is classified
* [ ] Contributing factors are displayed
* [ ] Missing data does not create fake values

### Maps

* [ ] Quick map loads
* [ ] Risk grid loads
* [ ] Grid cells display correct risk levels

### Alerts

* [ ] Alert threshold works
* [ ] High-risk alert appears
* [ ] Normal-risk message appears

### Historical Data

* [ ] CSV loads correctly
* [ ] Required columns are validated
* [ ] Historical map works
* [ ] Monthly analysis works
* [ ] Yearly analysis works

### Construction

* [ ] Building scan works
* [ ] Building risk assessment works
* [ ] Construction warnings appear correctly

### Dams

* [ ] Dam search works
* [ ] Safety score is calculated
* [ ] Impact map loads

---

# 🧭 Future Improvements

Possible future enhancements include:

* Integration with official real-time sensor networks
* More advanced hydrological modelling
* Official administrative boundary datasets
* Catchment-level analysis
* River-level monitoring
* Soil-moisture information
* Satellite-derived flood observations
* More detailed terrain modelling
* Improved calibration using larger historical datasets
* Automated model retraining
* Additional explainability visualizations
* Mobile-friendly emergency notifications
* SMS/email notification integrations
* Integration with official emergency-management feeds
* Role-based dashboards for administrators and field officers

---

# 📚 Technology Stack

| Layer                       | Technology             |
| --------------------------- | ---------------------- |
| Frontend                    | Streamlit              |
| Mapping                     | Folium                 |
| Charts                      | Plotly                 |
| Data Processing             | Pandas, NumPy          |
| HTTP/API Communication      | Requests               |
| Configuration               | python-dotenv          |
| Machine Learning Components | Scikit-learn           |
| Optional ML                 | XGBoost                |
| Explainability              | SHAP                   |
| Model Serialization         | Joblib                 |
| Weather                     | Open-Meteo             |
| Elevation                   | Open-Meteo             |
| Geocoding                   | Open-Meteo             |
| Geographic Data             | OpenStreetMap Overpass |

The source code imports these core libraries and conditionally detects XGBoost and SHAP availability.

---

# 📜 License

Add the appropriate project license here, for example:

```text
MIT License
```

if the repository is intended to be released under the MIT License.

Also review the terms and attribution requirements of all external data providers before redistributing data or deploying the application commercially.

---

# ⚠️ Disclaimer

This application is a **technical decision-support and monitoring system**.

It is not a replacement for official emergency warnings, government authorities, qualified engineers, hydrologists, meteorologists, or disaster-management professionals.

For an actual emergency, follow instructions issued by the appropriate local authorities and emergency services.

---

# 🌊 Summary

The Flash Flood Prediction System combines:

```text
Live Weather
     +
Historical Rainfall
     +
Elevation
     +
Terrain Slope
     +
River Proximity
     +
OpenStreetMap Features
     +
Historical Flood Events
     ↓
Risk Assessment
     ↓
Interactive Maps
     ↓
Flood Alerts
     ↓
Construction Safety
     +
Dam Safety Monitoring
```

The application is designed to make environmental risk information accessible through a single interactive Streamlit interface while maintaining explicit handling of unavailable data.
