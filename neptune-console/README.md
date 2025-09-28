# Neptune Console - React Version

A real-time maritime telemetry and incident management console built with React, providing superior performance and real-time capabilities compared to the Streamlit version.

## Features

### ✅ Completed
- **Real-time telemetry charts** with Chart.js - Three separate, properly scaled charts for:
  - Oil Fluorescence (ppb)
  - Chlorophyll (μg/L)
  - Backscatter (m⁻¹sr⁻¹)
- **Interactive ship tracking map** with Leaflet showing live vessel position and track
- **Incident management** with list view and detailed incident pages
- **Auto-refresh functionality** with configurable intervals (1s, 2s, 5s, 10s)
- **Navigation system** - starts on telemetry, click incidents to view details, back button to return
- **Alert system** for oil alarms and warnings
- **Responsive design** with modern UI/UX
- **Data server** to serve NDJSON telemetry data and incidents via REST API

### 🔧 Architecture

**Frontend (React + TypeScript):**
- `src/App.tsx` - Main application component with state management
- `src/components/TelemetryCharts.tsx` - Real-time Chart.js charts
- `src/components/ShipMap.tsx` - Leaflet map with ship tracking
- `src/components/IncidentList.tsx` - Incident listing and selection
- `src/components/IncidentDetail.tsx` - Detailed incident view
- `src/components/Sidebar.tsx` - Configuration and controls
- `src/types.ts` - TypeScript type definitions

**Backend (Python):**
- `data-server.py` - HTTP server providing REST API for telemetry and incident data
- Endpoints:
  - `GET /telemetry?tail=N&since=timestamp` - Live telemetry data
  - `GET /incidents` - Incident list
  - `GET /health` - Server health check

## Setup Instructions

### Quick Start

```bash
# From the Neptune project root
./start-react-console.sh
```

This will start the data server and give you instructions for the React app.

### Manual Setup

### 1. Start the Data Server

```bash
# From the Neptune project root
uv run python data-server.py
```

The server will run on http://localhost:8000 and serve:
- Live telemetry data from `data/ship/seaowl_live.ndjson`
- Incident data from `artifacts/` directory

### 2. Install Dependencies (first time only)

```bash
cd neptune-console
npm install
# or
yarn install
```

**Note:** If you have npm authentication issues, you can resolve dependencies and run the app once the packages are properly installed.

### 3. Start the React App

```bash
npm run dev
# or
yarn dev
```

The React app will run on http://localhost:3000

## Key Advantages over Streamlit

1. **True Real-time Updates** - No refresh delays, smooth chart animations
2. **Better Performance** - Handles large datasets efficiently, optimized rendering
3. **Superior UX** - Responsive design, fast navigation, no page reloads
4. **Extensible** - Easy to add new features, components, and integrations
5. **Production Ready** - Can be built and deployed as static assets

## Dependencies

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "chart.js": "^4.4.0",
  "react-chartjs-2": "^5.2.0",
  "leaflet": "^1.9.4",
  "react-leaflet": "^4.2.1",
  "axios": "^1.6.0",
  "typescript": "^5.2.2",
  "vite": "^5.0.8"
}
```

## API Endpoints

### GET /telemetry
Returns recent telemetry data points.

**Query Parameters:**
- `tail` (optional): Number of recent points to return (default: 100)
- `since` (optional): ISO timestamp to get data since

**Response:**
```json
[
  {
    "ts": "2025-09-28T03:35:37.715553Z",
    "lat": 40.704152,
    "lon": -73.987475,
    "oil_fluor_ppb": 0.0975,
    "chlorophyll_ug_per_l": 2.0695,
    "backscatter_m-1_sr-1": 0.00177,
    ...
  }
]
```

### GET /incidents
Returns list of incidents from artifacts directory.

**Response:**
```json
[
  {
    "incident_id": "incident_001",
    "scenario": "suspected_oil_spill",
    "confidence": 0.85,
    "summary": "Elevated oil fluorescence detected...",
    "ts_peak": "2025-09-28T03:30:00Z"
  }
]
```

## Development

The React app is configured with:
- **Vite** for fast development and building
- **TypeScript** for type safety
- **Hot reload** for development efficiency
- **CORS-enabled** data server for local development

To add new features:
1. Add new components in `src/components/`
2. Update types in `src/types.ts`
3. Extend API endpoints in `data-server.py` if needed

## Comparison with Streamlit Version

| Feature | Streamlit | React |
|---------|-----------|-------|
| Real-time updates | Problematic, 15s delays | Smooth, 1s intervals |
| Chart performance | Poor with large datasets | Optimized rendering |
| Navigation | Page reloads | Instant transitions |
| Customization | Limited | Fully customizable |
| Mobile support | Basic | Responsive design |
| Production deployment | Complex | Static build + CDN |
| Developer experience | Python-centric | Modern web dev tools |

The React version provides a significantly better user experience with true real-time capabilities and professional-grade performance.