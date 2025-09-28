import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { TelemetryData } from '../types';

// Fix for default markers in React Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface Props {
  data: TelemetryData[];
  onExplain?: () => void;
}

const ShipMap: React.FC<Props> = ({ data, onExplain }) => {
  const mapRef = useRef<any>(null);

  // Filter data with valid coordinates
  const validData = data.filter(d => d.lat && d.lon);

  // Get the last 50 points for the track
  const trackData = validData.slice(-50);
  const latestPoint = validData[validData.length - 1];

  // Create path coordinates
  const pathCoordinates: [number, number][] = trackData.map(d => [d.lat, d.lon]);

  // Calculate map bounds
  const lats = trackData.map(d => d.lat);
  const lons = trackData.map(d => d.lon);
  const boundsPadding = 0.01;
  const bounds: [[number, number], [number, number]] = validData.length > 0 ? [
    [Math.min(...lats) - boundsPadding, Math.min(...lons) - boundsPadding],
    [Math.max(...lats) + boundsPadding, Math.max(...lons) + boundsPadding]
  ] : [[0, 0], [0, 0]];

  // Auto-fit map when data changes - ALWAYS call useEffect
  useEffect(() => {
    if (mapRef.current && trackData.length > 1) {
      mapRef.current.fitBounds(bounds, { padding: [32, 32] });
    }
  }, [trackData.length, bounds]);

  // Early return AFTER all hooks
  if (validData.length === 0) {
    return (
      <div className="map-container">
        <h3>Ship Track</h3>
        <div className="alert alert-info">
          No geolocated samples yet...
        </div>
      </div>
    );
  }

  return (
    <div className="map-container">
      <div className="map-header">
        <h3>Ship Track</h3>
        {onExplain && (
          <button className="btn btn-secondary" onClick={() => onExplain()}>
            Explain anomaly
          </button>
        )}
      </div>
      <div style={{ height: '300px', width: '100%' }}>
        <MapContainer
          ref={mapRef}
          center={[latestPoint.lat, latestPoint.lon]}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />

          {/* Ship track */}
          {pathCoordinates.length > 1 && (
            <Polyline
              positions={pathCoordinates}
              color="#60a5fa"
              weight={4}
              opacity={0.9}
            />
          )}

          {/* Current position marker */}
          <Marker position={[latestPoint.lat, latestPoint.lon]}>
            <Popup>
              <div>
                <strong>Current Position</strong><br />
                Lat: {latestPoint.lat.toFixed(6)}<br />
                Lon: {latestPoint.lon.toFixed(6)}<br />
                Time: {new Date(latestPoint.ts).toLocaleTimeString()}<br />
                Oil: {latestPoint.oil_fluor_ppb.toFixed(3)} ppb
              </div>
            </Popup>
          </Marker>
        </MapContainer>
      </div>
    </div>
  );
};

export default ShipMap;
