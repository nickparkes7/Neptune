export interface TelemetryData {
  ts: string;
  lat: number;
  lon: number;
  depth_m: number;
  platform_id: string;
  sensor_id: string;
  sensor_type: string;
  sample_rate_hz: number;
  mode: string;
  oil_fluor_ppb: number;
  chlorophyll_ug_per_l: number;
  "backscatter_m-1_sr-1": number;
  temperature_c: number;
  qc_flags: {
    range: number;
    spike: number;
    stuck: number;
    biofouling: number;
  };
  event_phase: number;
  oil_warn?: boolean;
  oil_alarm?: boolean;
}

export interface Incident {
  incident_id: string;
  scenario: string;
  confidence: number;
  summary: string;
  ts_peak: string;
  path: string;
  status?: 'analyzing' | 'ready' | 'closed';
}
