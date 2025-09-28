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

export interface SensorStatus {
  streaming: boolean;
  last_timestamp?: string | null;
}

export type SensorStatusMap = Record<string, SensorStatus>;

export interface Incident {
  incident_id: string;
  scenario: string;
  confidence: number;
  summary: string;
  ts_peak: string;
  path: string;
  status?: 'analyzing' | 'ready' | 'closed';
}

export interface AgentBriefMedia {
  label: string;
  path: string;
  asset_path?: string;
  thumbnail?: string;
  kind?: 'image' | 'plot' | 'map' | 'document';
}

export interface AgentBriefObservation {
  id: string;
  title: string;
  summary: string;
  impact?: string;
  evidence: AgentBriefMedia[];
}

export interface AgentBriefAction {
  id: string;
  title: string;
  summary: string;
  urgency: 'low' | 'medium' | 'high' | 'critical';
}

export interface AgentBriefCitation {
  claim_id: string;
  label: string;
  path: string;
}

export interface AgentBrief {
  scenario_id: string;
  generated_at: string;
  risk_score: number;
  risk_label: 'low' | 'medium' | 'high' | 'critical';
  headline: string;
  summary: string;
  hero_image?: string;
  hero_caption?: string;
  observations: AgentBriefObservation[];
  recommended_actions: AgentBriefAction[];
  citations: AgentBriefCitation[];
  metrics: Record<string, number | string>;
  data_sources: Record<string, string | string[]>;
}
