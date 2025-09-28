import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

interface Props {
  incidentId: string;
  onBack: () => void;
  onExplain?: (incidentId: string, refresh?: boolean) => void | Promise<void>;
}

interface IncidentDetailData {
  incident_id: string;
  scenario?: string;
  confidence?: number;
  summary?: string;
  rationale?: string;
  recommended_actions?: string[];
  followup_scheduled?: boolean;
  followup_eta?: string;
  artifacts?: Record<string, string>;
  event?: {
    ts_peak?: string;
    lat?: number;
    lon?: number;
  };
  metrics?: Record<string, unknown>;
  brief_available?: boolean;
  trace_available?: boolean;
  status?: 'analyzing' | 'ready' | 'closed';
  agent_brief_available?: boolean;
  agent_brief?: {
    headline?: string;
    risk_label?: 'low' | 'medium' | 'high' | 'critical';
    risk_score?: number;
    generated_at?: string;
    hero_image?: string;
    summary?: string;
  };
}

const IncidentDetail: React.FC<Props> = ({ incidentId, onBack, onExplain }) => {
  const [incident, setIncident] = useState<IncidentDetailData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchIncidentDetail = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const response = await fetch(`${API_BASE}/incidents/${incidentId}`);
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        const data: IncidentDetailData = await response.json();
        if (!cancelled) {
          setIncident(data);
          setIsLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Failed to load incident details', err);
          setError('Failed to load incident details');
          setIsLoading(false);
        }
      }
    };

    fetchIncidentDetail();

    return () => {
      cancelled = true;
    };
  }, [incidentId]);

  const openArtifact = (path: string) => {
    window.open(path, '_blank', 'noopener,noreferrer');
  };

  const formatCoordinate = (value?: number) => (typeof value === 'number' ? value.toFixed(4) : '—');
  const formatStatus = (value?: string) => {
    if (value === 'analyzing') return 'Analyzing…';
    if (value === 'ready') return 'Ready';
    if (value === 'closed') return 'Closed';
    return 'Unknown';
  };

  if (isLoading) {
    return (
      <div className="incidents-container">
        <button className="btn btn-primary" onClick={onBack} style={{ marginBottom: '1rem' }}>
          ← Back to Live Telemetry
        </button>
        <div className="loading">Loading incident details...</div>
      </div>
    );
  }

  if (error || !incident) {
    return (
      <div className="incidents-container">
        <button className="btn btn-primary" onClick={onBack} style={{ marginBottom: '1rem' }}>
          ← Back to Live Telemetry
        </button>
        <div className="alert alert-error">
          {error || `Incident ${incidentId} not found`}
        </div>
      </div>
    );
  }

  return (
    <div className="incidents-container">
      <button className="btn btn-primary" onClick={onBack} style={{ marginBottom: '1rem' }}>
        ← Back to Live Telemetry
      </button>

      <h2>Incident Details</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '1.5rem' }}>
        <div>
          <h4>Scenario</h4>
          <p>{incident.scenario?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Unknown'}</p>
        </div>
        <div>
          <h4>Confidence</h4>
          <p>{incident.confidence ? `${(incident.confidence * 100).toFixed(1)}%` : 'N/A'}</p>
        </div>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <h4>Status</h4>
        <p>{formatStatus(incident.status)}</p>
        {incident.status === 'analyzing' && (
          <div className="alert alert-info" style={{ marginTop: '0.75rem' }}>
            Incident processing is in progress. Agent findings will appear once analysis completes.
          </div>
        )}
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <h4>Summary</h4>
        <p>{incident.summary}</p>
      </div>

      {incident.rationale && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Rationale</h4>
          <p>{incident.rationale}</p>
        </div>
      )}

      {incident.recommended_actions && incident.recommended_actions.length > 0 && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Recommended Actions</h4>
          <ul>
            {incident.recommended_actions.map((action, index) => (
              <li key={index} style={{ marginBottom: '0.5rem' }}>
                {action}
              </li>
            ))}
          </ul>
        </div>
      )}

      {incident.agent_brief && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Agent Brief Snapshot</h4>
          <p>
            {incident.agent_brief.summary || 'Agent brief available for this incident.'}
          </p>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginTop: '0.5rem' }}>
            {incident.agent_brief.risk_label && (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '0.25rem 0.6rem',
                  borderRadius: '999px',
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  fontSize: '0.7rem',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                }}
              >
                {incident.agent_brief.risk_label}
              </span>
            )}
            {typeof incident.agent_brief.risk_score === 'number' && (
              <span>Confidence {(incident.agent_brief.risk_score * 100).toFixed(0)}%</span>
            )}
            {incident.agent_brief.generated_at && (
              <span>
                Generated {new Date(incident.agent_brief.generated_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>
      )}

      {incident.followup_scheduled && (
        <div className="alert alert-info">
          Follow-up scheduled: {incident.followup_eta
            ? new Date(incident.followup_eta).toLocaleString()
            : 'Next Cerulean update'
          }
        </div>
      )}

      {incident.event?.ts_peak && (
        <div style={{ marginBottom: '1.5rem' }}>
          <h4>Event Peak</h4>
          <p>
            {new Date(incident.event.ts_peak).toLocaleString()} at{' '}
            {formatCoordinate(incident.event.lat)}, {formatCoordinate(incident.event.lon)}
          </p>
        </div>
      )}

      <div style={{ marginTop: '2rem' }}>
        <h4>Actions</h4>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button
            className="btn btn-primary"
            onClick={() => onExplain && onExplain(incident.incident_id)}
            disabled={!incident.agent_brief_available || !onExplain}
          >
            Explain anomaly
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => openArtifact(`${API_BASE}/incidents/${incidentId}/brief`)}
            disabled={!incident.brief_available}
          >
            Download JSON Brief
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => openArtifact(`${API_BASE}/incidents/${incidentId}/trace`)}
            disabled={!incident.trace_available}
          >
            View Agent Trace
          </button>
        </div>
      </div>
    </div>
  );
};

export default IncidentDetail;
