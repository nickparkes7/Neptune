import React from 'react';
import { Empty, Typography } from 'antd';
import { ExclamationCircleOutlined } from '@ant-design/icons';
import { Incident } from '../types';

const { Text } = Typography;

interface Props {
  incidents: Incident[];
  onIncidentSelect: (incidentId: string) => void;
}

const IncidentList: React.FC<Props> = ({ incidents, onIncidentSelect }) => {
  const getSeverity = (incident: Incident) => {
    const confidence = typeof incident.confidence === 'number' ? incident.confidence : 0;
    if (confidence >= 0.8) return 'critical';
    if (confidence >= 0.5) return 'warning';
    return 'info';
  };

  const formatConfidence = (value: number | undefined) => {
    if (typeof value !== 'number') {
      return 'N/A';
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatTimestamp = (value?: string) => {
    if (!value) {
      return 'Processing…';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString();
  };

  const scenarioTitle = (scenario?: string) => {
    if (!scenario) {
      return 'Suspected oil spill';
    }
    if (scenario === 'suspected_algal_bloom') {
      return 'Suspected algal bloom';
    }
    return 'Suspected oil spill';
  };

  const renderStatus = (status?: string) => {
    if (status === 'analyzing') {
      return <span className="incident-card-status analyzing">Analyzing…</span>;
    }
    if (status === 'closed') {
      return <span className="incident-card-status closed">Closed</span>;
    }
    return null;
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>, incidentId: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onIncidentSelect(incidentId);
    }
  };

  return (
    <div>
      <h2 style={{ marginBottom: '1rem', color: '#e2e8f0' }}>
        Incidents
      </h2>

      <div className="incidents-container">
        {incidents.length === 0 ? (
          <Empty
            image={<ExclamationCircleOutlined style={{ fontSize: 48, color: '#60a5fa' }} />}
            description={
              <Text style={{ color: '#94a3b8' }}>
                No incidents yet. When an alarm occurs, the agent will generate one.
              </Text>
            }
          />
        ) : (
          <div className="incident-feed">
            {incidents.map((incident) => {
              const severity = getSeverity(incident);
              return (
                <div
                  key={incident.incident_id}
                  className={`incident-card severity-${severity}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => onIncidentSelect(incident.incident_id)}
                  onKeyDown={(event) => handleKeyPress(event, incident.incident_id)}
                >
                  <div className="incident-card-header">
                    <span className="incident-card-title">
                      {incident.status === 'analyzing'
                        ? 'Analyzing onboard anomaly…'
                        : scenarioTitle(incident.scenario)}
                    </span>
                    <span className="incident-card-time">
                      {formatTimestamp(incident.ts_peak)}
                    </span>
                  </div>
                  <div className="incident-card-status-row">
                    {renderStatus(incident.status)}
                  </div>
                  <div className="incident-card-meta">
                    <span className="incident-card-confidence">
                      Confidence: {formatConfidence(incident.confidence)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default IncidentList;
