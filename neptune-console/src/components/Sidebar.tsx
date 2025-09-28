import React from 'react';
import { Card, Typography, Space } from 'antd';
import { AgentBrief, SensorStatusMap } from '../types';

const { Text } = Typography;

interface Props {
  lastUpdateTime: string;
  connectionStatus?: 'connecting' | 'connected' | 'disconnected' | 'error';
  isStreaming?: boolean;
  apiBase: string;
  agentBrief?: AgentBrief | null;
  onOpenAgentBrief?: () => void;
  isBriefLoading?: boolean;
  sensorStatus?: SensorStatusMap;
}

const Sidebar: React.FC<Props> = ({
  lastUpdateTime,
  connectionStatus = 'disconnected',
  isStreaming = false,
  apiBase,
  agentBrief = null,
  onOpenAgentBrief,
  isBriefLoading = false,
  sensorStatus,
}) => {
  const statuses = sensorStatus ?? {};

  const formatTimestamp = (value?: string | null) => {
    if (!value) {
      return null;
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return value;
    }
    return parsed.toLocaleTimeString();
  };

  const sensors: Array<{
    id: string;
    name: string;
    subtitle: string;
    status?: SensorStatusMap[string];
    fallbackStreaming?: boolean;
  }> = [
    {
      id: 'seaowl',
      name: 'SeaOWL',
      subtitle: 'Oil fluorescence',
      status: statuses.seaowl,
      fallbackStreaming: connectionStatus === 'connected' && isStreaming,
    },
    {
      id: 'eco_fl',
      name: 'ECO FL',
      subtitle: 'Chlorophyll & FDOM',
      status: statuses.eco_fl,
    },
    {
      id: 'eco_bb',
      name: 'ECO BB',
      subtitle: 'Backscatter',
      status: statuses['eco_bb'],
    },
    {
      id: 'acs',
      name: 'ac-s',
      subtitle: 'Absorption & attenuation',
      status: statuses.acs,
    },
  ];

  const buildUrl = (path?: string) => {
    if (!path) {
      return undefined;
    }
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }
    const trimmed = path.startsWith('/') ? path : `/${path}`;
    return `${apiBase}${trimmed}`;
  };

  const heroUrl = buildUrl(agentBrief?.hero_image);

  return (
    <div>
      <h2 style={{ marginBottom: '1rem', color: '#e2e8f0' }}>
        Onboard Sensors
      </h2>

      <Card
        className="sidebar"
        style={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center' }}
            onClick={onOpenAgentBrief}
            disabled={!onOpenAgentBrief || isBriefLoading}
          >
            {isBriefLoading ? 'Generating brief…' : 'Explain anomaly'}
          </button>

          {heroUrl && (
            <div className="agent-brief-preview">
              <img
                src={heroUrl}
                alt="Agent brief preview"
                onError={(event) => {
                  (event.target as HTMLImageElement).style.visibility = 'hidden';
                }}
              />
              <div>
                <Text style={{ color: '#e2e8f0', fontSize: '0.85rem', fontWeight: 600 }}>
                  {agentBrief.headline}
                </Text>
                <Text style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
                  {(agentBrief.generated_at && new Date(agentBrief.generated_at).toLocaleString()) || ''}
                </Text>
              </div>
            </div>
          )}

          {sensors.map((sensor, idx) => {
            const streaming = sensor.status?.streaming ?? sensor.fallbackStreaming ?? false;
            const indicatorColor = streaming ? '#22c55e' : '#ef4444';
            const lastSeen = sensor.status?.last_timestamp;
            const formatted = lastSeen ? formatTimestamp(lastSeen) : sensor.id === 'seaowl' ? lastUpdateTime : null;
            return (
              <React.Fragment key={sensor.id}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      backgroundColor: indicatorColor,
                      boxShadow: sensor.streaming ? '0 0 6px rgba(34, 197, 94, 0.6)' : '0 0 6px rgba(239, 68, 68, 0.4)'
                    }}
                  />
                  <div>
                    <Text style={{ color: '#e2e8f0', fontSize: '0.875rem', fontWeight: 500 }}>
                      {sensor.name}
                    </Text>
                    <div>
                      <Text style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
                        {sensor.subtitle}
                      </Text>
                      {formatted && (
                        <div>
                          <Text style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
                            Last update: {formatted}
                          </Text>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </React.Fragment>
            );
          })}
        </Space>
      </Card>

      <h2 style={{ margin: '1.5rem 0 1rem', color: '#e2e8f0' }}>
        External Sources
      </h2>

      <Card
        className="sidebar"
        style={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span role="img" aria-label="Satellite" style={{ fontSize: '1.1rem' }}>
              🛰️
            </span>
            <div>
              <Text style={{ color: '#e2e8f0', fontSize: '0.875rem', fontWeight: 500 }}>
                Sentinel-1
              </Text>
              <div>
                <Text style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
                  Synthetic aperture radar
                </Text>
              </div>
            </div>
          </div>
        </Space>
      </Card>
    </div>
  );
};

export default Sidebar;
