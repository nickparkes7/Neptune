import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Incident, AgentBrief } from './types';
import TelemetryCharts from './components/TelemetryCharts';
import ShipMap from './components/ShipMap';
import IncidentList from './components/IncidentList';
import IncidentDetail from './components/IncidentDetail';
import Sidebar from './components/Sidebar';
import AgentBriefDrawer from './components/AgentBriefDrawer';
import { useStreamingData } from './hooks/useStreamingData';

const API_BASE = 'http://localhost:8000';

function App() {
  const [incidentFeed, setIncidentFeed] = useState<Incident[]>([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [activeIncidentAlert, setActiveIncidentAlert] = useState<Incident | null>(null);
  const [agentBrief, setAgentBrief] = useState<AgentBrief | null>(null);
  const [isBriefOpen, setIsBriefOpen] = useState(false);
  const [briefLoading, setBriefLoading] = useState(false);
  const [briefError, setBriefError] = useState<string | null>(null);
  const [agentBriefIncidentId, setAgentBriefIncidentId] = useState<string | null>(null);
  const dismissedAlertsRef = useRef<Set<string>>(new Set());
  const [, forceAlertState] = useState(0);
  const seenIncidentIdsRef = useRef<Set<string>>(new Set());
  const hasLiveIncidentRef = useRef(false);

  const classifySeverity = (incident: Incident) => {
    const confidence = typeof incident.confidence === 'number' ? incident.confidence : 0;
    if (confidence >= 0.8) return 'critical';
    if (confidence >= 0.5) return 'warning';
    return 'info';
  };

  const sortIncidents = (incidents: Incident[]) =>
    incidents.sort((a, b) => {
      const aTime = a.ts_peak ? new Date(a.ts_peak).getTime() : 0;
      const bTime = b.ts_peak ? new Date(b.ts_peak).getTime() : 0;
      return bTime - aTime;
    });

  const handleIncomingIncidents = useCallback((incoming: Incident[]) => {
    if (!incoming || incoming.length === 0) {
      return;
    }

    if (!hasLiveIncidentRef.current) {
      // Ignore initial pre-populated incidents until a live incident has appeared this session
      return;
    }

    setIncidentFeed((prev) => {
      const map = new Map(prev.map((incident) => [incident.incident_id, incident]));

      incoming.forEach((incomingIncident) => {
        if (!seenIncidentIdsRef.current.has(incomingIncident.incident_id)) {
          return;
        }

        const existing = map.get(incomingIncident.incident_id);
        const merged: Incident = {
          ...existing,
          ...incomingIncident,
        };

        map.set(merged.incident_id, merged);
      });

      return sortIncidents(Array.from(map.values()));
    });

    // pop the latest incident into the alert if it just got populated with a real summary
    const sortedIncoming = sortIncidents([...incoming]);
    if (sortedIncoming.length > 0) {
      const latest = sortedIncoming[0];
      if (!dismissedAlertsRef.current.has(latest.incident_id)) {
        setActiveIncidentAlert(latest);
      }
    }
  }, []);

  const fetchIncidents = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/incidents`);
      if (response.ok) {
        const data = await response.json();
        handleIncomingIncidents(data);
      }
    } catch (error) {
      console.error('Failed to fetch incidents:', error);
    }
  }, [handleIncomingIncidents]);

  const handleIncidentTransition = useCallback((message: any) => {
    const transitions = Array.isArray(message?.transitions) ? message.transitions : [];
    let placeholderAlert: Incident | null = null;

    setIncidentFeed((prev) => {
      if (transitions.length === 0) {
        return prev;
      }

      const map = new Map(prev.map((incident) => [incident.incident_id, incident]));

      transitions.forEach((transition: any) => {
        const incidentId: string | undefined = transition?.incident_id;
        if (!incidentId) {
          return;
        }

        if (transition.kind === 'opened') {
          hasLiveIncidentRef.current = true;
          if (dismissedAlertsRef.current.has(incidentId)) {
            const next = new Set(dismissedAlertsRef.current);
            next.delete(incidentId);
            dismissedAlertsRef.current = next;
            forceAlertState((tick) => tick + 1);
          }
          if (!map.has(incidentId)) {
            const placeholder: Incident = {
              incident_id: incidentId,
              scenario: 'pending_analysis',
              confidence: 0,
              summary: 'Analyzing SeaOWL anomaly…',
              ts_peak: transition.ts_peak ?? new Date().toISOString(),
              path: '',
              status: 'analyzing',
            };
            map.set(incidentId, placeholder);
            placeholderAlert = placeholder;
            seenIncidentIdsRef.current.add(incidentId);
          } else {
            const existing = map.get(incidentId)!;
            map.set(incidentId, { ...existing, status: existing.status || 'analyzing' });
          }
        } else if (transition.kind === 'closed') {
          const existing = map.get(incidentId);
          if (existing) {
            map.set(incidentId, { ...existing, status: 'closed' });
          }
        } else if (transition.kind === 'updated') {
          const existing = map.get(incidentId);
          if (existing) {
            map.set(incidentId, { ...existing, status: 'ready' });
          }
        }
      });

      return sortIncidents(Array.from(map.values()));
    });

    if (placeholderAlert && !dismissedAlertsRef.current.has(placeholderAlert.incident_id)) {
      setActiveIncidentAlert(placeholderAlert);
    }

    fetchIncidents();
  }, [fetchIncidents]);

  // Use streaming data hook for real-time telemetry
  const {
    data: telemetryData,
    isConnected,
    connectionStatus,
    lastUpdateTime
  } = useStreamingData({
    websocketUrl: 'ws://localhost:8001',
    maxBufferSize: 1000,
    enableAutoReconnect: true,
    onIncidentTransition: handleIncidentTransition
  });

  const isLoading = !isConnected && telemetryData.length === 0;

  const handleIncidentSelect = (incidentId: string) => {
    setSelectedIncidentId(incidentId);
  };

  const handleBackToTelemetry = () => {
    setSelectedIncidentId(null);
  };

  const loadAgentBrief = useCallback(
    async (incidentId: string, options?: { refresh?: boolean }) => {
      setBriefLoading(true);
      setBriefError(null);
      try {
        const refreshSuffix = options?.refresh ? '?refresh=1' : '';
        const response = await fetch(`${API_BASE}/incidents/${incidentId}/agent-brief${refreshSuffix}`);
        if (!response.ok) {
          throw new Error(`Agent brief request failed with status ${response.status}`);
        }
        const data: AgentBrief = await response.json();
        setAgentBriefIncidentId(incidentId);
        setAgentBrief(data);
        setIsBriefOpen(true);
      } catch (err) {
        console.error('Failed to load agent brief:', err);
        setAgentBriefIncidentId(incidentId);
        setAgentBrief(null);
        setBriefError('Unable to load agent brief. Try regenerating.');
        setIsBriefOpen(true);
      } finally {
        setBriefLoading(false);
      }
    },
    []
  );

  // Periodically poll for new incidents
  useEffect(() => {
    fetchIncidents();
    const interval = setInterval(() => {
      fetchIncidents();
    }, 2000);

    return () => clearInterval(interval);
  }, [fetchIncidents]);

  // Auto-clear active incident alert after a short time
  useEffect(() => {
    if (!activeIncidentAlert) {
      return;
    }
    const timer = setTimeout(() => {
      setActiveIncidentAlert(null);
    }, 10000);
    return () => clearTimeout(timer);
  }, [activeIncidentAlert]);

  // Get latest telemetry point for alerts
  const latestData = telemetryData[telemetryData.length - 1];
  const hasOilAlarm = latestData?.oil_alarm || false;
  const hasOilWarning = latestData?.oil_warn || false;

  return (
    <div className="app">
      <header className="header">
        <h1>🌊 Neptune Console</h1>
      </header>

      <div className="app-layout">
        <div className="sidebar-left">
          <Sidebar
            apiBase={API_BASE}
            lastUpdateTime={lastUpdateTime}
            connectionStatus={connectionStatus}
            isStreaming={isConnected}
            agentBrief={agentBrief}
            latestIncidentId={incidentFeed[0]?.incident_id || null}
            onOpenAgentBrief={loadAgentBrief}
            isBriefLoading={briefLoading}
          />
        </div>

        <div className="main-container">
          {selectedIncidentId ? (
            <IncidentDetail
              incidentId={selectedIncidentId}
              onBack={handleBackToTelemetry}
              onExplain={loadAgentBrief}
            />
          ) : (
            <>
              {/* Alert Messages */}
              {activeIncidentAlert && (
                <div
                  className={`alert ${
                    classifySeverity(activeIncidentAlert) === 'critical'
                      ? 'alert-error'
                      : classifySeverity(activeIncidentAlert) === 'warning'
                      ? 'alert-warning'
                      : 'alert-info'
                  }`}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}
                >
                  <div>
                    <strong>New incident detected:</strong>{' '}
                    {activeIncidentAlert.summary || `Incident ${activeIncidentAlert.incident_id}`}
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        if (activeIncidentAlert) {
                          const next = new Set(dismissedAlertsRef.current);
                          next.add(activeIncidentAlert.incident_id);
                          dismissedAlertsRef.current = next;
                          forceAlertState((tick) => tick + 1);
                        }
                        setActiveIncidentAlert(null);
                      }}
                      aria-label="Dismiss incident alert"
                    >
                      ×
                    </button>
                    <button
                      className="btn btn-primary"
                      onClick={() => handleIncidentSelect(activeIncidentAlert.incident_id)}
                    >
                      View
                    </button>
                  </div>
                </div>
              )}

              {hasOilAlarm && (
                <div className="alert alert-error">
                  ALARM: Oil channel elevated
                </div>
              )}
              {hasOilWarning && !hasOilAlarm && (
                <div className="alert alert-warning">
                  WARNING: Oil channel elevated
                </div>
              )}

              {isLoading ? (
                <div className="loading">Loading telemetry data...</div>
              ) : (
                <>
                  <div className="telemetry-section">
                    <TelemetryCharts data={telemetryData} />
                  </div>
                  <div className="map-section">
                    <ShipMap
                      data={telemetryData}
                    />
                  </div>
                </>
              )}
            </>
          )}
        </div>

        <div className="sidebar-right">
          <IncidentList
            incidents={incidentFeed}
            onIncidentSelect={handleIncidentSelect}
            onExplain={loadAgentBrief}
          />
        </div>
      </div>

      <AgentBriefDrawer
        brief={agentBrief}
        apiBase={API_BASE}
        isOpen={isBriefOpen}
        isLoading={briefLoading}
        error={briefError}
        onClose={() => setIsBriefOpen(false)}
        onRetry={agentBriefIncidentId ? () => loadAgentBrief(agentBriefIncidentId, { refresh: true }) : undefined}
        incidentId={agentBriefIncidentId}
      />
    </div>
  );
}

export default App;
