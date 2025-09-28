import React from 'react';
import {
  AgentBrief,
  AgentBriefObservation,
  AgentBriefMedia,
  AgentBriefAction,
} from '../types';

interface Props {
  brief: AgentBrief | null;
  apiBase: string;
  isOpen: boolean;
  isLoading?: boolean;
  error?: string | null;
  onClose: () => void;
  onRetry?: () => void;
}

const riskColors: Record<AgentBrief['risk_label'], string> = {
  low: '#22c55e',
  medium: '#facc15',
  high: '#f97316',
  critical: '#f87171',
};

const AgentBriefDrawer: React.FC<Props> = ({
  brief,
  apiBase,
  isOpen,
  isLoading = false,
  error = null,
  onClose,
  onRetry,
}) => {
  if (!isOpen && !isLoading) {
    return null;
  }

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

  const renderMedia = (media: AgentBriefMedia, index: number) => {
    const src = buildUrl(media.path);
    if (!src) {
      return null;
    }
    return (
      <div className="agent-brief-media" key={`${media.label}-${index}`}>
        <img src={src} alt={media.label} />
        <span>{media.label}</span>
      </div>
    );
  };

  const renderObservation = (observation: AgentBriefObservation) => (
    <div className="agent-brief-card" key={observation.id}>
      <h4>{observation.title}</h4>
      <p>{observation.summary}</p>
      {observation.impact && (
        <p className="agent-brief-impact">Impact: {observation.impact}</p>
      )}
      {observation.evidence && observation.evidence.length > 0 && (
        <div className="agent-brief-media-grid">
          {observation.evidence.map(renderMedia)}
        </div>
      )}
    </div>
  );

  const renderAction = (action: AgentBriefAction) => (
    <li key={action.id} className={`agent-brief-action agent-brief-action-${action.urgency}`}>
      <strong>{action.title}</strong>
      <span className="agent-brief-action-urgency">{action.urgency.toUpperCase()}</span>
      <p>{action.summary}</p>
    </li>
  );

  const heroSrc = brief ? buildUrl(brief.hero_image) : undefined;
  const riskColor = brief ? riskColors[brief.risk_label] : '#94a3b8';

  return (
    <div className={`agent-brief-overlay ${isOpen ? 'open' : ''}`}>
      <div className="agent-brief-panel">
        <div className="agent-brief-header">
          <div className="agent-brief-title-group">
            <span className="agent-brief-badge" style={{ backgroundColor: riskColor }}>
              {brief ? `${brief.risk_label.toUpperCase()} • ${(brief.risk_score * 100).toFixed(0)}%` : 'AGENT BRIEF'}
            </span>
            <h2>{brief?.headline || 'Agent Brief'}</h2>
            <p className="agent-brief-subhead">
              {brief?.summary || 'Visual summary generated from cached SeaOWL evidence.'}
            </p>
          </div>
          <button className="agent-brief-close" onClick={onClose} aria-label="Close agent brief">
            ×
          </button>
        </div>

        {isLoading && (
          <div className="agent-brief-placeholder">Loading agent brief…</div>
        )}

        {!isLoading && error && (
          <div className="agent-brief-error">
            <p>{error}</p>
            {onRetry && (
              <button className="btn btn-secondary" onClick={onRetry}>
                Retry
              </button>
            )}
          </div>
        )}

        {!isLoading && brief && (
          <div className="agent-brief-content">
            {heroSrc && (
              <div className="agent-brief-hero">
                <img src={heroSrc} alt="Hero visual" />
                {brief.hero_caption && <span>{brief.hero_caption}</span>}
              </div>
            )}

            {brief.metrics && Object.keys(brief.metrics).length > 0 && (
              <div className="agent-brief-metrics">
                {Object.entries(brief.metrics).map(([key, value]) => (
                  <div key={key} className="agent-brief-metric">
                    <span className="agent-brief-metric-label">{key.replace(/_/g, ' ')}</span>
                    <span className="agent-brief-metric-value">{value}</span>
                  </div>
                ))}
              </div>
            )}

            {brief.observations && brief.observations.length > 0 && (
              <section>
                <h3>Observations</h3>
                <div className="agent-brief-grid">
                  {brief.observations.map(renderObservation)}
                </div>
              </section>
            )}

            {brief.recommended_actions && brief.recommended_actions.length > 0 && (
              <section>
                <h3>Recommended Actions</h3>
                <ul className="agent-brief-actions">
                  {brief.recommended_actions.map(renderAction)}
                </ul>
              </section>
            )}

            {brief.data_sources && (
              <section>
                <h3>Data Sources</h3>
                <ul className="agent-brief-data">
                  {Object.entries(brief.data_sources).map(([key, value]) => (
                    <li key={key}>
                      <strong>{key.replace(/_/g, ' ')}:</strong>{' '}
                      {Array.isArray(value) ? value.join(', ') : value}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <div className="agent-brief-footer">
              <button
                className="btn btn-secondary"
                onClick={() => window.open(`${apiBase}/agent-brief/markdown`, '_blank', 'noopener,noreferrer')}
              >
                Export Markdown
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => window.open(`${apiBase}/agent-brief`, '_blank', 'noopener,noreferrer')}
              >
                Download JSON
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentBriefDrawer;
