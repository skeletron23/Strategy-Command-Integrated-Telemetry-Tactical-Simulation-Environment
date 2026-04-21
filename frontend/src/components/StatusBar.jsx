import React from 'react';

export default function StatusBar({ telemetryConnected, inferenceConnected, lap }) {
  return (
    <div className="status-bar">
      <div className="status-bar__brand">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="24" height="24" rx="4" stroke="#e10600" strokeWidth="2" fill="none" />
          <path d="M8 8h12v4H8zM8 16h8v4H8z" fill="#e10600" opacity="0.8" />
          <circle cx="21" cy="20" r="3" fill="#00d4ff" opacity="0.9" />
        </svg>
        <span className="status-bar__title">S.C.I.T.T.S.E. Command Center</span>
        <span className="status-bar__version">v3.0</span>
      </div>

      <div className="status-bar__indicators">
        <div className="status-indicator">
          <span className={`status-dot ${telemetryConnected ? 'status-dot--online' : 'status-dot--offline'}`} />
          <span>{telemetryConnected ? 'TELEMETRY LIVE' : 'TELEMETRY DEMO'}</span>
        </div>
        <div className="status-indicator">
          <span className={`status-dot ${inferenceConnected ? 'status-dot--online' : 'status-dot--offline'}`} />
          <span>{inferenceConnected ? 'ML ONLINE' : 'ML DEMO'}</span>
        </div>

        <div className="status-bar__lap">
          <span>LAP</span>
          <span className="status-bar__lap-num">{lap || '—'}</span>
        </div>
      </div>
    </div>
  );
}
