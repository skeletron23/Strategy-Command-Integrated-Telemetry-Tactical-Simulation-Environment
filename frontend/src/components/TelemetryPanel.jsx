import React, { useMemo } from 'react';

function SpeedCard({ speed }) {
  const maxSpeed = 370;
  const pct = Math.min(1, (speed || 0) / maxSpeed);
  const circumference = 2 * Math.PI * 28;
  const arcLength = circumference * 0.75; // 270° arc
  const offset = arcLength * (1 - pct);

  // Color: green → cyan → amber → red as speed increases
  let color = '#00e676';
  if (pct > 0.75) color = '#e10600';
  else if (pct > 0.55) color = '#ff8c00';
  else if (pct > 0.3) color = '#00d4ff';

  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.05s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">Speed</span>
        <span className="telem-card__icon">🏎️</span>
      </div>
      <div className="speed-ring-container">
        <div className="speed-ring">
          <svg width="72" height="72" viewBox="0 0 64 64">
            <circle className="speed-ring__bg" cx="32" cy="32" r="28" strokeDasharray={`${arcLength} ${circumference}`} />
            <circle
              className="speed-ring__fill"
              cx="32" cy="32" r="28"
              stroke={color}
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeDashoffset={offset}
              style={{ filter: `drop-shadow(0 0 4px ${color})` }}
            />
          </svg>
          <span className="speed-ring__value">{speed || 0}</span>
        </div>
        <div>
          <span className="telem-card__unit" style={{ fontSize: '12px' }}>km/h</span>
        </div>
      </div>
    </div>
  );
}

function RPMCard({ rpm }) {
  const maxRPM = 15000;
  const pct = Math.min(100, ((rpm || 0) / maxRPM) * 100);
  let barColor = 'linear-gradient(90deg, #00d4ff, #00e676)';
  if (pct > 85) barColor = 'linear-gradient(90deg, #ff8c00, #e10600)';
  else if (pct > 70) barColor = 'linear-gradient(90deg, #00d4ff, #ff8c00)';

  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.1s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">RPM</span>
        <span className="telem-card__icon">⚡</span>
      </div>
      <div className="telem-card__value">
        {(rpm || 0).toLocaleString()}
      </div>
      <div className="telem-card__bar-track">
        <div
          className="telem-card__bar-fill"
          style={{ width: `${pct}%`, background: barColor }}
        />
      </div>
    </div>
  );
}

function GearCard({ gear }) {
  const g = gear || 0;
  const maxGear = 8;

  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.15s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">Gear</span>
        <span className="telem-card__icon">⚙️</span>
      </div>
      <div className="gear-display">
        <span className="gear-number">{g === 0 ? 'N' : g}</span>
        <div className="gear-dots">
          {Array.from({ length: maxGear }, (_, i) => (
            <span
              key={i}
              className={`gear-dot ${i < g ? 'gear-dot--active' : ''}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ThrottleCard({ throttle }) {
  const val = throttle || 0;
  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.2s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">Throttle</span>
        <span className="telem-card__icon">🟢</span>
      </div>
      <div className="telem-card__value">
        {val}<span className="telem-card__unit">%</span>
      </div>
      <div className="telem-card__bar-track">
        <div
          className="telem-card__bar-fill"
          style={{
            width: `${val}%`,
            background: 'linear-gradient(90deg, #00e676, #00d4ff)',
          }}
        />
      </div>
    </div>
  );
}

function BrakeCard({ brake }) {
  const val = brake || 0;
  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.25s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">Brake</span>
        <span className="telem-card__icon">🔴</span>
      </div>
      <div className="telem-card__value" style={{ color: val > 0 ? '#e10600' : undefined }}>
        {val}<span className="telem-card__unit">%</span>
      </div>
      <div className="telem-card__bar-track">
        <div
          className="telem-card__bar-fill"
          style={{
            width: `${val}%`,
            background: 'linear-gradient(90deg, #ff8c00, #e10600)',
          }}
        />
      </div>
    </div>
  );
}

function GForceCard({ gForce }) {
  const val = gForce || 0;
  let cls = 'gforce-neutral';
  if (val > 0.3) cls = 'gforce-positive';
  else if (val < -0.3) cls = 'gforce-negative';

  return (
    <div className="telem-card glass-panel" style={{ animationDelay: '0.3s' }}>
      <div className="telem-card__header">
        <span className="telem-card__label">G-Force</span>
        <span className="telem-card__icon">💫</span>
      </div>
      <div className={`telem-card__value gforce-value ${cls}`}>
        {val > 0 ? '+' : ''}{val.toFixed(2)}<span className="telem-card__unit">G</span>
      </div>
    </div>
  );
}

export default function TelemetryPanel({ telemetry }) {
  const { speed, rpm, gear, throttle, brake, g_force } = telemetry || {};

  return (
    <div className="telemetry-panel">
      <SpeedCard speed={speed} />
      <RPMCard rpm={rpm} />
      <GearCard gear={gear} />
      <ThrottleCard throttle={throttle} />
      <BrakeCard brake={brake} />
      <GForceCard gForce={g_force} />
    </div>
  );
}
