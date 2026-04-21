import React, { useRef, useEffect } from 'react';

export default function LapTimingTable({ lapTimes }) {
  const bodyRef = useRef(null);

  // Auto-scroll to latest lap
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [lapTimes]);

  if (!lapTimes || lapTimes.length === 0) {
    return (
      <div className="lap-table glass-panel animate-fade-in" style={{ animationDelay: '0.4s' }}>
        <div className="lap-table__title">🏁 Lap Timing</div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          color: 'var(--text-muted)',
          padding: '20px 0',
          textAlign: 'center'
        }}>
          Waiting for lap data…
        </div>
      </div>
    );
  }

  // Calculate deltas
  const rows = lapTimes.map((entry, idx) => {
    let delta = null;
    if (idx > 0) {
      delta = entry.time - lapTimes[idx - 1].time;
    }
    return { ...entry, delta };
  });

  // Find fastest lap for highlighting
  const fastestTime = Math.min(...lapTimes.map((l) => l.time));

  return (
    <div className="lap-table glass-panel animate-fade-in" style={{ animationDelay: '0.4s' }}>
      <div className="lap-table__title">🏁 Lap Timing</div>

      <div className="lap-table__header">
        <span>LAP</span>
        <span>TIME</span>
        <span style={{ textAlign: 'right' }}>DELTA</span>
      </div>

      <div className="lap-table__body" ref={bodyRef}>
        {rows.map((row) => {
          const isFastest = row.time === fastestTime;
          let deltaCls = 'delta-neutral';
          let deltaText = '—';

          if (row.delta !== null) {
            if (row.delta < -0.01) {
              deltaCls = 'delta-faster';
              deltaText = `−${Math.abs(row.delta).toFixed(3)}`;
            } else if (row.delta > 0.01) {
              deltaCls = 'delta-slower';
              deltaText = `+${row.delta.toFixed(3)}`;
            } else {
              deltaText = `±${Math.abs(row.delta).toFixed(3)}`;
            }
          }

          return (
            <div
              key={row.lap}
              className="lap-table__row"
              style={isFastest ? { background: 'rgba(179, 136, 255, 0.08)' } : undefined}
            >
              <span className="lap-table__lap-num">{row.lap}</span>
              <span className="lap-table__time" style={isFastest ? { color: '#b388ff' } : undefined}>
                {row.time.toFixed(3)}
              </span>
              <span className={`lap-table__delta ${deltaCls}`}>{deltaText}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
