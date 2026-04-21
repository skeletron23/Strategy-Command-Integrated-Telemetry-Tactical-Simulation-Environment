import React, { useRef, useEffect, useCallback } from 'react';

export default function StintProjection({ prediction }) {
  const canvasRef = useRef(null);

  const projection = prediction?.projection || [];
  const summary = prediction?.summary || {};

  const nextLap = projection.length > 0 ? projection[0].predicted_lap_time_s : null;
  const stintDelta = summary.stint_time_delta_s ?? null;

  // Draw the mini stint curve chart
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || projection.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const w = parent.clientWidth;
    const h = 120;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    const times = projection.map((p) => p.predicted_lap_time_s);
    const minT = Math.min(...times) - 0.1;
    const maxT = Math.max(...times) + 0.1;
    const range = maxT - minT || 1;

    const padX = 8;
    const padY = 16;
    const chartW = w - padX * 2;
    const chartH = h - padY * 2;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padY + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padX, y);
      ctx.lineTo(w - padX, y);
      ctx.stroke();
    }

    // Gradient fill under curve
    const gradient = ctx.createLinearGradient(0, padY, 0, h);
    gradient.addColorStop(0, 'rgba(225, 6, 0, 0.2)');
    gradient.addColorStop(1, 'rgba(225, 6, 0, 0)');

    const points = times.map((t, i) => ({
      x: padX + (i / (times.length - 1 || 1)) * chartW,
      y: padY + (1 - (t - minT) / range) * chartH,
    }));

    // Fill
    ctx.beginPath();
    ctx.moveTo(points[0].x, h);
    points.forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#e10600';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();

    // Glow line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(225, 6, 0, 0.3)';
    ctx.lineWidth = 6;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();

    // Data points
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#e10600';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
    });

    // Labels — first and last time
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${times[0].toFixed(2)}s`, points[0].x, points[0].y - 8);
    ctx.textAlign = 'right';
    ctx.fillText(`${times[times.length - 1].toFixed(2)}s`, points[points.length - 1].x, points[points.length - 1].y - 8);
  }, [projection]);

  useEffect(() => {
    drawChart();
    window.addEventListener('resize', drawChart);
    return () => window.removeEventListener('resize', drawChart);
  }, [drawChart]);

  return (
    <div className="stint-card glass-panel-active animate-fade-in">
      <div className="stint-card__title">⏱ Tire Degradation Projection</div>

      <div className="stint-card__prediction">
        <div className="stint-stat">
          <div className="stint-stat__label">Next Lap</div>
          <div className="stint-stat__value">
            {nextLap !== null ? nextLap.toFixed(3) : '—'}
            <span className="stint-stat__unit">s</span>
          </div>
        </div>
        <div className="stint-stat">
          <div className="stint-stat__label">Stint ∆</div>
          <div className="stint-stat__value" style={{ color: stintDelta > 0.5 ? '#e10600' : stintDelta > 0.2 ? '#ff8c00' : '#00e676' }}>
            {stintDelta !== null ? `+${stintDelta.toFixed(3)}` : '—'}
            <span className="stint-stat__unit">s</span>
          </div>
        </div>
      </div>

      <div className="stint-chart">
        <canvas ref={canvasRef} />
      </div>

      <div style={{ marginTop: '10px', display: 'flex', gap: '6px' }}>
        <span className="compound-badge compound-medium">MEDIUM</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', alignSelf: 'center', marginLeft: '4px' }}>
          {projection.length} lap projection
        </span>
      </div>
    </div>
  );
}
