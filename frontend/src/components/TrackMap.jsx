import React, { useRef, useEffect, useCallback } from 'react';

export default function TrackMap({ trackPath, telemetry }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = wrapper.clientWidth;
    const h = wrapper.clientHeight;

    if (w === 0 || h === 0) return;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, w, h);

    if (!trackPath || trackPath.length === 0) {
      ctx.fillStyle = '#484f58';
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('AWAITING TRACK DATA…', w / 2, h / 2);
      return;
    }

    // ── Aspect-ratio-safe mapping ──
    const PADDING = 50;
    const usableW = w - PADDING * 2;
    const usableH = h - PADDING * 2;
    const maxDraw = Math.min(usableW, usableH);
    const offsetX = (w - maxDraw) / 2;
    const offsetY = (h - maxDraw) / 2;

    const toScreen = (pt) => ({
      x: offsetX + pt.x * maxDraw,
      y: offsetY + (1.0 - pt.y) * maxDraw,
    });

    // ── Draw track outline (wide, dark base) ──
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const first = toScreen(trackPath[0]);
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < trackPath.length; i++) {
      const p = toScreen(trackPath[i]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();

    // ── Bright racing-line overlay for recent ~300 points ──
    const recentStart = Math.max(0, trackPath.length - 300);
    for (let i = recentStart + 1; i < trackPath.length; i++) {
      const prev = toScreen(trackPath[i - 1]);
      const curr = toScreen(trackPath[i]);
      const progress = (i - recentStart) / (trackPath.length - recentStart);

      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(curr.x, curr.y);
      ctx.strokeStyle = getSpeedColor(progress);
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // ── Car position ──
    if (telemetry) {
      const carPos = toScreen({ x: telemetry.x, y: telemetry.y });

      // Outer glow
      const glowGrad = ctx.createRadialGradient(carPos.x, carPos.y, 0, carPos.x, carPos.y, 28);
      glowGrad.addColorStop(0, 'rgba(225, 6, 0, 0.6)');
      glowGrad.addColorStop(0.4, 'rgba(225, 6, 0, 0.15)');
      glowGrad.addColorStop(1, 'rgba(225, 6, 0, 0)');
      ctx.beginPath();
      ctx.arc(carPos.x, carPos.y, 28, 0, Math.PI * 2);
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Car dot
      ctx.beginPath();
      ctx.arc(carPos.x, carPos.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#e10600';
      ctx.fill();

      // Inner white core
      ctx.beginPath();
      ctx.arc(carPos.x, carPos.y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
  }, [trackPath, telemetry]);

  useEffect(() => {
    const id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [draw]);

  const turn = telemetry?.turn;
  const location = turn == null ? '—' : turn === 'Straight' ? 'Straight' : `Turn ${turn}`;

  return (
    <div className="track-map-panel">
      <span className="track-map-panel__label">Circuit Map — Live</span>
      <div className="track-map-canvas-wrapper" ref={wrapperRef}>
        <canvas ref={canvasRef} />
      </div>
      <span className="track-map-panel__location">{location}</span>
    </div>
  );
}

function getSpeedColor(progress) {
  if (progress < 0.3) {
    return `rgba(0, 100, 180, ${0.15 + progress * 0.5})`;
  } else if (progress < 0.7) {
    const t = (progress - 0.3) / 0.4;
    const r = Math.round(t * 225);
    const g = Math.round(212 - t * 200);
    const b = Math.round(255 - t * 255);
    return `rgba(${r}, ${g}, ${b}, ${0.5 + progress * 0.3})`;
  } else {
    const t = (progress - 0.7) / 0.3;
    return `rgba(225, ${Math.round(6 + (1 - t) * 80)}, ${Math.round(t * 20)}, ${0.7 + t * 0.3})`;
  }
}
