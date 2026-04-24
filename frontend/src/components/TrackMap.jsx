import React, { useRef, useEffect } from 'react';

export default function TrackMap({ trackPathRef, telemetryRef }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);
  const rafRef = useRef(null);
  const locationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext('2d');

    // ── Fix 1 & 2: ResizeObserver keeps canvas sized to container ──
    let cssW = 0;
    let cssH = 0;

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      cssW = wrapper.clientWidth;
      cssH = wrapper.clientHeight;
      if (cssW === 0 || cssH === 0) return;

      // Set the backing store to match physical pixels (sharp on HiDPI)
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;

      // Reset transform so all drawing uses CSS-pixel coordinates
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resizeCanvas();

    const ro = new ResizeObserver(() => resizeCanvas());
    ro.observe(wrapper);

    // ── Fix 3: Persistent rAF draw loop reads refs every frame ──
    function draw() {
      // Guard against zero-size canvas (e.g. hidden tab)
      if (cssW === 0 || cssH === 0) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      ctx.clearRect(0, 0, cssW, cssH);

      // Read latest data from refs — no React dependency
      const trackPath = trackPathRef.current;
      const telemetry = telemetryRef.current;

      if (!trackPath || trackPath.length === 0) {
        ctx.fillStyle = '#484f58';
        ctx.font = '14px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('AWAITING TRACK DATA…', cssW / 2, cssH / 2);
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // ── Aspect-ratio-safe mapping (identical to HTML version) ──
      const PADDING = 30;
      const usableWidth = cssW - PADDING * 2;
      const usableHeight = cssH - PADDING * 2;
      const maxDrawArea = Math.min(usableWidth, usableHeight);
      const offsetX = (cssW - maxDrawArea) / 2;
      const offsetY = (cssH - maxDrawArea) / 2;

      // Convert normalized 0–1 coordinates to screen pixels
      const toScreen = (pt) => ({
        x: offsetX + pt.x * maxDrawArea,
        y: offsetY + (1.0 - pt.y) * maxDrawArea,
      });

      // ── Draw the permanent static track (dark gray line) ──
      ctx.beginPath();
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const first = toScreen(trackPath[0]);
      ctx.moveTo(first.x, first.y);

      for (let i = 1; i < trackPath.length; i++) {
        const p = toScreen(trackPath[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();

      // ── Draw car position ──
      if (telemetry) {
        const carPos = toScreen({ x: telemetry.x, y: telemetry.y });

        // Draw glow around the car
        ctx.beginPath();
        ctx.arc(carPos.x, carPos.y, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
        ctx.fill();

        // Draw the car dot (bright red)
        ctx.beginPath();
        ctx.arc(carPos.x, carPos.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ff0000';
        ctx.fill();
      }

      // Update location label via DOM (cheaper than re-render)
      if (locationRef.current) {
        const turn = telemetry?.turn;
        const text = turn == null ? '—' : turn === 'Straight' ? 'Straight' : `Turn ${turn}`;
        locationRef.current.textContent = text;
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, []); // Empty deps — refs are stable, loop runs for component lifetime

  return (
    <div className="track-map-panel">
      <span className="track-map-panel__label">Circuit Map — Live</span>
      <div className="track-map-canvas-wrapper" ref={wrapperRef}>
        <canvas ref={canvasRef} />
      </div>
      <span className="track-map-panel__location" ref={locationRef}>—</span>
    </div>
  );
}
