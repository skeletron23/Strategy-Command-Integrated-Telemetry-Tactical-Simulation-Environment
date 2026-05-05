import React, { useRef, useEffect } from 'react';

// ── Smooth interpolation (exponential ease-out, frame-rate independent) ──
const LERP_SPEED = 14;

function lerp(current, target, dt) {
  const t = 1 - Math.exp(-LERP_SPEED * dt);
  return current + (target - current) * t;
}

export default function TrackMap({ trackPathRef, telemetryRef }) {
  // Static canvas = track outline (drawn once, persists)
  const trackCanvasRef = useRef(null);
  // Dynamic canvas = car dot (cleared & redrawn every frame)
  const carCanvasRef = useRef(null);
  const wrapperRef = useRef(null);
  const rafRef = useRef(null);
  const locationRef = useRef(null);

  useEffect(() => {
    const trackCanvas = trackCanvasRef.current;
    const carCanvas = carCanvasRef.current;
    const wrapper = wrapperRef.current;
    if (!trackCanvas || !carCanvas || !wrapper) return;

    const trackCtx = trackCanvas.getContext('2d');
    const carCtx = carCanvas.getContext('2d');

    // ── Sizing ──
    let cssW = 0;
    let cssH = 0;

    // ── Track path state — used to detect when the track needs a redraw ──
    let drawnTrackLen = 0;        // how many points were in the path the last time we drew

    // ── Cached mapping values (recomputed on resize) ──
    let maxDrawArea = 0;
    let offsetX = 0;
    let offsetY = 0;
    const PADDING = 30;

    function recomputeMapping() {
      const usableWidth = cssW - PADDING * 2;
      const usableHeight = cssH - PADDING * 2;
      maxDrawArea = Math.min(usableWidth, usableHeight);
      offsetX = (cssW - maxDrawArea) / 2;
      offsetY = (cssH - maxDrawArea) / 2;
    }

    function toScreen(pt) {
      return {
        x: offsetX + pt.x * maxDrawArea,
        y: offsetY + (1.0 - pt.y) * maxDrawArea,
      };
    }

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      cssW = wrapper.clientWidth;
      cssH = wrapper.clientHeight;
      if (cssW === 0 || cssH === 0) return;

      // Resize both canvases
      trackCanvas.width = cssW * dpr;
      trackCanvas.height = cssH * dpr;
      trackCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      carCanvas.width = cssW * dpr;
      carCanvas.height = cssH * dpr;
      carCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      recomputeMapping();

      // Force track redraw after resize
      drawnTrackLen = 0;
      drawTrack();
    }

    // ── Draw the track outline onto the STATIC canvas ──
    function drawTrack() {
      const trackPath = trackPathRef.current;
      if (!trackPath || trackPath.length === 0) return;
      if (cssW === 0 || cssH === 0) return;

      trackCtx.clearRect(0, 0, cssW, cssH);

      const currentLen = trackPath.length;

      // Down-sample for very large paths
      const step = currentLen > 2000 ? Math.floor(currentLen / 1000) : 1;
      const first = toScreen(trackPath[0]);

      trackCtx.beginPath();
      trackCtx.moveTo(first.x, first.y);
      for (let i = step; i < currentLen; i += step) {
        const p = toScreen(trackPath[i]);
        trackCtx.lineTo(p.x, p.y);
      }
      // Always include last point
      if (step > 1) {
        const last = toScreen(trackPath[currentLen - 1]);
        trackCtx.lineTo(last.x, last.y);
      }

      trackCtx.strokeStyle = '#333333';
      trackCtx.lineWidth = 6;
      trackCtx.lineCap = 'round';
      trackCtx.lineJoin = 'round';
      trackCtx.stroke();

      drawnTrackLen = currentLen;
    }

    resizeCanvas();
    const ro = new ResizeObserver(() => resizeCanvas());
    ro.observe(wrapper);

    // ── Smoothed car position ──
    const carDisplay = { x: 0, y: 0 };
    let carInitialized = false;
    let lastTime = performance.now();

    // ── Persistent rAF draw loop — ONLY redraws the car dot ──
    function draw(now) {
      if (cssW === 0 || cssH === 0) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      const trackPath = trackPathRef.current;
      const telemetry = telemetryRef.current;

      // If track data hasn't arrived yet, show placeholder on the car canvas
      if (!trackPath || trackPath.length === 0) {
        carCtx.clearRect(0, 0, cssW, cssH);
        carCtx.fillStyle = '#484f58';
        carCtx.font = '14px "JetBrains Mono", monospace';
        carCtx.textAlign = 'center';
        carCtx.textBaseline = 'middle';
        carCtx.fillText('AWAITING TRACK DATA…', cssW / 2, cssH / 2);
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // ── Update static track canvas only when needed ──
      // Redraw only when the path has grown (new points arrived).
      // Once the path hits its max capacity the length stabilizes and
      // this comparison short-circuits every frame — zero cost.
      if (trackPath.length !== drawnTrackLen) {
        drawTrack();
      }

      // ── Clear ONLY the car canvas and redraw the dot ──
      carCtx.clearRect(0, 0, cssW, cssH);

      if (telemetry) {
        const targetX = telemetry.x;
        const targetY = telemetry.y;

        if (!carInitialized) {
          carDisplay.x = targetX;
          carDisplay.y = targetY;
          carInitialized = true;
        } else {
          carDisplay.x = lerp(carDisplay.x, targetX, dt);
          carDisplay.y = lerp(carDisplay.y, targetY, dt);
        }

        const carPos = toScreen(carDisplay);

        // Glow
        carCtx.beginPath();
        carCtx.arc(carPos.x, carPos.y, 12, 0, Math.PI * 2);
        carCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
        carCtx.fill();

        // Car dot
        carCtx.beginPath();
        carCtx.arc(carPos.x, carPos.y, 6, 0, Math.PI * 2);
        carCtx.fillStyle = '#ff0000';
        carCtx.fill();
      }

      // Update location label via DOM
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
        {/* Static layer: track outline — drawn once, never cleared per-frame */}
        <canvas ref={trackCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
        {/* Dynamic layer: car dot — cleared & redrawn every frame */}
        <canvas ref={carCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
      </div>
      <span className="track-map-panel__location" ref={locationRef}>—</span>
    </div>
  );
}
