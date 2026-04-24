import React, { useRef, useEffect } from 'react';

// ── Color constants matching the CSS design system ──
const COLORS = {
  bgCard: 'rgba(17, 24, 32, 0.7)',
  borderSubtle: 'rgba(255, 255, 255, 0.06)',
  borderActive: 'rgba(0, 212, 255, 0.2)',
  textPrimary: '#f0f2f5',
  textSecondary: '#8b949e',
  textMuted: '#484f58',
  f1Red: '#e10600',
  cyan: '#00d4ff',
  green: '#00e676',
  amber: '#ff8c00',
};

const FONT_MONO = '"JetBrains Mono", "Fira Code", monospace';

export default function TelemetryPanel({ telemetryRef }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    function draw() {
      const canvas = canvasRef.current;
      const wrapper = wrapperRef.current;
      if (!canvas || !wrapper) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const dpr = window.devicePixelRatio || 1;
      const w = wrapper.clientWidth;
      const h = wrapper.clientHeight;

      if (w === 0 || h === 0) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const data = telemetryRef.current;
      const speed = data?.speed || 0;
      const rpm = data?.rpm || 0;
      const gear = data?.gear || 0;
      const throttle = data?.throttle || 0;
      const brake = data?.brake || 0;
      const gForce = data?.g_force || 0;

      // ── Layout: vertical stack of cards ──
      const cardPadX = 12;
      const cardGap = 8;
      const cardW = w - cardPadX * 2;
      const cardPadInner = 14;
      const headerH = 22;
      const cardRadius = 16;

      let y = 12; // top padding matching .telemetry-panel padding

      // ===== SPEED CARD =====
      const speedCardH = 110;
      drawCardBg(ctx, cardPadX, y, cardW, speedCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'SPEED', '🏎️');

      // Speed ring
      const ringCx = cardPadX + cardPadInner + 36;
      const ringCy = y + headerH + cardPadInner + 32;
      const ringR = 28;
      const maxSpeed = 370;
      const speedPct = Math.min(1, speed / maxSpeed);
      const circumference = 2 * Math.PI * ringR;
      const arcLength = circumference * 0.75; // 270°

      let speedColor = COLORS.green;
      if (speedPct > 0.75) speedColor = COLORS.f1Red;
      else if (speedPct > 0.55) speedColor = COLORS.amber;
      else if (speedPct > 0.3) speedColor = COLORS.cyan;

      // Ring background (270° arc)
      ctx.beginPath();
      const startAngle = (135 * Math.PI) / 180;
      const endAngle = startAngle + (270 * Math.PI) / 180;
      ctx.arc(ringCx, ringCy, ringR, startAngle, endAngle);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Ring fill
      const fillEnd = startAngle + (270 * Math.PI / 180) * speedPct;
      ctx.beginPath();
      ctx.arc(ringCx, ringCy, ringR, startAngle, fillEnd);
      ctx.strokeStyle = speedColor;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Speed value in center of ring
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = `700 18px ${FONT_MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(speed), ringCx, ringCy);

      // km/h label
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = `12px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('km/h', ringCx + ringR + 16, ringCy);

      y += speedCardH + cardGap;

      // ===== RPM CARD =====
      const rpmCardH = 86;
      drawCardBg(ctx, cardPadX, y, cardW, rpmCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'RPM', '⚡');

      // RPM value
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = `700 32px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(rpm.toLocaleString(), cardPadX + cardPadInner, y + headerH + cardPadInner + 2);

      // RPM bar
      const maxRPM = 15000;
      const rpmPct = Math.min(1, rpm / maxRPM);
      const barY = y + rpmCardH - cardPadInner - 4;
      const barW = cardW - cardPadInner * 2;
      drawBar(ctx, cardPadX + cardPadInner, barY, barW, 4, rpmPct,
        rpmPct > 0.85 ? ['#ff8c00', '#e10600'] :
        rpmPct > 0.70 ? ['#00d4ff', '#ff8c00'] :
        ['#00d4ff', '#00e676']
      );

      y += rpmCardH + cardGap;

      // ===== GEAR CARD =====
      const gearCardH = 86;
      drawCardBg(ctx, cardPadX, y, cardW, gearCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'GEAR', '⚙️');

      // Gear number with gradient text effect
      const gearStr = gear === 0 ? 'N' : String(gear);
      ctx.fillStyle = COLORS.cyan;
      ctx.font = `800 48px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(gearStr, cardPadX + cardPadInner, y + headerH + cardPadInner - 6);

      // Gear dots
      const maxGear = 8;
      const dotX = cardPadX + cardPadInner + 50;
      const dotStartY = y + headerH + cardPadInner + 2;
      for (let i = 0; i < maxGear; i++) {
        const dy = dotStartY + i * 9;
        ctx.beginPath();
        ctx.arc(dotX, dy, 3, 0, Math.PI * 2);
        if (i < gear) {
          ctx.fillStyle = COLORS.cyan;
          ctx.shadowColor = 'rgba(0, 212, 255, 0.3)';
          ctx.shadowBlur = 6;
        } else {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
          ctx.shadowColor = 'transparent';
          ctx.shadowBlur = 0;
        }
        ctx.fill();
      }
      ctx.shadowBlur = 0;

      y += gearCardH + cardGap;

      // ===== THROTTLE CARD =====
      const throttleCardH = 86;
      drawCardBg(ctx, cardPadX, y, cardW, throttleCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'THROTTLE', '🟢');

      // Throttle value
      ctx.fillStyle = COLORS.textPrimary;
      ctx.font = `700 32px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(String(throttle), cardPadX + cardPadInner, y + headerH + cardPadInner + 2);
      const throttleNumW = ctx.measureText(String(throttle)).width;
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = `14px ${FONT_MONO}`;
      ctx.fillText('%', cardPadX + cardPadInner + throttleNumW + 4, y + headerH + cardPadInner + 12);

      // Throttle bar
      const tBarY = y + throttleCardH - cardPadInner - 4;
      drawBar(ctx, cardPadX + cardPadInner, tBarY, barW, 4, throttle / 100, ['#00e676', '#00d4ff']);

      y += throttleCardH + cardGap;

      // ===== BRAKE CARD =====
      const brakeCardH = 86;
      drawCardBg(ctx, cardPadX, y, cardW, brakeCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'BRAKE', '🔴');

      // Brake value
      ctx.fillStyle = brake > 0 ? COLORS.f1Red : COLORS.textPrimary;
      ctx.font = `700 32px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(String(brake), cardPadX + cardPadInner, y + headerH + cardPadInner + 2);
      const brakeNumW = ctx.measureText(String(brake)).width;
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = `14px ${FONT_MONO}`;
      ctx.fillText('%', cardPadX + cardPadInner + brakeNumW + 4, y + headerH + cardPadInner + 12);

      // Brake bar
      const bBarY = y + brakeCardH - cardPadInner - 4;
      drawBar(ctx, cardPadX + cardPadInner, bBarY, barW, 4, brake / 100, ['#ff8c00', '#e10600']);

      y += brakeCardH + cardGap;

      // ===== G-FORCE CARD =====
      const gForceCardH = 76;
      drawCardBg(ctx, cardPadX, y, cardW, gForceCardH, cardRadius);
      drawCardHeader(ctx, cardPadX + cardPadInner, y + cardPadInner, cardW - cardPadInner * 2, 'G-FORCE', '💫');

      // G-Force value
      let gColor = COLORS.textPrimary;
      if (gForce > 0.3) gColor = COLORS.green;
      else if (gForce < -0.3) gColor = COLORS.f1Red;

      ctx.fillStyle = gColor;
      ctx.font = `700 32px ${FONT_MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      const gStr = (gForce > 0 ? '+' : '') + gForce.toFixed(2);
      ctx.fillText(gStr, cardPadX + cardPadInner, y + headerH + cardPadInner + 2);
      const gNumW = ctx.measureText(gStr).width;
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = `14px ${FONT_MONO}`;
      ctx.fillText('G', cardPadX + cardPadInner + gNumW + 4, y + headerH + cardPadInner + 12);

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []); // Empty deps — telemetryRef is stable

  return (
    <div className="telemetry-panel" ref={wrapperRef} style={{ position: 'relative' }}>
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
    </div>
  );
}

// ── Helper: draw a rounded-rect glass card background ──
function drawCardBg(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();

  ctx.fillStyle = COLORS.bgCard;
  ctx.fill();
  ctx.strokeStyle = COLORS.borderSubtle;
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ── Helper: draw a card header (LABEL + emoji icon) ──
function drawCardHeader(ctx, x, y, w, label, icon) {
  ctx.fillStyle = COLORS.textMuted;
  ctx.font = `10px ${FONT_MONO}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.letterSpacing = '2px';
  ctx.fillText(label, x, y);

  ctx.font = '14px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(icon, x + w, y);
  ctx.letterSpacing = '0px';
}

// ── Helper: draw a horizontal progress bar with gradient ──
function drawBar(ctx, x, y, w, h, pct, colorStops) {
  // Track
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, h / 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.fill();

  // Fill
  if (pct > 0) {
    const fillW = Math.max(h, w * pct); // min width = bar height for rounded caps
    const grad = ctx.createLinearGradient(x, 0, x + fillW, 0);
    colorStops.forEach((color, i) => {
      grad.addColorStop(i / (colorStops.length - 1), color);
    });

    ctx.beginPath();
    ctx.roundRect(x, y, fillW, h, h / 2);
    ctx.fillStyle = grad;
    ctx.fill();
  }
}
