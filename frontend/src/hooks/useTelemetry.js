import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = 'ws://127.0.0.1:8000/ws/telemetry';
const RECONNECT_MS = 3000;
const TRACK_PATH_MAX = 10000;

// ─── Silverstone GP Circuit (approximated normalized coordinates) ───
// Each waypoint: [x, y, speed_kmh, gear, turnLabel]
// Coordinates normalized 0–1, based on real Silverstone topology
const SILVERSTONE_WAYPOINTS = [
  // Start/Finish Straight → Turn 1 (Abbey)
  [0.52, 0.18, 310, 8, 'Straight'], [0.50, 0.17, 315, 8, 'Straight'],
  [0.48, 0.16, 318, 8, 'Straight'], [0.46, 0.155, 310, 8, 'Straight'],
  [0.44, 0.15, 290, 7, '1'], [0.42, 0.148, 250, 6, '1'],
  [0.40, 0.15, 220, 5, '1'], [0.38, 0.155, 195, 4, '1'],
  // Turn 2 (Farm)
  [0.36, 0.165, 180, 4, '2'], [0.34, 0.18, 170, 3, '2'],
  [0.325, 0.195, 165, 3, '2'],
  // Turn 3 (Village) → Loop
  [0.31, 0.215, 175, 4, '3'], [0.30, 0.235, 190, 4, '3'],
  [0.295, 0.26, 205, 5, 'Straight'], [0.29, 0.28, 220, 5, 'Straight'],
  [0.285, 0.30, 240, 6, 'Straight'],
  // The Loop (Turn 4-5)
  [0.28, 0.32, 230, 5, '4'], [0.275, 0.34, 195, 4, '4'],
  [0.27, 0.36, 160, 3, '5'], [0.275, 0.38, 145, 3, '5'],
  [0.285, 0.395, 140, 3, '5'],
  // Aintree (Turn 6) exit → Wellington Straight
  [0.30, 0.41, 155, 4, '6'], [0.32, 0.42, 175, 4, '6'],
  [0.34, 0.425, 200, 5, 'Straight'], [0.36, 0.43, 225, 5, 'Straight'],
  [0.38, 0.44, 250, 6, 'Straight'], [0.40, 0.45, 270, 7, 'Straight'],
  [0.42, 0.46, 285, 7, 'Straight'],
  // Brooklands (Turn 7)
  [0.44, 0.47, 270, 6, '7'], [0.46, 0.485, 230, 5, '7'],
  [0.475, 0.50, 195, 4, '7'],
  // Luffield (Turn 8-9)
  [0.49, 0.515, 170, 3, '8'], [0.50, 0.535, 150, 3, '8'],
  [0.505, 0.555, 140, 3, '9'], [0.51, 0.575, 135, 3, '9'],
  [0.52, 0.59, 140, 3, '9'],
  // Woodcote exit → Copse approach
  [0.535, 0.60, 165, 4, 'Straight'], [0.55, 0.605, 195, 5, 'Straight'],
  [0.57, 0.61, 225, 6, 'Straight'], [0.59, 0.605, 255, 6, 'Straight'],
  [0.61, 0.595, 275, 7, 'Straight'], [0.63, 0.58, 290, 7, 'Straight'],
  [0.65, 0.565, 300, 8, 'Straight'],
  // Copse (Turn 10)
  [0.67, 0.545, 290, 7, '10'], [0.685, 0.525, 275, 7, '10'],
  [0.695, 0.505, 265, 6, '10'],
  // Maggots-Becketts complex (Turns 11-14)
  [0.70, 0.48, 280, 7, 'Straight'], [0.705, 0.46, 290, 7, 'Straight'],
  [0.71, 0.44, 265, 6, '11'], [0.72, 0.42, 235, 5, '11'],
  [0.735, 0.405, 210, 5, '12'], [0.745, 0.39, 225, 5, '12'],
  [0.755, 0.375, 245, 6, '13'], [0.76, 0.355, 220, 5, '13'],
  [0.765, 0.335, 235, 5, '14'], [0.77, 0.315, 250, 6, '14'],
  // Chapel (Turn 15) → Hangar Straight
  [0.775, 0.295, 270, 7, '15'], [0.775, 0.275, 285, 7, '15'],
  [0.77, 0.255, 295, 7, 'Straight'], [0.765, 0.235, 310, 8, 'Straight'],
  [0.755, 0.22, 318, 8, 'Straight'], [0.74, 0.21, 322, 8, 'Straight'],
  [0.72, 0.205, 325, 8, 'Straight'], [0.70, 0.20, 328, 8, 'Straight'],
  [0.68, 0.195, 330, 8, 'Straight'], [0.66, 0.19, 332, 8, 'Straight'],
  // Stowe (Turn 16)
  [0.64, 0.185, 320, 8, '16'], [0.62, 0.18, 290, 7, '16'],
  [0.605, 0.175, 260, 6, '16'],
  // Vale–Club (Turns 17-18) → back to S/F
  [0.59, 0.17, 230, 5, '17'], [0.58, 0.165, 195, 4, '17'],
  [0.57, 0.16, 160, 3, '17'], [0.565, 0.155, 140, 3, '18'],
  [0.56, 0.16, 130, 3, '18'], [0.555, 0.165, 135, 3, '18'],
  [0.55, 0.17, 155, 4, 'Straight'], [0.54, 0.175, 200, 5, 'Straight'],
  [0.53, 0.178, 260, 6, 'Straight'], [0.52, 0.18, 295, 7, 'Straight'],
];

function interpolateWaypoints(waypoints, totalPoints) {
  const result = [];
  const n = waypoints.length;
  for (let i = 0; i < totalPoints; i++) {
    const t = (i / totalPoints) * n;
    const idx = Math.floor(t) % n;
    const next = (idx + 1) % n;
    const frac = t - Math.floor(t);

    const [x1, y1, s1, g1, turn1] = waypoints[idx];
    const [x2, y2, s2, g2] = waypoints[next];

    result.push({
      x: x1 + (x2 - x1) * frac,
      y: y1 + (y2 - y1) * frac,
      speed: Math.round(s1 + (s2 - s1) * frac),
      gear: frac < 0.5 ? g1 : g2,
      turn: turn1,
    });
  }
  return result;
}

// Pre-compute a full lap of demo frames
const DEMO_FRAMES_PER_LAP = 600;
const DEMO_LAP = interpolateWaypoints(SILVERSTONE_WAYPOINTS, DEMO_FRAMES_PER_LAP);

function generateDemoFrame(tick) {
  const frameIdx = tick % DEMO_LAP.length;
  const lap = Math.floor(tick / DEMO_LAP.length) + 1;
  const frame = DEMO_LAP[frameIdx];

  const rpm = Math.round(5500 + (frame.speed / 335) * 9500);
  const throttle = frame.speed > 200 ? Math.min(100, Math.round((frame.speed - 100) / 2.35)) : Math.round(frame.speed / 3.5);
  const brake = (() => {
    // Check if decelerating from previous frame
    const prevIdx = (frameIdx - 1 + DEMO_LAP.length) % DEMO_LAP.length;
    const speedDrop = DEMO_LAP[prevIdx].speed - frame.speed;
    if (speedDrop > 5) return Math.min(100, Math.round(speedDrop * 3.5));
    return 0;
  })();

  const prevIdx = (frameIdx - 1 + DEMO_LAP.length) % DEMO_LAP.length;
  const speedDelta = frame.speed - DEMO_LAP[prevIdx].speed;
  const gForce = parseFloat((speedDelta / 15).toFixed(2));

  return {
    speed: frame.speed,
    rpm,
    gear: frame.gear,
    throttle,
    brake,
    g_force: gForce,
    x: frame.x,
    y: frame.y,
    lap,
    turn: frame.turn,
  };
}

export default function useTelemetry() {
  const [telemetry, setTelemetry] = useState(null);
  const [trackPath, setTrackPath] = useState([]);
  const [connected, setConnected] = useState(false);
  const [lapTimes, setLapTimes] = useState([]);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const demoTimer = useRef(null);
  const demoTick = useRef(0);
  const currentLap = useRef(0);
  const lapStart = useRef(Date.now());

  const clearDemo = useCallback(() => {
    if (demoTimer.current) {
      clearInterval(demoTimer.current);
      demoTimer.current = null;
    }
  }, []);

  const processFrame = useCallback((data) => {
    setTelemetry(data);
    setTrackPath((prev) => {
      const next = [...prev, { x: data.x, y: data.y }];
      return next.length > TRACK_PATH_MAX ? next.slice(-TRACK_PATH_MAX) : next;
    });

    if (data.lap !== currentLap.current && currentLap.current !== 0) {
      const now = Date.now();
      const lapTime = ((now - lapStart.current) / 1000).toFixed(3);
      setLapTimes((prev) => {
        const base = 90 + Math.random() * 3;
        const time = currentLap.current === 1 ? parseFloat(lapTime) : parseFloat(base.toFixed(3));
        const newTimes = [...prev, { lap: currentLap.current, time }];
        return newTimes.slice(-30);
      });
      lapStart.current = now;
    }
    if (data.lap !== currentLap.current) {
      currentLap.current = data.lap;
      lapStart.current = Date.now();
    }
  }, []);

  const startDemo = useCallback(() => {
    clearDemo();
    demoTick.current = 0;
    demoTimer.current = setInterval(() => {
      const frame = generateDemoFrame(demoTick.current++);
      processFrame(frame);
    }, 50); // 20 Hz demo
  }, [clearDemo, processFrame]);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        clearDemo();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          processFrame(data);
        } catch { /* ignore parse errors */ }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
        startDemo();
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      setConnected(false);
      startDemo();
      reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
    }
  }, [clearDemo, startDemo, processFrame]);

  useEffect(() => {
    connect();

    return () => {
      clearDemo();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { telemetry, trackPath, connected, lapTimes };
}
