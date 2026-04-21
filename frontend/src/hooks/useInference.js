import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = 'ws://127.0.0.1:8001/ws/predict/tire-degradation';
const RECONNECT_MS = 5000;
const SEND_INTERVAL_MS = 1500;

function generateDemoProjection() {
  const base = 92 + Math.random() * 2;
  const projection = [];
  for (let i = 1; i <= 6; i++) {
    projection.push({
      lap_offset: i,
      projected_lap_number: 10 + i,
      projected_tyre_life: 10 + i,
      predicted_lap_time_s: parseFloat((base + i * 0.08 + Math.random() * 0.05).toFixed(3)),
    });
  }
  return {
    projection,
    summary: {
      first_projected_lap_time_s: projection[0].predicted_lap_time_s,
      last_projected_lap_time_s: projection[projection.length - 1].predicted_lap_time_s,
      stint_time_delta_s: parseFloat((projection[projection.length - 1].predicted_lap_time_s - projection[0].predicted_lap_time_s).toFixed(3)),
    },
  };
}

export default function useInference(currentLap = 1) {
  const [prediction, setPrediction] = useState(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const sendTimer = useRef(null);
  const demoTimer = useRef(null);
  const lapRef = useRef(currentLap);

  useEffect(() => { lapRef.current = currentLap; }, [currentLap]);

  const clearDemo = useCallback(() => {
    if (demoTimer.current) {
      clearInterval(demoTimer.current);
      demoTimer.current = null;
    }
  }, []);

  const startDemo = useCallback(() => {
    clearDemo();
    demoTimer.current = setInterval(() => {
      setPrediction(generateDemoProjection());
    }, 2000);
    setPrediction(generateDemoProjection());
  }, [clearDemo]);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        clearDemo();

        // Send periodic inference requests
        sendTimer.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            const payload = {
              tyre_life: Math.max(1, lapRef.current),
              compound: 'MEDIUM',
              lap_number: Math.max(1, lapRef.current),
              stint_laps_ahead: 6,
              circuit: 'Great Britain',
              fresh_tyre: 0,
              track_temp: 32.0,
              air_temp: 24.0,
              humidity: 45.0,
              pressure: 1013.0,
              wind_speed: 3.0,
              total_race_laps: 52,
              year: 2024,
            };
            ws.send(JSON.stringify(payload));
          }
        }, SEND_INTERVAL_MS);
      };

      ws.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);
          if (result && result.projection && result.projection.length > 0) {
            setPrediction(result);
          }
        } catch { /* ignore */ }
      };

      ws.onclose = () => {
        setConnected(false);
        if (sendTimer.current) clearInterval(sendTimer.current);
        reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
        startDemo();
      };

      ws.onerror = () => { ws.close(); };
    } catch {
      setConnected(false);
      startDemo();
      reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
    }
  }, [clearDemo, startDemo]);

  useEffect(() => {
    connect();
    return () => {
      clearDemo();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (sendTimer.current) clearInterval(sendTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { prediction, connected };
}
