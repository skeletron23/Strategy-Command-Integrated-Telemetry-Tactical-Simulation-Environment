import React from 'react';
import useTelemetry from './hooks/useTelemetry';
import useInference from './hooks/useInference';
import StatusBar from './components/StatusBar';
import TelemetryPanel from './components/TelemetryPanel';
import TrackMap from './components/TrackMap';
import StintProjection from './components/StintProjection';
import LapTimingTable from './components/LapTimingTable';

export default function App() {
  const { telemetryRef, trackPathRef, connected: telemetryConnected, lapTimes, lap } = useTelemetry();
  const { prediction, connected: inferenceConnected } = useInference(lap);

  return (
    <div className="dashboard">
      <StatusBar
        telemetryConnected={telemetryConnected}
        inferenceConnected={inferenceConnected}
        lap={lap}
      />

      <TelemetryPanel telemetryRef={telemetryRef} />

      <TrackMap trackPathRef={trackPathRef} telemetryRef={telemetryRef} />

      <div className="strategy-panel">
        <StintProjection prediction={prediction} />
        <LapTimingTable lapTimes={lapTimes} />
      </div>
    </div>
  );
}
