import React from 'react';
import useTelemetry from './hooks/useTelemetry';
import useInference from './hooks/useInference';
import StatusBar from './components/StatusBar';
import TelemetryPanel from './components/TelemetryPanel';
import TrackMap from './components/TrackMap';
import StintProjection from './components/StintProjection';
import LapTimingTable from './components/LapTimingTable';

export default function App() {
  const { telemetry, trackPath, connected: telemetryConnected, lapTimes } = useTelemetry();
  const { prediction, connected: inferenceConnected } = useInference(telemetry?.lap);

  return (
    <div className="dashboard">
      <StatusBar
        telemetryConnected={telemetryConnected}
        inferenceConnected={inferenceConnected}
        lap={telemetry?.lap}
      />

      <TelemetryPanel telemetry={telemetry} />

      <TrackMap trackPath={trackPath} telemetry={telemetry} />

      <div className="strategy-panel">
        <StintProjection prediction={prediction} />
        <LapTimingTable lapTimes={lapTimes} />
      </div>
    </div>
  );
}
