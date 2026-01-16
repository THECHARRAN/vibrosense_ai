import LiveTimeseriesChart from "../components/charts/LiveTimeseriesChart";

export default function Dashboard() {
  return (
    <div className="dashboard">

      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h1>Hello Operator</h1>
          <p>Real-time vibration intelligence</p>
        </div>
        <span className="live-badge">● Live</span>
      </div>

      {/* Top cards */}
      <div className="metrics">
        <Metric title="Health Score" value="0.87" green />
        <Metric title="Anomaly Score" value="0.12" red />
        <Metric title="Device Status" value="Normal" />
        <Metric title="Uptime" value="3h 24m" />
      </div>

      {/* Charts */}
      <div className="charts">
        {/* ✅ LIVE TIME SERIES (P1) */}
        <LiveTimeseriesChart deviceId="pi-01" />

        {/* Placeholder for future P2 / secondary chart */}
        <div className="card">
          <h3>Anomaly Score</h3>
          <div className="chart-placeholder" />
        </div>
      </div>

      {/* Device row */}
      <div className="device-row">
        <span>Device: pi-01</span>
        <span>Model: v2026.01</span>
        <span className="healthy">● Healthy</span>
      </div>

    </div>
  );
}

function Metric({ title, value, green, red }) {
  return (
    <div className="metric-card">
      <p>{title}</p>
      <h2 className={green ? "green" : red ? "red" : ""}>{value}</h2>
    </div>
  );
}
