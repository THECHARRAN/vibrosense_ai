import ReactECharts from "echarts-for-react";
import { useLiveStream } from "../ws/useLiveStream";
import { useDeviceStore } from "../store/deviceStore";

export default function LiveTimeseriesChart() {
  // Start WebSocket side-effect (safe to call multiple times)
  useLiveStream();

  const timeseries = useDeviceStore((s) => s.timeseries);

  // 🛡️ Safety guard: prevents blank screen
  if (!timeseries || !timeseries.ts || timeseries.ts.length === 0) {
    return (
      <div className="card large">
        <h3>Live Health & Anomaly</h3>
        <div style={{ opacity: 0.6, paddingTop: "20px" }}>
          Waiting for live data…
        </div>
      </div>
    );
  }

  const { ts, health, anomaly } = timeseries;

  const option = {
    animation: false,
    tooltip: { trigger: "axis" },

    legend: {
      data: ["Health Score", "Anomaly Score"],
      textStyle: { color: "#dce1ff" },
    },

    grid: {
      left: 40,
      right: 20,
      top: 30,
      bottom: 40,
    },

    xAxis: {
      type: "time",
      axisLine: { lineStyle: { color: "#6b7190" } },
      axisLabel: { color: "#b3b9d4" },
      splitLine: { show: false },
    },

    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLine: { lineStyle: { color: "#6b7190" } },
      axisLabel: { color: "#b3b9d4" },
      splitLine: {
        lineStyle: { color: "rgba(255,255,255,0.12)" },
      },
    },

    dataZoom: [
      { type: "inside", throttle: 50 },
      { type: "slider" },
    ],

    series: [
      {
        name: "Health Score",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: ts.map((t, i) => [t, health[i]]),
        lineStyle: { width: 3, color: "#3aff7a" },
        areaStyle: { opacity: 0.15, color: "#3aff7a" },
      },
      {
        name: "Anomaly Score",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: ts.map((t, i) => [t, anomaly[i]]),
        lineStyle: { width: 3, color: "#ff6b6b" },
        areaStyle: { opacity: 0.15, color: "#ff6b6b" },
      },
    ],
  };

  return (
    <div className="card large">
      <h3>Live Health & Anomaly</h3>
      <ReactECharts
        option={option}
        style={{ height: "260px", width: "100%" }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
