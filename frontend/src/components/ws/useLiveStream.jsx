import { useEffect, useRef } from "react";
import { useDeviceStore } from "../store/deviceStore";

export function useLiveStream() {
  const wsRef = useRef(null);

  const deviceId = useDeviceStore((s) => s.deviceId);
  const pushTimeseries = useDeviceStore((s) => s.pushTimeseries);
  const setLatestWindow = useDeviceStore((s) => s.setLatestWindow);
  const addAlert = useDeviceStore((s) => s.addAlert);

  useEffect(() => {
    // prevent multiple sockets
    if (wsRef.current) return;

    const ws = new WebSocket("ws://localhost:8000/ws/live");
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.device_id !== deviceId) return;

      if (msg.type === "health") {
        pushTimeseries({
          ts: msg.ts * 1000,
          health: msg.health_score,
          anomaly: msg.anomaly_score,
        });
      }

      if (msg.type === "feature") {
        setLatestWindow(msg);
      }

      if (msg.type === "alert") {
        addAlert(msg);
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
    };
  }, [deviceId]);
}
