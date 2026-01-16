import { create } from "zustand";

const MAX_POINTS = 600; // ~10 min @ 1s

export const useDeviceStore = create((set) => ({
  deviceId: "pi-01",

  timeseries: {
    ts: [],
    health: [],
    anomaly: [],
  },

  latestWindow: null,
  alerts: [],

  pushTimeseries: (point) =>
    set((state) => {
      const ts = [...state.timeseries.ts, point.ts].slice(-MAX_POINTS);
      const health = [...state.timeseries.health, point.health].slice(-MAX_POINTS);
      const anomaly = [...state.timeseries.anomaly, point.anomaly].slice(-MAX_POINTS);

      return {
        timeseries: { ts, health, anomaly },
      };
    }),

  setLatestWindow: (window) => set({ latestWindow: window }),

  addAlert: (alert) =>
    set((state) => ({
      alerts: [alert, ...state.alerts],
    })),
}));
