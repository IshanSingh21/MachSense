"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  Database,
  Globe,
  Radio,
  RefreshCw,
  Server,
  ShieldCheck,
  Terminal,
  Zap,
} from "lucide-react";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api } from "@/lib/api";
import { HealthResponse, ReadinessResponse } from "@/types";

const SENSOR_BOUNDARIES = [
  { parameter: "Air Temperature (T_air)", min: "295.0 K (21.85 °C)", max: "305.0 K (31.85 °C)", nominal: "298 - 302 K", unit: "Kelvin" },
  { parameter: "Process Temperature (T_process)", min: "305.0 K (31.85 °C)", max: "315.0 K (41.85 °C)", nominal: "308 - 312 K", unit: "Kelvin" },
  { parameter: "Rotational Speed (ω)", min: "1100 RPM", max: "2900 RPM", nominal: "1380 - 1800 RPM", unit: "RPM" },
  { parameter: "Mechanical Torque (τ)", min: "3.0 Nm", max: "80.0 Nm", nominal: "10.0 - 55.0 Nm", unit: "Newton-meters" },
  { parameter: "Tool Wear Duration (t_wear)", min: "0 min", max: "260 min", nominal: "< 200 min", unit: "Minutes" },
  { parameter: "Thermodynamic Limit", min: "T_process > T_air", max: "ΔT ≥ 8.6 K", nominal: "ΔT ≈ 10.0 K", unit: "Physics Invariant" },
];

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [pingLoading, setPingLoading] = useState<boolean>(false);
  const [pingError, setPingError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setPingLoading(true);
    setPingError(null);
    try {
      const [h, r] = await Promise.all([api.getHealth(), api.getReadiness()]);
      setHealth(h);
      setReadiness(r);
    } catch (err: unknown) {
      setPingError(
        err instanceof Error
          ? err.message
          : "Failed to connect to FastAPI backend on " + api.getBaseUrl()
      );
    } finally {
      setPingLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="space-y-6">
      {/* 1. Header & Live Ping */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-slate-100">MachSense System Diagnostics & Settings</h2>
          <p className="text-xs text-slate-400 mt-1">
            Environment telemetry, FastAPI health probes, ML model registry metadata, and physical safety boundaries.
          </p>
        </div>

        <button
          onClick={fetchStatus}
          disabled={pingLoading}
          className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-2 border border-slate-700 transition-colors cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${pingLoading ? "animate-spin" : ""}`} />
          <span>{pingLoading ? "Probing Backend..." : "Test Connection"}</span>
        </button>
      </div>

      {/* 2. System Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Backend API Probe */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                FastAPI Backend Probe
              </h3>
            </div>
            <StatusBadge status={health ? "HEALTHY" : "WARNING"} size="sm" />
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Target Endpoint</span>
              <span className="text-cyan-400">{api.getBaseUrl()}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Liveness Probe</span>
              <span className={health ? "text-emerald-400" : "text-amber-400"}>
                {health ? `${health.status.toUpperCase()} (HTTP 200)` : "UNREACHABLE"}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Environment</span>
              <span className="text-slate-200">{health?.environment || "development"}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">API Version</span>
              <span className="text-slate-200">{health?.version || "0.1.0"}</span>
            </div>
          </div>
        </div>

        {/* Model Registry Card */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Active ML Model
              </h3>
            </div>
            <StatusBadge status={readiness?.model_loaded ? "HEALTHY" : "WARNING"} size="sm" />
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Champion Model</span>
              <span className="text-slate-200">{readiness?.model_name || "tuned_random_forest"}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Version Tag</span>
              <span className="text-cyan-400">{readiness?.model_version || "v1.0.0"}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Decision Threshold</span>
              <span className="text-amber-400 font-bold">{readiness?.optimal_threshold || "0.5608"}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">TreeSHAP Engine</span>
              <span className="text-emerald-400">INITIALIZED & READY</span>
            </div>
          </div>
        </div>

        {/* Frontend Console */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Frontend Console
              </h3>
            </div>
            <StatusBadge status="HEALTHY" size="sm" label="OPERATIONAL" />
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Framework</span>
              <span className="text-slate-200">Next.js 16 + React 19</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Styling & Charts</span>
              <span className="text-slate-200">Tailwind v4 + Recharts</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Architecture</span>
              <span className="text-slate-200">Dark Industrial Control Center</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Client Version</span>
              <span className="text-cyan-400">v1.0.0</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Physical Sensor Boundaries Table */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex items-center gap-2 mb-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Physical Sensor Boundaries & Domain Validation Rules
          </h3>
        </div>
        <p className="text-xs text-slate-400 mb-4">
          Strict runtime bounds enforced by the domain validator before ML preprocessor transformation.
        </p>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Telemetry Parameter</th>
                <th className="py-3 px-4">Lower Bound</th>
                <th className="py-3 px-4">Upper Bound</th>
                <th className="py-3 px-4">Nominal Safe Range</th>
                <th className="py-3 px-4">Engineering Unit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              {SENSOR_BOUNDARIES.map((b, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40">
                  <td className="py-3 px-4 font-sans font-semibold text-slate-200">{b.parameter}</td>
                  <td className="py-3 px-4 text-slate-400">{b.min}</td>
                  <td className="py-3 px-4 text-slate-400">{b.max}</td>
                  <td className="py-3 px-4 text-emerald-400 font-bold">{b.nominal}</td>
                  <td className="py-3 px-4 text-slate-400">{b.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
