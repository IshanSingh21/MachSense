"use client";

import React, { use, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowLeft,
  Calendar,
  CheckCircle2,
  Clock,
  Cpu,
  Flame,
  Gauge,
  History,
  Info,
  Layers,
  Radio,
  RefreshCw,
  Shield,
  Sliders,
  Sparkles,
  Wrench,
  Zap,
} from "lucide-react";
import { ShapWaterfallChart } from "@/components/charts/ShapWaterfallChart";
import { TelemetryStreamChart } from "@/components/charts/TelemetryStreamChart";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { TelemetryCard } from "@/components/ui/TelemetryCard";
import { api } from "@/lib/api";
import { MOCK_MACHINES, MOCK_TELEMETRY_SERIES } from "@/lib/mockData";
import { PredictionResponse } from "@/types";

export default function MachineDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const machine = MOCK_MACHINES.find((m) => m.id === id);

  const [activeChartTab, setActiveChartTab] = useState<
    "temp_diff" | "power" | "torque" | "failure_prob"
  >("temp_diff");

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!machine) {
    return (
      <div className="text-center py-20 bg-[#111827] border border-slate-800 rounded-2xl p-8">
        <Cpu className="w-12 h-12 text-slate-600 mx-auto mb-3" />
        <h2 className="text-base font-bold text-slate-100">Machine Not Found</h2>
        <p className="text-xs text-slate-400 mt-1">
          No telemetry record matches the machine ID <code className="text-cyan-400">{id}</code>.
        </p>
        <Link
          href="/machines"
          className="mt-5 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-cyan-600 text-white text-xs font-semibold"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Fleet Directory</span>
        </Link>
      </div>
    );
  }

  // Calculate composite machine health score (0 - 100)
  const healthScore = Math.max(0, Math.round((1 - machine.failure_probability) * 100));

  const handleRunDiagnostic = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictSingle(machine.telemetry, true);
      setPrediction(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Inference failed. Verify the FastAPI backend is running on http://127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 1. Header Bar with Back Link, Status, and Diagnostic Trigger */}
      <div className="bg-[#111827] border border-slate-800 p-5 sm:p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link
            href="/machines"
            className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
            title="Back to Fleet"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>

          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <h2 className="text-base sm:text-lg font-bold text-slate-100">{machine.name}</h2>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold">
                {machine.id}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60 font-semibold">
                Variant Type {machine.type}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Location: <strong className="text-slate-300">{machine.location}</strong> • Serial:{" "}
              <span className="font-mono text-slate-300">{machine.telemetry.product_id}</span> • UDI:{" "}
              <span className="font-mono text-slate-300">{machine.telemetry.udi}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-end">
          <StatusBadge status={machine.status} size="md" />

          <button
            onClick={handleRunDiagnostic}
            disabled={loading}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white text-xs font-bold shadow-lg shadow-cyan-600/20 flex items-center gap-2 transition-all cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>{loading ? "Evaluating Machine..." : "Live Sensor Inference"}</span>
          </button>
        </div>
      </div>

      {/* 2. Top KPI Cards: Health Score, Failure Prob, Power & Overstrain */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Composite Health Score */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Condition Health Score
            </span>
            <div className="flex items-baseline gap-1 mt-1">
              <span
                className={`text-2xl font-bold font-mono ${
                  healthScore > 80
                    ? "text-emerald-400"
                    : healthScore > 40
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}
              >
                {healthScore}
              </span>
              <span className="text-xs text-slate-400">/ 100</span>
            </div>
          </div>
          <div
            className={`p-3 rounded-xl border ${
              healthScore > 80
                ? "bg-emerald-950/40 text-emerald-400 border-emerald-800/50"
                : healthScore > 40
                ? "bg-amber-950/40 text-amber-400 border-amber-800/50"
                : "bg-rose-950/40 text-rose-400 border-rose-800/50"
            }`}
          >
            <Shield className="w-5 h-5" />
          </div>
        </div>

        {/* Failure Probability */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Failure Probability
            </span>
            <div className="text-2xl font-bold font-mono text-rose-400 mt-1">
              {(machine.failure_probability * 100).toFixed(1)}%
            </div>
          </div>
          <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-400">
            <AlertOctagon className="w-5 h-5" />
          </div>
        </div>

        {/* Mechanical Power */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Spindle Power (P=τω)
            </span>
            <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
              {machine.derived.power_w.toLocaleString()} W
            </div>
          </div>
          <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-800/50 text-cyan-400">
            <Zap className="w-5 h-5" />
          </div>
        </div>

        {/* Overstrain Index */}
        <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl flex items-center justify-between">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Overstrain Index
            </span>
            <div className="text-2xl font-bold font-mono text-purple-400 mt-1">
              {machine.derived.overstrain_index.toLocaleString()}
            </div>
          </div>
          <div className="p-3 rounded-xl bg-purple-950/40 border border-purple-800/50 text-purple-400">
            <Activity className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 3. Sensor Telemetry Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <TelemetryCard
          label="Air Temp"
          value={machine.telemetry.air_temperature_k}
          unit="K"
          nominalRange="295 - 304 K"
        />
        <TelemetryCard
          label="Process Temp"
          value={machine.telemetry.process_temperature_k}
          unit="K"
          nominalRange="305 - 314 K"
        />
        <TelemetryCard
          label="Thermal ΔT"
          value={machine.derived.temp_diff_k.toFixed(1)}
          unit="K"
          status={machine.derived.temp_diff_k < 8.6 ? "critical" : "normal"}
          nominalRange="> 8.6 K"
        />
        <TelemetryCard
          label="Spindle Speed"
          value={machine.telemetry.rotational_speed_rpm}
          unit="RPM"
          nominalRange="1380 - 2800"
        />
        <TelemetryCard
          label="Cutting Torque"
          value={machine.telemetry.torque_nm}
          unit="Nm"
          status={machine.telemetry.torque_nm > 60 ? "critical" : "normal"}
          nominalRange="10 - 55 Nm"
        />
        <TelemetryCard
          label="Tool Wear"
          value={machine.telemetry.tool_wear_min}
          unit="min"
          status={machine.telemetry.tool_wear_min > 200 ? "warning" : "normal"}
          nominalRange="< 200 min"
        />
      </div>

      {/* 4. Live FastAPI Prediction Evaluation Result */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800 text-xs text-rose-300 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">FastAPI Evaluation Error: </span>
            {error}
          </div>
        </div>
      )}

      {prediction && (
        <div className="bg-[#111827] border border-cyan-800/80 rounded-2xl p-6 shadow-xl shadow-cyan-950/20 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-xl bg-cyan-950 text-cyan-400 border border-cyan-800">
                <Activity className="w-5 h-5" />
              </span>
              <div>
                <h3 className="text-sm font-bold text-slate-100">
                  FastAPI Real-Time Evaluation Result ({prediction.model_version})
                </h3>
                <span className="text-xs text-slate-400">
                  Latency: {prediction.latency_ms.toFixed(2)} ms • Applied Threshold: {prediction.threshold_used}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <StatusBadge
                status={prediction.predicted_label === "HEALTHY" ? "HEALTHY" : "CRITICAL"}
                label={prediction.predicted_label}
              />
              <span className="font-mono text-lg font-bold text-slate-100">
                {(prediction.failure_probability * 100).toFixed(2)}% Failure Risk
              </span>
            </div>
          </div>

          {prediction.operator_summary && (
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-200 leading-relaxed font-mono">
              <span className="text-cyan-400 font-bold">Diagnostic Alert: </span>
              {prediction.operator_summary}
            </div>
          )}

          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
              Instance TreeSHAP Attribution Breakdown
            </h4>
            <ShapWaterfallChart
              escalators={prediction.top_risk_escalators}
              stabilizers={prediction.top_stabilizers}
              height={200}
            />
          </div>
        </div>
      )}

      {/* 5. Historical Sensor Behavior & Telemetry Degradation Charts */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Sensor Telemetry Behavior Timeline
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Historical sensor signals mapped with physical warning boundaries
            </p>
          </div>

          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
            <button
              onClick={() => setActiveChartTab("temp_diff")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                activeChartTab === "temp_diff"
                  ? "bg-amber-950 text-amber-300 border border-amber-800/60 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Thermal ΔT (K)
            </button>
            <button
              onClick={() => setActiveChartTab("power")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                activeChartTab === "power"
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-800/60 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Power P=τω (W)
            </button>
            <button
              onClick={() => setActiveChartTab("torque")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                activeChartTab === "torque"
                  ? "bg-purple-950 text-purple-300 border border-purple-800/60 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Torque (Nm)
            </button>
            <button
              onClick={() => setActiveChartTab("failure_prob")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                activeChartTab === "failure_prob"
                  ? "bg-rose-950 text-rose-300 border border-rose-800/60 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Failure Risk %
            </button>
          </div>
        </div>

        <TelemetryStreamChart data={MOCK_TELEMETRY_SERIES} metricKey={activeChartTab} height={260} />
      </div>

      {/* 6. Machine Maintenance & Operational Log */}
      <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex items-center gap-2 mb-4">
          <History className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Maintenance History & Telemetry Log
          </h3>
        </div>

        <div className="space-y-3 text-xs">
          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <span className="p-1.5 rounded-lg bg-rose-950/60 text-rose-400 border border-rose-900/60 mt-0.5">
                <AlertTriangle className="w-3.5 h-3.5" />
              </span>
              <div>
                <span className="font-bold text-slate-200">Overstrain Precursor Triggered</span>
                <p className="text-slate-400 mt-0.5">
                  Torque reached 62.5 Nm with insert wear at 215 min. Failure probability spiked to 94.1%.
                </p>
              </div>
            </div>
            <span className="font-mono text-slate-400 shrink-0">Today, 10:42 UTC</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <span className="p-1.5 rounded-lg bg-cyan-950/60 text-cyan-400 border border-cyan-900/60 mt-0.5">
                <Wrench className="w-3.5 h-3.5" />
              </span>
              <div>
                <span className="font-bold text-slate-200">Spindle Bearings Lubrication Routine</span>
                <p className="text-slate-400 mt-0.5">
                  Scheduled preventive lubrication completed by technician. Spindle runout &lt; 2 µm.
                </p>
              </div>
            </div>
            <span className="font-mono text-slate-400 shrink-0">2 days ago</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <span className="p-1.5 rounded-lg bg-emerald-950/60 text-emerald-400 border border-emerald-900/60 mt-0.5">
                <CheckCircle2 className="w-3.5 h-3.5" />
              </span>
              <div>
                <span className="font-bold text-slate-200">Calibration & Domain Boundary Check</span>
                <p className="text-slate-400 mt-0.5">
                  All 5 thermal and rotational telemetry channels verified against ISO standards.
                </p>
              </div>
            </div>
            <span className="font-mono text-slate-400 shrink-0">5 days ago</span>
          </div>
        </div>
      </div>
    </div>
  );
}
