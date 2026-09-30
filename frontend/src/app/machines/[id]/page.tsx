"use client";

import React, { use, useState } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Cpu,
  Flame,
  Gauge,
  Layers,
  Radio,
  RefreshCw,
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

  if (!machine) {
    return (
      <div className="text-center py-20 bg-[#111827] border border-slate-800 rounded-2xl">
        <h2 className="text-lg font-bold text-slate-200">Machine Not Found</h2>
        <p className="text-xs text-slate-400 mt-1">No machine with ID {id} was found in the active telemetry registry.</p>
        <Link
          href="/machines"
          className="mt-4 inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 font-semibold"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Return to Machine Fleet</span>
        </Link>
      </div>
    );
  }

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
          : "Could not execute prediction via FastAPI backend. Ensure backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 1. Header Navigation & Machine Title */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-[#111827] border border-slate-800 p-6 rounded-2xl">
        <div className="flex items-center gap-4">
          <Link
            href="/machines"
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-bold text-slate-100">{machine.name}</h2>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold">
                {machine.id}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60 font-semibold">
                Type {machine.type}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              {machine.location} • UDI: {machine.telemetry.udi} • Product ID: {machine.telemetry.product_id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto justify-end">
          <StatusBadge status={machine.status} size="md" />
          <button
            onClick={handleRunDiagnostic}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-cyan-600/20 flex items-center gap-2 transition-all cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>{loading ? "Evaluating Telemetry..." : "Run Live Inference"}</span>
          </button>
        </div>
      </div>

      {/* 2. Sensor Telemetry Grid */}
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

      {/* 3. Live Diagnostic Evaluation Result Box */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Inference Request Failed: </span>
            {error}
          </div>
        </div>
      )}

      {prediction && (
        <div className="bg-[#111827] border border-cyan-800/60 rounded-2xl p-6 shadow-lg shadow-cyan-950/20">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-xl bg-cyan-950 border border-cyan-800 text-cyan-400">
                <Activity className="w-5 h-5" />
              </span>
              <div>
                <h3 className="text-sm font-bold text-slate-100">
                  FastAPI Real-Time Evaluation Result ({prediction.model_version})
                </h3>
                <span className="text-xs text-slate-400">
                  Inference latency: {prediction.latency_ms.toFixed(2)} ms • Applied Threshold: {prediction.threshold_used}
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

          {/* Operator Summary */}
          {prediction.operator_summary && (
            <div className="my-4 p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-200 leading-relaxed">
              <span className="font-bold text-cyan-400">Diagnostic Summary: </span>
              {prediction.operator_summary}
            </div>
          )}

          {/* Feature Attribution Chart */}
          <div className="mt-4">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
              Instance SHAP Attribution Breakdown
            </h4>
            <ShapWaterfallChart
              escalators={prediction.top_risk_escalators}
              stabilizers={prediction.top_stabilizers}
              height={200}
            />
          </div>
        </div>
      )}

      {/* 4. Telemetry Historical Degradation Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-amber-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Thermal & Heat Dissipation (ΔT)
              </h3>
            </div>
            <span className="text-[11px] text-slate-400 font-mono">Last 3 Hours</span>
          </div>
          <TelemetryStreamChart data={MOCK_TELEMETRY_SERIES} metricKey="temp_diff" height={220} />
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Mechanical Power Load (P = τ·ω)
              </h3>
            </div>
            <span className="text-[11px] text-slate-400 font-mono">Spindle Wattage</span>
          </div>
          <TelemetryStreamChart data={MOCK_TELEMETRY_SERIES} metricKey="power" height={220} />
        </div>
      </div>
    </div>
  );
}
