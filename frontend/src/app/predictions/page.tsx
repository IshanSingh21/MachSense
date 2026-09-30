"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileSpreadsheet,
  Info,
  Layers,
  Play,
  RefreshCw,
  Sliders,
  Sparkles,
  Upload,
} from "lucide-react";
import { ShapWaterfallChart } from "@/components/charts/ShapWaterfallChart";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api } from "@/lib/api";
import { FAILURE_PRESETS } from "@/lib/mockData";
import {
  BatchPredictionResponse,
  MachineType,
  PredictionResponse,
  TelemetryItem,
} from "@/types";

export default function PredictionsPage() {
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single");

  // Single Form Telemetry State
  const [telemetry, setTelemetry] = useState<TelemetryItem>({
    type: "L",
    air_temperature_k: 302.5,
    process_temperature_k: 311.8,
    rotational_speed_rpm: 1380.0,
    torque_nm: 62.5,
    tool_wear_min: 215.0,
  });

  const [threshold, setThreshold] = useState<number>(0.5608);
  const [explain, setExplain] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Batch CSV State
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // Apply failure preset
  const handleApplyPreset = (presetId: string) => {
    const p = FAILURE_PRESETS.find((x) => x.id === presetId);
    if (p) {
      setTelemetry({ ...p.telemetry });
      setPrediction(null);
      setError(null);
    }
  };

  // Run single prediction
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await api.predictSingle(telemetry, explain, threshold);
      setPrediction(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Prediction failed. Verify that FastAPI server is active on http://127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  // Batch evaluation demo
  const handleRunSampleBatch = async () => {
    setBatchLoading(true);
    setBatchError(null);

    const sampleFleet: TelemetryItem[] = [
      { type: "M", air_temperature_k: 298.1, process_temperature_k: 308.6, rotational_speed_rpm: 1550, torque_nm: 40, tool_wear_min: 30, udi: 1 },
      { type: "L", air_temperature_k: 302.8, process_temperature_k: 311.2, rotational_speed_rpm: 1360, torque_nm: 52, tool_wear_min: 105, udi: 2 },
      { type: "L", air_temperature_k: 300.5, process_temperature_k: 310.8, rotational_speed_rpm: 2840, torque_nm: 65, tool_wear_min: 35, udi: 3 },
      { type: "L", air_temperature_k: 302.5, process_temperature_k: 311.8, rotational_speed_rpm: 1380, torque_nm: 62.5, tool_wear_min: 215, udi: 4 },
      { type: "M", air_temperature_k: 299.0, process_temperature_k: 309.2, rotational_speed_rpm: 1420, torque_nm: 48, tool_wear_min: 240, udi: 5 },
    ];

    try {
      const res = await api.predictBatch(sampleFleet, false, threshold);
      setBatchResult(res);
    } catch (err: unknown) {
      setBatchError(
        err instanceof Error
          ? err.message
          : "Batch evaluation failed. Ensure backend server is active."
      );
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 1. Mode Switcher Tabs */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
          <button
            onClick={() => setActiveTab("single")}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              activeTab === "single"
                ? "bg-cyan-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Single Telemetry Inference</span>
          </button>
          <button
            onClick={() => setActiveTab("batch")}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors flex items-center gap-2 ${
              activeTab === "batch"
                ? "bg-cyan-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            <span>Fleet Batch Scoring</span>
          </button>
        </div>

        <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>Champion Model: Tuned Random Forest (v1.0.0)</span>
        </div>
      </div>

      {activeTab === "single" ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Presets & Form Controls (5 cols) */}
          <div className="lg:col-span-5 space-y-5">
            {/* Presets Selector */}
            <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  Domain Failure Presets
                </h3>
              </div>
              <p className="text-xs text-slate-400 mb-3">
                Load physical sensor failure signatures directly into the telemetry inputs:
              </p>

              <div className="grid grid-cols-1 gap-2">
                {FAILURE_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleApplyPreset(preset.id)}
                    className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800/80 border border-slate-800 text-left transition-all group flex items-start justify-between gap-2"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-200 group-hover:text-cyan-400 transition-colors">
                          {preset.name}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 font-mono text-slate-400 font-semibold">
                          {preset.short_code}
                        </span>
                      </div>
                      <span className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">
                        {preset.description}
                      </span>
                    </div>
                    <span className="text-[10px] font-mono text-cyan-400 shrink-0 font-medium mt-0.5">
                      Load
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Telemetry Input Form */}
            <form
              onSubmit={handlePredict}
              className="bg-[#111827] border border-slate-800 p-5 rounded-2xl space-y-4"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Telemetry Input Parameters
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-slate-400">SI Units</span>
              </div>

              {/* Machine Type */}
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">
                  Machine Quality Type
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(["L", "M", "H"] as MachineType[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setTelemetry({ ...telemetry, type: t })}
                      className={`py-1.5 rounded-lg text-xs font-bold font-mono transition-colors ${
                        telemetry.type === t
                          ? "bg-cyan-600 text-white"
                          : "bg-slate-900 text-slate-400 border border-slate-800 hover:border-slate-700"
                      }`}
                    >
                      Type {t} {t === "L" ? "(Low)" : t === "M" ? "(Med)" : "(High)"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Air Temperature */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Air Temperature (K)</span>
                  <span className="font-mono text-cyan-400 font-bold">{telemetry.air_temperature_k} K</span>
                </div>
                <input
                  type="range"
                  min="295"
                  max="305"
                  step="0.1"
                  value={telemetry.air_temperature_k}
                  onChange={(e) =>
                    setTelemetry({ ...telemetry, air_temperature_k: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Process Temperature */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Process Temperature (K)</span>
                  <span className="font-mono text-cyan-400 font-bold">{telemetry.process_temperature_k} K</span>
                </div>
                <input
                  type="range"
                  min="305"
                  max="315"
                  step="0.1"
                  value={telemetry.process_temperature_k}
                  onChange={(e) =>
                    setTelemetry({ ...telemetry, process_temperature_k: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Rotational Speed */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Rotational Speed (RPM)</span>
                  <span className="font-mono text-cyan-400 font-bold">{telemetry.rotational_speed_rpm} RPM</span>
                </div>
                <input
                  type="range"
                  min="1100"
                  max="2900"
                  step="10"
                  value={telemetry.rotational_speed_rpm}
                  onChange={(e) =>
                    setTelemetry({ ...telemetry, rotational_speed_rpm: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Torque */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Mechanical Torque (Nm)</span>
                  <span className="font-mono text-cyan-400 font-bold">{telemetry.torque_nm} Nm</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="80"
                  step="0.5"
                  value={telemetry.torque_nm}
                  onChange={(e) =>
                    setTelemetry({ ...telemetry, torque_nm: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Tool Wear */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Cumulative Tool Wear (min)</span>
                  <span className="font-mono text-cyan-400 font-bold">{telemetry.tool_wear_min} min</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="260"
                  step="1"
                  value={telemetry.tool_wear_min}
                  onChange={(e) =>
                    setTelemetry({ ...telemetry, tool_wear_min: parseFloat(e.target.value) })
                  }
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Threshold Override */}
              <div className="pt-3 border-t border-slate-800">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400 font-medium">Decision Threshold (τ)</span>
                  <span className="font-mono text-amber-400 font-bold">{threshold.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full accent-amber-500"
                />
                <span className="text-[10px] text-slate-400">
                  Optimal calibrated threshold: 0.5608 (Maximizes F1 & bounds recall)
                </span>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold text-xs shadow-lg shadow-cyan-600/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
              >
                <Play className={`w-4 h-4 fill-white ${loading ? "animate-spin" : ""}`} />
                <span>{loading ? "Running Neural TreeSHAP Engine..." : "Execute Prediction"}</span>
              </button>
            </form>
          </div>

          {/* Right Column: Prediction Results & SHAP Waterfall (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            {error && (
              <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800 text-xs text-rose-300 flex items-start gap-3">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-rose-200">API Execution Error: </span>
                  {error}
                </div>
              </div>
            )}

            {prediction ? (
              <div className="space-y-6">
                {/* Result Hero Card */}
                <div
                  className={`p-6 rounded-2xl border ${
                    prediction.predicted_label === "FAILURE_IMMINENT"
                      ? "bg-rose-950/20 border-rose-800/80 shadow-lg shadow-rose-950/30"
                      : "bg-emerald-950/20 border-emerald-800/80 shadow-lg shadow-emerald-950/30"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                    <div className="flex items-center gap-3">
                      <span
                        className={`p-2 rounded-xl border ${
                          prediction.predicted_label === "FAILURE_IMMINENT"
                            ? "bg-rose-950 text-rose-400 border-rose-800"
                            : "bg-emerald-950 text-emerald-400 border-emerald-800"
                        }`}
                      >
                        <Activity className="w-5 h-5" />
                      </span>
                      <div>
                        <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                          Classification Outcome
                        </span>
                        <h2 className="text-lg font-bold text-slate-100">
                          {prediction.predicted_label === "FAILURE_IMMINENT"
                            ? "CRITICAL: Machine Failure Imminent"
                            : "NOMINAL: Machine Operating Normally"}
                        </h2>
                      </div>
                    </div>

                    <StatusBadge
                      status={prediction.risk_level as any}
                      size="md"
                    />
                  </div>

                  {/* Failure Probability Bar */}
                  <div className="space-y-2 mt-4">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-400">Failure Risk Probability:</span>
                      <span
                        className={`font-bold text-base ${
                          prediction.failure_probability > 0.5
                            ? "text-rose-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {(prediction.failure_probability * 100).toFixed(2)}%
                      </span>
                    </div>

                    {/* Progress Track */}
                    <div className="w-full h-3 rounded-full bg-slate-900 overflow-hidden relative border border-slate-800">
                      <div
                        className={`h-full transition-all duration-500 rounded-full ${
                          prediction.failure_probability > 0.5
                            ? "bg-gradient-to-r from-amber-500 to-rose-500"
                            : "bg-gradient-to-r from-cyan-500 to-emerald-500"
                        }`}
                        style={{ width: `${Math.min(100, prediction.failure_probability * 100)}%` }}
                      />
                      {/* Threshold marker */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-sm"
                        style={{ left: `${threshold * 100}%` }}
                        title={`Threshold: ${(threshold * 100).toFixed(1)}%`}
                      />
                    </div>

                    <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                      <span>0% Nominal</span>
                      <span className="text-amber-400 font-medium">
                        Active Threshold: {(threshold * 100).toFixed(1)}%
                      </span>
                      <span>100% Critical</span>
                    </div>
                  </div>

                  {/* Latency & Metadata */}
                  <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>Model: {prediction.model_version}</span>
                    <span>Latency: {prediction.latency_ms.toFixed(2)} ms</span>
                  </div>
                </div>

                {/* Operator Diagnostic Translation Alert */}
                {prediction.operator_summary && (
                  <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl">
                    <div className="flex items-center gap-2 mb-2 text-cyan-400">
                      <Info className="w-4 h-4" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                        Operator Guidance & Diagnostic Alert
                      </h3>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed font-mono bg-slate-900 p-3 rounded-xl border border-slate-800/80">
                      {prediction.operator_summary}
                    </p>
                  </div>
                )}

                {/* SHAP Feature Attribution Waterfall */}
                <div className="bg-[#111827] border border-slate-800 p-5 rounded-2xl">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Layers className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                        TreeSHAP Local Feature Contribution
                      </h3>
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">Additive Shapley Values</span>
                  </div>

                  <ShapWaterfallChart
                    escalators={prediction.top_risk_escalators}
                    stabilizers={prediction.top_stabilizers}
                    height={220}
                  />

                  {prediction.non_causal_disclaimer && (
                    <p className="mt-4 text-[10px] text-slate-400 italic border-t border-slate-800/60 pt-2">
                      {prediction.non_causal_disclaimer}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-full min-h-[400px] flex flex-col items-center justify-center p-8 bg-[#111827] border border-dashed border-slate-800 rounded-2xl text-center">
                <Activity className="w-12 h-12 text-slate-600 mb-3" />
                <h3 className="text-sm font-bold text-slate-200">Inference Studio Ready</h3>
                <p className="text-xs text-slate-400 max-w-sm mt-1">
                  Adjust the telemetry sliders on the left or choose a failure preset to run real-time predictive diagnostics.
                </p>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Batch Prediction Tab */
        <div className="space-y-6">
          <div className="bg-[#111827] border border-slate-800 p-6 rounded-2xl">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-100">Batch Fleet Telemetry Evaluation</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Vectorized multi-machine inference via FastAPI <code className="text-cyan-400">/api/v1/predict/batch</code> (Capped at 5,000 records).
                </p>
              </div>

              <button
                onClick={handleRunSampleBatch}
                disabled={batchLoading}
                className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs flex items-center gap-2 shadow-lg shadow-cyan-600/20 transition-all cursor-pointer"
              >
                <Play className={`w-3.5 h-3.5 ${batchLoading ? "animate-spin" : ""}`} />
                <span>{batchLoading ? "Scoring Fleet..." : "Score Sample Fleet (5 Machines)"}</span>
              </button>
            </div>

            {batchError && (
              <div className="mb-4 p-3 rounded-xl bg-rose-950/30 border border-rose-800 text-xs text-rose-300">
                {batchError}
              </div>
            )}

            {batchResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-center">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block">Total Evaluated</span>
                    <span className="text-base font-mono font-bold text-slate-100">{batchResult.total_records}</span>
                  </div>
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-center">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block">Imminent Failures</span>
                    <span className="text-base font-mono font-bold text-rose-400">{batchResult.failure_count}</span>
                  </div>
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-center">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block">Mean Failure Risk</span>
                    <span className="text-base font-mono font-bold text-amber-400">{(batchResult.mean_failure_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-center">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block">Total Latency</span>
                    <span className="text-base font-mono font-bold text-cyan-400">{batchResult.total_latency_ms.toFixed(2)} ms</span>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-xl border border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
                      <tr>
                        <th className="py-2.5 px-3">Record UDI</th>
                        <th className="py-2.5 px-3">Predicted Label</th>
                        <th className="py-2.5 px-3">Risk Assessment</th>
                        <th className="py-2.5 px-3">Failure Probability</th>
                        <th className="py-2.5 px-3">Threshold Used</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {batchResult.predictions.map((p, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/40">
                          <td className="py-2.5 px-3 font-mono text-slate-300">UDI-{p.udi || idx + 1}</td>
                          <td className="py-2.5 px-3">
                            <StatusBadge status={p.predicted_label === "FAILURE_IMMINENT" ? "CRITICAL" : "HEALTHY"} size="sm" label={p.predicted_label} />
                          </td>
                          <td className="py-2.5 px-3 font-semibold text-slate-300">{p.risk_level}</td>
                          <td className="py-2.5 px-3 font-mono font-bold">
                            <span className={p.failure_probability > 0.5 ? "text-rose-400" : "text-emerald-400"}>
                              {(p.failure_probability * 100).toFixed(2)}%
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-400">{p.threshold_used}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
